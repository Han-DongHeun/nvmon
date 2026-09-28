#!/usr/bin/env python3
"""nvmon: a compact, btop-style NVIDIA GPU monitor with zero dependencies.

One box per GPU: utilization history on the left (0 % at the bottom, 100 % at
the top) and clock / power / memory / PCIe numbers on the right. It talks to
the NVML library that ships with the NVIDIA driver through ctypes and draws
with plain ANSI escape codes, so a single file runs on any Python >= 3.6.

    nvmon [-i SECONDS] [-g 0,2,4-7]        q / Esc / Ctrl+C to quit
"""
import argparse
import ctypes
import itertools
import math
import os
import signal
import socket
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_int, c_uint, c_ulonglong, c_void_p
from typing import NamedTuple, Optional

__version__ = "0.1.0"

INFO_W = 32                          # width of the stats column
INFO_ROWS = 5                        # GPU, MEM, CPU->GPU, GPU->CPU, PWR
CHROME_W = 7                         # "│ " + " │ " + " │" around graph and stats
MIN_GRAPH_W = 30                     # a narrower graph is not worth a second column
MAX_INNER_H = 8                      # tallest graph: 8 rows x 8 sub-levels = 64 steps
HISTORY = 1024                       # samples kept per GPU; wider than any terminal
BLOCKS = " ▁▂▃▄▅▆▇█"                 # a character cell filled in 1/8 steps


def _fg(n):
    return "\x1b[38;5;{}m".format(n)


def _rgb(rgb):
    return "\x1b[38;2;{};{};{}m".format(*rgb)


def _gradient(stops):
    """Truecolor codes for 0..100, interpolated through (position, (r, g, b)) `stops`."""
    colours = []
    for i in range(101):
        x = min(max(i, stops[0][0]), stops[-1][0])
        (x0, c0), (x1, c1) = next(pair for pair in zip(stops, stops[1:]) if x <= pair[1][0])
        colours.append(_rgb([round(a + (b - a) * (x - x0) / (x1 - x0)) for a, b in zip(c0, c1)]))
    return colours


GREEN, YELLOW, ORANGE, RED = (95, 175, 95), (215, 215, 95), (215, 135, 95), (215, 95, 95)
# Utilization and shares: green -> yellow -> orange -> red over 0-100 %.
HEAT = _gradient([(0, GREEN), (20, (175, 215, 95)), (40, YELLOW), (60, (215, 175, 95)), (80, ORANGE), (100, RED)])
# Temperature in °C: idle GPUs sit at 30-45, busy ones at 60-80, most throttle from about 85-90.
TEMP = _gradient([(30, (95, 135, 215)), (45, (95, 175, 175)), (60, GREEN), (72, YELLOW), (80, ORANGE), (88, RED)])
# Warnings on the top edge: yellow = worth a look, orange = slowed, red = act.
WARN, SLOW, ALERT = _rgb(YELLOW), _rgb(ORANGE), _rgb(RED)
# Everything else sticks to the 256-colour palette.
DIM, FAINT, PROC = _fg(240), _fg(236), _fg(110)
BOLD, RESET = "\x1b[1m", "\x1b[0m"


def heat(t):
    """Colour for a position t in [0, 1] of the 0-100 % scale."""
    return HEAT[round(min(max(t, 0), 1) * 100)]


# ── NVML (libnvidia-ml / nvml.dll, part of the NVIDIA driver) ───────────────

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS = 0
NVML_PCIE_UTIL_TX_BYTES, NVML_PCIE_UTIL_RX_BYTES = 0, 1
NVML_DEVICE_NAME_BUFFER_SIZE = 96
NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class Memory(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class MemoryV2(ctypes.Structure):
    _fields_ = [("version", c_uint), ("total", c_ulonglong), ("reserved", c_ulonglong),
                ("free", c_ulonglong), ("used", c_ulonglong)]


class TemperatureV1(ctypes.Structure):
    _fields_ = [("version", c_uint), ("sensorType", c_int), ("temperature", c_int)]


class ProcessInfo(ctypes.Structure):  # nvmlProcessInfo_v2_t
    _fields_ = [("pid", c_uint), ("usedGpuMemory", c_ulonglong),
                ("gpuInstanceId", c_uint), ("computeInstanceId", c_uint)]


NVML_VALUE_NOT_AVAILABLE = 2**64 - 1  # usedGpuMemory under Windows WDDM
MAX_PROCESSES = 64


class GpmSupport(ctypes.Structure):  # nvmlGpmSupport_t
    _fields_ = [("version", c_uint), ("isSupportedDevice", c_uint)]


class GpmMetric(ctypes.Structure):  # nvmlGpmMetric_t
    _fields_ = [("metricId", c_uint), ("nvmlReturn", c_int), ("value", ctypes.c_double),
                ("shortName", ctypes.c_char_p), ("longName", ctypes.c_char_p), ("unit", ctypes.c_char_p)]


class GpmMetricsGet(ctypes.Structure):  # nvmlGpmMetricsGet_t, sized for the two metrics we ask for
    _fields_ = [("version", c_uint), ("numMetrics", c_uint), ("sample1", c_void_p), ("sample2", c_void_p),
                ("metrics", GpmMetric * 2)]


NVML_GPM_METRIC_SM_UTIL, NVML_GPM_METRIC_ANY_TENSOR_UTIL = 2, 5

# nvmlClocksEventReason bits that mean "held back", most serious first. The others (idle,
# application clocks, sync boost, display) are normal operation and stay hidden.
SLOWDOWNS = [(0x20 | 0x40, "SLOWED: heat", ALERT),   # software / hardware thermal slowdown
             (0x08 | 0x80, "SLOWED: hw", ALERT),     # hardware slowdown / power brake
             (0x04, "SLOWED: power", SLOW)]          # software power cap

# PCIe payload bandwidth per lane and direction, GB/s, by link generation.
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}

# NVML_STRUCT_VERSION(): struct size with the version number in the top byte.
MEMORY_V2 = ctypes.sizeof(MemoryV2) | 2 << 24
TEMPERATURE_V1 = ctypes.sizeof(TemperatureV1) | 1 << 24


class NvmlError(Exception):
    pass


def load_nvml():
    """Open the driver's NVML library and initialise it."""
    if os.name == "nt":
        paths = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvml.dll"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                              "NVIDIA Corporation", "NVSMI", "nvml.dll")]
    else:
        paths = ["libnvidia-ml.so.1"]
    for path in paths:
        try:
            nv = ctypes.CDLL(path)  # CDLL releases the GIL during calls, so GPUs poll in parallel
        except OSError:
            continue
        nv.nvmlErrorString.restype = ctypes.c_char_p
        check(nv, nv.nvmlInit_v2())
        return nv
    raise NvmlError("NVML library not found - is the NVIDIA driver installed?")


def check(nv, rc):
    if rc != NVML_SUCCESS:
        raise NvmlError(nv.nvmlErrorString(rc).decode())


class Sample(NamedTuple):
    util: Optional[int]           # %
    temp: Optional[int]           # °C
    fan: Optional[int]            # %
    power: Optional[float]        # W
    power_limit: Optional[float]  # W
    clock: Optional[int]          # graphics clock, MHz
    mem_used: Optional[float]     # GiB
    mem_total: Optional[float]    # GiB
    tx: Optional[int]             # PCIe GPU -> CPU, kB/s
    rx: Optional[int]             # PCIe CPU -> GPU, kB/s
    pcie_width: Optional[int]     # current link width (lanes)
    cores: Optional[float]        # % of SMs busy (GPM, Hopper and newer)
    tensor: Optional[float]       # % Tensor Core activity (GPM)
    slowdown: Optional[tuple]     # (label, colour) when the clock is held back
    processes: list               # [(name, GiB or None)], biggest first


class Gpu:
    def __init__(self, nv, index, handle):
        self.nv, self.index, self.handle = nv, index, handle
        name = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
        nv.nvmlDeviceGetName(handle, name, NVML_DEVICE_NAME_BUFFER_SIZE)
        name = name.value.decode("utf-8", "replace")
        self.name = name[len("NVIDIA "):] if name.startswith("NVIDIA ") else name
        self.history = deque(maxlen=HISTORY)
        self.now = None
        self.pcie_max_gen = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        self.pcie_max_width = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        self._gpm = self._gpm_samples()
        self.has_activity = self._gpm is not None

    def _ok(self, fn, *args):
        return getattr(self.nv, fn)(self.handle, *args) == NVML_SUCCESS

    def _uint(self, fn, *args):
        """Scalar query; None when this GPU does not support it (e.g. fan on an H100)."""
        value = c_uint()
        return value.value if self._ok(fn, *args, byref(value)) else None

    def _memory(self):
        """(used, total) bytes. v2 (R510+) leaves out driver-reserved memory, like nvidia-smi."""
        if hasattr(self.nv, "nvmlDeviceGetMemoryInfo_v2"):
            mem = MemoryV2(version=MEMORY_V2)
            if self._ok("nvmlDeviceGetMemoryInfo_v2", byref(mem)):
                return mem.used, mem.total
        mem = Memory()
        return (mem.used, mem.total) if self._ok("nvmlDeviceGetMemoryInfo", byref(mem)) else (None, None)

    def _temperature(self):
        """nvmlDeviceGetTemperature is deprecated; its replacement only exists on R565+ drivers."""
        if hasattr(self.nv, "nvmlDeviceGetTemperatureV"):
            temp = TemperatureV1(version=TEMPERATURE_V1, sensorType=NVML_TEMPERATURE_GPU)
            if self._ok("nvmlDeviceGetTemperatureV", byref(temp)):
                return temp.temperature
        return self._uint("nvmlDeviceGetTemperature", NVML_TEMPERATURE_GPU)

    def _gpm_samples(self):
        """Two GPM sample buffers, or None where GPM is unavailable (it needs Hopper or newer)."""
        support = GpmSupport(version=1)
        if not (hasattr(self.nv, "nvmlGpmQueryDeviceSupport")
                and self._ok("nvmlGpmQueryDeviceSupport", byref(support)) and support.isSupportedDevice):
            return None
        samples = [c_void_p(), c_void_p()]
        if any(self.nv.nvmlGpmSampleAlloc(byref(sample)) != NVML_SUCCESS for sample in samples):
            return None
        # Take the first sample now, on the main thread: NVML crashes when a device's
        # first GPM sample is taken from several threads at once.
        if self.nv.nvmlGpmSampleGet(self.handle, samples[0]) != NVML_SUCCESS:
            return None
        return samples  # [previous, current], kept for the whole run

    def _activity(self):
        """(cores %, tensor %) between this poll and the previous one."""
        if self._gpm is None or self.nv.nvmlGpmSampleGet(self.handle, self._gpm[1]) != NVML_SUCCESS:
            return None, None
        previous, current = self._gpm
        self._gpm.reverse()  # this sample is the baseline for the next poll
        query = GpmMetricsGet(version=1, numMetrics=2, sample1=previous, sample2=current)
        query.metrics[0].metricId, query.metrics[1].metricId = NVML_GPM_METRIC_SM_UTIL, NVML_GPM_METRIC_ANY_TENSOR_UTIL
        if self.nv.nvmlGpmMetricsGet(byref(query)) != NVML_SUCCESS:
            return None, None
        return tuple(None if m.nvmlReturn != NVML_SUCCESS else m.value for m in query.metrics)

    def _slowdown(self):
        """(label, colour) for why the clock is held back, or None."""
        fn = ("nvmlDeviceGetCurrentClocksEventReasons" if hasattr(self.nv, "nvmlDeviceGetCurrentClocksEventReasons")
              else "nvmlDeviceGetCurrentClocksThrottleReasons")  # the pre-R535 name
        reasons = c_ulonglong()
        if not self._ok(fn, byref(reasons)):
            return None
        return next(((label, colour) for bits, label, colour in SLOWDOWNS if reasons.value & bits), None)

    def _processes(self):
        infos, count = (ProcessInfo * MAX_PROCESSES)(), c_uint(MAX_PROCESSES)
        if not (hasattr(self.nv, "nvmlDeviceGetComputeRunningProcesses_v3") and
                self._ok("nvmlDeviceGetComputeRunningProcesses_v3", byref(count), infos)):
            return []
        procs = [(self._process_name(p.pid),
                  None if p.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE else p.usedGpuMemory / 2**30)
                 for p in infos[:count.value]]
        return sorted(procs, key=lambda p: -(p[1] or 0))

    def _process_name(self, pid):
        """Short name; for a Python interpreter, the script or module it runs (train.py, torch.distributed.run)."""
        try:
            with open("/proc/{}/cmdline".format(pid), "rb") as f:
                argv = [a for a in f.read().decode("utf-8", "replace").split("\0") if a]
        except OSError:  # not Linux, or the process already exited
            argv = []
        if not argv:
            name = ctypes.create_string_buffer(256)
            if self.nv.nvmlSystemGetProcessName(pid, name, 256) != NVML_SUCCESS:
                return str(pid)
            argv = [name.value.decode("utf-8", "replace")]
        exe = os.path.basename(argv[0])
        if exe.startswith("python"):
            args = iter(argv[1:])
            for arg in args:
                if arg == "-m":
                    return next(args, exe)
                if not arg.startswith("-"):
                    return os.path.basename(arg)
        return exe

    def poll(self):
        util = Utilization()
        util = util.gpu if self._ok("nvmlDeviceGetUtilizationRates", byref(util)) else None
        used, total = self._memory()
        power = self._uint("nvmlDeviceGetPowerUsage")            # mW
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit")    # mW
        cores, tensor = self._activity()
        self.now = Sample(
            util=util,
            temp=self._temperature(),
            fan=self._uint("nvmlDeviceGetFanSpeed"),
            power=None if power is None else power / 1000,
            power_limit=None if limit is None else limit / 1000,
            clock=self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_GRAPHICS),
            mem_used=None if used is None else used / 2**30,
            mem_total=None if total is None else total / 2**30,
            # Each PCIe query samples a counter for ~20 ms, hence the parallel polling.
            tx=self._uint("nvmlDeviceGetPcieThroughput", NVML_PCIE_UTIL_TX_BYTES),
            rx=self._uint("nvmlDeviceGetPcieThroughput", NVML_PCIE_UTIL_RX_BYTES),
            pcie_width=self._uint("nvmlDeviceGetCurrPcieLinkWidth"),
            cores=cores,
            tensor=tensor,
            slowdown=self._slowdown(),
            processes=self._processes(),
        )
        self.history.append(util or 0)


def versions(nv):
    """'driver 580.173.02  CUDA 13.0'; fixed while the program runs, so read once."""
    driver, cuda, parts = ctypes.create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), c_int(), []
    if nv.nvmlSystemGetDriverVersion(driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE) == NVML_SUCCESS:
        parts.append("driver " + driver.value.decode())
    if nv.nvmlSystemGetCudaDriverVersion_v2(byref(cuda)) == NVML_SUCCESS:  # e.g. 13000 -> 13.0
        parts.append("CUDA {}.{}".format(cuda.value // 1000, cuda.value % 1000 // 10))
    return "  ".join(parts)


def open_gpus(nv, wanted=None):
    """GPUs whose index is in `wanted` (all when None)."""
    count = c_uint()
    check(nv, nv.nvmlDeviceGetCount_v2(byref(count)))
    gpus = []
    for index in range(count.value):
        if wanted is not None and index not in wanted:
            continue
        handle = c_void_p()
        # The count includes GPUs we may not open (cgroups, /dev/nvidiaN permissions): skip those.
        if nv.nvmlDeviceGetHandleByIndex_v2(index, byref(handle)) == NVML_SUCCESS:
            gpus.append(Gpu(nv, index, handle))
    return gpus


# ── rendering ────────────────────────────────────────────────────────────────
# A line is a list of (text, style) segments; every character is one cell wide.

def width_of(line):
    return sum(len(text) for text, _ in line)


def clip(line, width):
    out = []
    for text, style in line:
        if width <= 0:
            break
        out.append((text[:width], style))
        width -= len(text)
    return out


def pad(line, width):
    return line + [(" " * (width - width_of(line)), "")]


def graph(history, width, height):
    """Stacked block chart, newest sample on the right, top row first.

    Each cell takes the colour of the level its top reaches: full cells shade row by row,
    and the ragged top edge shows the exact colour of each value.
    """
    vals = list(history)[-width:] if width > 0 else []
    steps = height * 8
    # Any non-zero value gets at least one sub-level so light load stays visible.
    levels = [0] * (width - len(vals)) + [max(round(v * steps / 100), 1 if v else 0) for v in vals]
    rows = []
    for r in reversed(range(height)):
        fills = [min(8, max(0, lv - r * 8)) for lv in levels]
        # Blank and full cells share the row colour so they join into long runs.
        cells = [(BLOCKS[f], heat((r * 8 + (f or 8)) / steps)) for f in fills]
        rows.append([("".join(text for text, _ in run), style)
                     for style, run in itertools.groupby(cells, key=lambda cell: cell[1])])
    return rows


def row(left, right=()):
    """One stats line: `left`, then the `right` segments flush right; exactly INFO_W cells."""
    room = INFO_W - width_of(right)
    return pad(clip(left, room - 1 if right else room), room) + list(right)


def share(part, whole):
    """A 4-cell percentage, coloured by its value."""
    if part is None or not whole:
        return [("   -", "")]
    return [("{:.0f}%".format(100 * part / whole).rjust(4), heat(part / whole))]


def num(value, fmt):
    return "-" if value is None else fmt.format(value)


def info(gpu, height):
    s = gpu.now
    total, limit = num(s.mem_total, "{:.1f}"), num(s.power_limit, "{:.0f}")
    busy = [("GPU ", "")] + share(s.util, 100)
    if gpu.has_activity:  # "cores" = share of SMs at work, "tensor" = Tensor Core activity
        busy += [("  cores " + share(s.cores, 100)[0][0] + " tensor " + share(s.tensor, 100)[0][0], FAINT)]
    # A link's top speed: its generation's per-lane rate times the lanes it runs on now.
    lanes = s.pcie_width or gpu.pcie_max_width
    top = PCIE_LANE_GBS.get(gpu.pcie_max_gen, 0) * (lanes or 0)
    cap = (" / {:.0f}".format(top), WARN if degraded(gpu) else "") if top else ("", "")

    def link(label, kb):  # e.g. "CPU -> GPU  0.16 / 63 GB/s", like MEM's used/total
        return row([(label + ("-" if kb is None else "{:.2f}".format(kb / 1e6)).rjust(6), ""), cap, (" GB/s", "")])

    lines = [
        row(busy),
        row([("MEM ", "")] + share(s.mem_used, s.mem_total), [("{} / {} GiB".format(num(s.mem_used, "{:.1f}"), total), "")]),
        link("CPU -> GPU ", s.rx),
        link("GPU -> CPU ", s.tx),
        row([("PWR {} / {} W".format(num(s.power, "{:.0f}"), limit), "")]),
    ]
    if height < len(lines):  # a short panel keeps the top lines
        return lines[:height]
    if height > len(lines):  # room to spare: a rule (None) after MEM sets GPU and MEM apart
        lines.insert(2, None)
    # PWR stays on the bottom row; any spare rows go just above it.
    return lines[:-1] + [row([])] * (height - len(lines)) + lines[-1:]


# The rule above the PCIe rows doubles as their title: " ├─ PCIe transfer ────┤".
RULE = [(" ├─ ", DIM), ("PCIe transfer", ""), (" " + "─" * (INFO_W - len("PCIe transfer") - 1) + "┤", DIM)]


def degraded(gpu):
    """True when the PCIe link runs on fewer lanes than card and slot allow, e.g. a loose card."""
    s = gpu.now
    return bool(s.pcie_width and gpu.pcie_max_width and s.pcie_width < gpu.pcie_max_width)


def panel(gpu, width, height):
    graph_w = max(0, width - CHROME_W - INFO_W)
    s = gpu.now
    # Fixed widths keep the right end of the border still while values change.
    parts = [] if s.slowdown is None else [s.slowdown]
    if degraded(gpu):
        parts += [("PCIe DEGRADED: x{} -> x{}".format(gpu.pcie_max_width, s.pcie_width), WARN)]
    parts += [] if s.fan is None else [("FAN {:>3}%".format(s.fan), "")]
    parts += [] if s.temp is None else [("{:>3}°C".format(s.temp), TEMP[min(max(s.temp, 0), 100)])]
    parts += [] if s.clock is None else [("{:>4} MHz".format(s.clock), s.slowdown[1] if s.slowdown else "")]
    top = edge(width, [("GPU {}".format(gpu.index), BOLD), ("  " + gpu.name, "")], parts)
    split = 2 + graph_w + 1  # column of the graph | stats divider
    bottom = bottom_edge(width, split, process_label(s.processes, split - 4))
    body = [[("│ ", DIM)] + g + (RULE if i is None
                                  else [(" │ ", DIM)] + i + [(" │", DIM)])
            for g, i in zip(graph(gpu.history, graph_w, height), info(gpu, height))]
    return [top] + body + [bottom]


def edge(width, label, parts):
    """Top border, "╭─ label ───── parts ─╮", fitted to `width`.

    The GPU number (the label's first segment) always shows: when space runs out the right-hand
    parts go first, from the end (clock, then temperature, ...), so warnings go last. The other
    label segments show whole or not at all, so nothing is left half-written.
    """
    parts = list(parts)
    while True:
        note = [seg for i, part in enumerate(parts) for seg in ([("  ", "")] if i else []) + [part]]
        tail = ([(" ", "")] + note + [(" ", "")] if note else []) + [("─╮", DIM)]
        room = width - 4 - width_of(tail)  # 4 = "╭─" + a space on each side of the label
        if not parts or room >= width_of(label[:1]):
            break
        parts.pop()
    kept = clip(label[:1], room)
    for segment in label[1:]:
        if width_of(kept) + len(segment[0]) > room:
            break
        kept.append(segment)
    label = kept
    head = [("╭─", DIM)] + ([(" ", "")] + label + [(" ", "")] if label else [])
    return head + [("─" * max(0, width - width_of(head) - width_of(tail)), DIM)] + tail


def process_label(processes, room):
    """As many whole "name 1.2G" entries as fit in `room` cells, then "+N" for the rest."""
    label = []
    for i, (name, mem) in enumerate(processes):
        entry = [("  " if label else "", ""), (name, PROC)] + ([] if mem is None else [(" {:.1f}G".format(mem), DIM)])
        rest = len(processes) - i - 1
        if width_of(label + entry) + (len("  +{}".format(rest)) if rest else 0) > room:
            # The first entry is always shown, cut if need be; later ones collapse into the count.
            return clip(entry, room) if not label else label + [("  +{}".format(rest + 1), DIM)]
        label += entry
    return label


def bottom_edge(width, split, label):
    """Bottom border with `label` flush right against column `split`, the graph | stats divider."""
    label = clip(label, split - 4)  # keep "╰─" and a space on each side
    middle = [(" ", "")] + label + [(" ", "")] if label else []
    return ([("╰" + "─" * max(0, split - 1 - width_of(middle)), DIM)] + middle
            + [("─" * max(0, width - split - 1) + "╯", DIM)])


def render(gpus, width, height):
    """The whole screen: exactly `height` lines of exactly `width` cells."""
    # Two columns only when one column cannot show every GPU at full height.
    two_fit = width // 2 >= CHROME_W + INFO_W + MIN_GRAPH_W
    cols = 2 if two_fit and len(gpus) * (INFO_ROWS + 2) > height else 1
    rows = math.ceil(len(gpus) / cols)
    inner = max(1, min(MAX_INNER_H, height // rows - 2))
    lines = []
    for r in range(rows):
        panels = [panel(g, width // cols, inner) for g in gpus[r * cols:(r + 1) * cols]]
        lines += [[seg for part in parts for seg in part] for parts in zip(*panels)]
    lines = [pad(clip(line, width), width) for line in lines[:height]]
    return lines + [[(" " * width, "")]] * (height - len(lines))


def spread(width, left, right):
    """`left`, then `right` flush right: one line of exactly `width` cells."""
    room = width - width_of(right)
    return clip(pad(clip(left, room - 1), room) + right, width)


def header(width, interval, driver):
    """Top line: name, version, host, local time | driver, refresh interval."""
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime(now)) + ".{:02d}".format(int(now % 1 * 100))
    left = [("nvmon", BOLD), (" " + __version__, DIM), ("  " + socket.gethostname(), ""), ("  " + stamp, "")]
    right = [("refresh ", DIM), ("{:g}s".format(interval), "")]
    if driver and width_of(left) + 2 + len(driver) + 3 + width_of(right) <= width:  # dropped when narrow
        right = [(driver + "   ", DIM)] + right
    return spread(width, left, right)


def footer(width):
    """Bottom line: how to quit."""
    return spread(width, [], [("Esc / q quit", DIM)])


def paint(lines):
    """Join lines into one string, sending a style code only where the style changes."""
    out, current = [], ""
    for i, line in enumerate(lines):
        if i:
            out.append("\r\n")
        for text, style in line:
            if style != current:
                # One colour replaces another directly; anything else (bold, plain) needs a reset first.
                colours = style.startswith("\x1b[38") and current.startswith("\x1b[38")
                out.append(style if colours else RESET + style)
                current = style
            out.append(text)
    return "".join(out) + RESET


# ── terminal ─────────────────────────────────────────────────────────────────

class Screen:
    """Alternate screen, hidden cursor, no auto-wrap, unbuffered keys; all restored on exit."""

    def __enter__(self):
        self._keys = sys.stdin.isatty()
        self._restore = [enable_vt(), raw_keys() if self._keys else (lambda: None)]
        self._write("\x1b[?1049h\x1b[?25l\x1b[?7l")
        return self

    def __exit__(self, *exc):
        self._write(RESET + "\x1b[?7h\x1b[?25h\x1b[?1049l")
        for undo in self._restore:
            undo()

    def wait(self, seconds):
        """Sleep for `seconds`; returns True early if q or Esc is pressed."""
        end = time.monotonic() + seconds
        if not self._keys:
            time.sleep(seconds)
            return False
        if os.name == "nt":
            import msvcrt
            while True:
                while msvcrt.kbhit():
                    key = msvcrt.getwch()
                    if key in ("\x00", "\xe0"):  # arrows and function keys come as two characters
                        msvcrt.getwch()
                    elif key in ("q", "Q", "\x1b"):
                        return True
                left = end - time.monotonic()
                if left <= 0:
                    return False
                time.sleep(min(left, 0.05))
        import select
        fd = sys.stdin.fileno()
        while True:
            left = end - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                return False
            keys = os.read(fd, 64)
            # A lone ESC is the Esc key; arrows and the like arrive as ESC [ ... sequences.
            if b"q" in keys or b"Q" in keys or keys == b"\x1b":
                return True
            if not keys:  # stdin closed: nothing more to read, just sleep
                time.sleep(max(0, end - time.monotonic()))
                return False

    def draw(self, lines):
        # 2026 = synchronized output: terminals that know it swap the frame in at once.
        self._write("\x1b[?2026h\x1b[H" + paint(lines) + "\x1b[?2026l")

    @staticmethod
    def _write(text):
        # Raw UTF-8 bytes: works even where Python picked an ASCII stdout (LANG=C on 3.6).
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()


def enable_vt():
    """Make the Windows console interpret ANSI codes; returns a function that undoes it."""
    if os.name != "nt":
        return lambda: None
    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = c_void_p
    handle = c_void_p(kernel32.GetStdHandle(-11))  # STD_OUTPUT_HANDLE
    mode = ctypes.c_ulong()
    if not kernel32.GetConsoleMode(handle, byref(mode)):
        return lambda: None
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN (VT-style end of line)
    kernel32.SetConsoleMode(handle, mode.value | 0x0004 | 0x0008)
    return lambda: kernel32.SetConsoleMode(handle, mode.value)


def raw_keys():
    """Deliver key presses at once and without echo; returns a function that undoes it."""
    if os.name == "nt":
        return lambda: None  # msvcrt already reads single keys without echo
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # Ctrl+C keeps working: cbreak leaves signal keys alone
    return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def gpu_list(spec):
    """argparse type: '0,2,4-7' -> {0, 2, 4, 5, 6, 7}."""
    picked = set()
    try:
        for part in spec.split(","):
            first, _, last = part.strip().partition("-")
            picked.update(range(int(first), int(last or first) + 1))
    except ValueError:
        picked = set()
    if not picked:
        raise argparse.ArgumentTypeError("expected GPU numbers such as 0,2,4-7, got {!r}".format(spec))
    return picked


def main():
    parser = argparse.ArgumentParser(prog="nvmon", description="Compact btop-style NVIDIA GPU monitor.")
    parser.add_argument("-i", "--interval", type=float, default=0.5, metavar="SEC",
                        help="seconds between updates (default: 0.5)")
    parser.add_argument("-g", "--gpus", type=gpu_list, metavar="LIST",
                        help="only these GPUs, e.g. 0,2,4-7 (default: all)")
    parser.add_argument("-V", "--version", action="version", version="nvmon " + __version__)
    args = parser.parse_args()
    if not 0 < args.interval < math.inf:  # also rejects NaN
        parser.error("--interval must be a positive number of seconds")
    if not sys.stdout.isatty():
        sys.exit("nvmon: output is not a terminal (over ssh, use: ssh -t HOST nvmon)")

    try:
        nv = load_nvml()
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))
    try:
        gpus = open_gpus(nv, args.gpus)
        missing = sorted((args.gpus or set()) - {g.index for g in gpus})
        if missing:
            sys.exit("nvmon: GPU {} not found or not accessible".format(", ".join(map(str, missing))))
        if not gpus:
            sys.exit("nvmon: no accessible NVIDIA GPU")
        driver = versions(nv)
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        with ThreadPoolExecutor(len(gpus)) as pool, Screen() as screen:
            deadline = time.monotonic()
            while True:
                list(pool.map(Gpu.poll, gpus))
                width, height = os.get_terminal_size()
                screen.draw([header(width, args.interval, driver)] + render(gpus, width, height - 2)
                            + [footer(width)])
                # Fixed-rate ticks; a late tick restarts the schedule instead of bursting.
                now = time.monotonic()
                deadline = max(deadline + args.interval, now)
                if screen.wait(deadline - now):
                    break
    except KeyboardInterrupt:
        pass
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
