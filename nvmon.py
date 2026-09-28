#!/usr/bin/env python3
"""nvmon: a compact, btop-style NVIDIA GPU monitor with zero dependencies.

One box per GPU: utilization history on the left (0 % at the bottom, 100 % at
the top) and clock / power / memory / PCIe numbers on the right. It talks to
the NVML library that ships with the NVIDIA driver through ctypes and draws
with plain ANSI escape codes, so a single file runs on any Python >= 3.6.

    nvmon [-i SECONDS]        Ctrl+C to quit
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

INFO_W = 28                          # width of the stats column
INFO_ROWS = 4                        # GPU, MEM, PWR, PCIe
CHROME_W = 7                         # "│ " + " │ " + " │" around graph and stats
MIN_GRAPH_W = 30                     # a narrower graph is not worth a second column
MAX_INNER_H = 8                      # tallest graph: 8 rows x 8 sub-levels = 64 steps
HISTORY = 1024                       # samples kept per GPU; wider than any terminal
BLOCKS = " ▁▂▃▄▅▆▇█"                 # a character cell filled in 1/8 steps


def _fg(n):
    return "\x1b[38;5;{}m".format(n)


def _gradient(stops, steps=101):
    """24-bit colours evenly interpolated through the RGB `stops`."""
    colours = []
    for i in range(steps):
        t = i / (steps - 1) * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        rgb = (round(a + (b - a) * (t - k)) for a, b in zip(stops[k], stops[k + 1]))
        colours.append("\x1b[38;2;{};{};{}m".format(*rgb))
    return colours


# Utilization uses a 101-step truecolor gradient, green -> yellow -> orange -> red;
# everything else sticks to the 256-colour palette.
HEAT = _gradient([(95, 175, 95), (175, 215, 95), (215, 215, 95), (215, 175, 95), (215, 135, 95), (215, 95, 95)])
DIM, PROC = _fg(240), _fg(110)
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
    tx: Optional[int]             # PCIe, KiB/s
    rx: Optional[int]             # PCIe, KiB/s
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


def open_gpus(nv):
    count = c_uint()
    check(nv, nv.nvmlDeviceGetCount_v2(byref(count)))
    gpus = []
    for index in range(count.value):
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
    """Right-hand percentage, coloured by its value."""
    if part is None or not whole:
        return [("   -", "")]
    return [("{:.0f}%".format(100 * part / whole).rjust(4), heat(part / whole))]


def num(value, fmt):
    return "-" if value is None else fmt.format(value)


def rate(kib):
    """KiB/s as a fixed 10-cell string, e.g. '8.61 MiB/s'."""
    if kib is None:
        return "-".rjust(10)
    for unit in ("KiB", "MiB", "GiB"):
        if kib < 999.5 or unit == "GiB":
            break
        kib /= 1024
    digits = 2 if kib < 9.995 else 1 if kib < 99.95 else 0
    return "{:.{}f} {}/s".format(kib, digits, unit).rjust(10)


def info(gpu, height):
    s = gpu.now
    lines = [  # the three percentages share one right-hand column
        row([("GPU", "")], share(s.util, 100)),
        row([("MEM  {} / {} GiB".format(num(s.mem_used, "{:.1f}"), num(s.mem_total, "{:.1f}")), "")],
            share(s.mem_used, s.mem_total)),
        row([("PWR  {} / {} W".format(num(s.power, "{:.0f}"), num(s.power_limit, "{:.0f}")), "")],
            share(s.power, s.power_limit)),
        row([("TX {} RX {}".format(rate(s.tx), rate(s.rx)), "")]),
    ]
    return lines[:height] + [row([])] * (height - len(lines))  # a short panel keeps the top lines


def panel(gpu, width, height):
    graph_w = max(0, width - CHROME_W - INFO_W)
    s = gpu.now
    note = "  ".join(text.format(v) for text, v in (("FAN {}%", s.fan), ("{}°C", s.temp), ("{} MHz", s.clock))
                     if v is not None)
    top = edge(width, "╭╮", [("GPU {}".format(gpu.index), BOLD), ("  " + gpu.name, "")], note)
    procs = []
    for name, mem in s.processes:
        procs += [("  " if procs else "", ""), (name, PROC)] + ([] if mem is None else [(" {:.1f}G".format(mem), DIM)])
    bottom = edge(width, "╰╯", procs)
    body = [[("│ ", DIM)] + g + [(" │ ", DIM)] + i + [(" │", DIM)]
            for g, i in zip(graph(gpu.history, graph_w, height), info(gpu, height))]
    return [top] + body + [bottom]


def edge(width, corners, label, note=""):
    """Top or bottom border, "╭─ label ───── note ─╮"; a long label is cut to fit."""
    tail = ([(" " + note + " ", "")] if note else []) + [("─" + corners[1], DIM)]
    label = clip(label, width - 4 - width_of(tail))  # 4 = "╭─" + a space on each side of the label
    head = [(corners[0] + "─", DIM)] + ([(" ", "")] + label + [(" ", "")] if label else [])
    return head + [("─" * max(0, width - width_of(head) - width_of(tail)), DIM)] + tail


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


def totals(gpus):
    """'8 GPUs  2.1 kW  62 / 637 GiB' over every GPU shown."""
    def total(field):
        return sum(v for v in (getattr(g.now, field) for g in gpus) if v is not None)
    power = total("power")
    watts = "{:.1f} kW".format(power / 1000) if power >= 1000 else "{:.0f} W".format(power)
    return "{} GPU{}  {}  {:.0f} / {:.0f} GiB".format(
        len(gpus), "s" if len(gpus) > 1 else "", watts, total("mem_used"), total("mem_total"))


def header(width, interval, gpus, driver):
    """Top line: name, version, host, local time | totals, driver, refresh interval."""
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime(now)) + ".{:02d}".format(int(now % 1 * 100))
    left = [("nvmon", BOLD), (" " + __version__, DIM), ("  " + socket.gethostname(), ""), ("  " + stamp, "")]
    right = [("refresh ", DIM), ("{:g}s".format(interval), "")]
    # Optional parts, most important first; each is shown only if it still fits beside the left side.
    for part in ([(totals(gpus), ""), ("   ", "")], [(driver, DIM), ("   ", "")] if driver else []):
        if part and width_of(left) + 2 + width_of(part) + width_of(right) <= width:
            right = part + right
    room = width - width_of(right)
    return clip(pad(clip(left, room - 1), room) + right, width)


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
    """Alternate screen, hidden cursor, no auto-wrap; all restored on exit."""

    def __enter__(self):
        self._restore_console = enable_vt()
        self._write("\x1b[?1049h\x1b[?25l\x1b[?7l")
        return self

    def __exit__(self, *exc):
        self._write(RESET + "\x1b[?7h\x1b[?25h\x1b[?1049l")
        self._restore_console()

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


def main():
    parser = argparse.ArgumentParser(prog="nvmon", description="Compact btop-style NVIDIA GPU monitor.")
    parser.add_argument("-i", "--interval", type=float, default=0.5, metavar="SEC",
                        help="seconds between updates (default: 0.5)")
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
        gpus = open_gpus(nv)
        if not gpus:
            sys.exit("nvmon: no accessible NVIDIA GPU")
        driver = versions(nv)
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        with ThreadPoolExecutor(len(gpus)) as pool, Screen() as screen:
            deadline = time.monotonic()
            while True:
                list(pool.map(Gpu.poll, gpus))
                width, height = os.get_terminal_size()
                screen.draw([header(width, args.interval, gpus, driver)] + render(gpus, width, height - 1))
                # Fixed-rate ticks; a late tick restarts the schedule instead of bursting.
                now = time.monotonic()
                deadline = max(deadline + args.interval, now)
                time.sleep(deadline - now)
    except KeyboardInterrupt:
        pass
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))
    finally:
        nv.nvmlShutdown()


if __name__ == "__main__":
    main()
