#!/usr/bin/env python3
"""nvmon: a fancy NVIDIA GPU monitor for the terminal. Single Python file, no dependencies.

One box per GPU: utilization history on the left (0 % at the bottom, 100 % at
the top); utilization, memory and PCIe traffic on the right; name, power,
warnings, temperature, fan and clock on the top edge; processes on the bottom
edge. It talks to the NVML library that ships with the NVIDIA driver through
ctypes and draws with plain ANSI escape codes, so a single file runs on any
Python >= 3.6. At start it asks PyPI whether a newer release is out
(NVMON_NO_UPDATE_CHECK=1 turns that off).

    nvmon [-i SECONDS] [-g 0,2,4-7]        q / Esc / Ctrl+C to quit
"""
import argparse
import ctypes
import itertools
import math
import operator
import os
import re
import signal
import socket
import sys
import threading
import time
from collections import deque
from ctypes import byref, c_int, c_uint, c_ulonglong, c_void_p
from functools import lru_cache
from typing import NamedTuple, Optional

__version__ = "0.2.1+dev"  # "+dev": work past this release; the release commit sets the next number
REPO = "https://github.com/Han-DongHeun/nvmon"
RELEASES = REPO + "/releases"                 # release notes, and nvmon.py for those who copy the file
PYPI_JSON = "https://pypi.org/pypi/nvmon/json"

INFO_W = 32                          # width of the stats column
CHROME_W = 7                         # "│ " + " │ " + " │" around graph and stats
MIN_GRAPH_W = 30                     # a narrower graph is not worth a second column
MAX_INNER_H = 6                      # tallest panel: the full stats column (graph: 6 x 8 levels)
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
DIM, FAINT, PROC = _fg(240), _fg(237), _fg(110)
NEWS = PROC  # a newer release is news, not trouble: the process names' calm blue, not warning yellow
BOLD, RESET = "\x1b[1m", "\x1b[0m"


def heat(t):
    """Colour for a position t in [0, 1] of the 0-100 % scale."""
    return HEAT[round(min(max(t, 0), 1) * 100)]


# ── NVML (libnvidia-ml / nvml.dll, part of the NVIDIA driver) ───────────────

NVML_SUCCESS = 0
# Failures that hold for as long as we run: the GPU, or this driver version, lacks the query.
NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_FUNCTION_NOT_FOUND, NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 3, 13, 25
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


# The GPM metrics we read: SM busy %, Tensor Core activity %, PCIe MiB/s from and to the GPU.
GPM_METRICS = (2, 5, 20, 21)


class GpmMetricsGet(ctypes.Structure):  # nvmlGpmMetricsGet_t, sized for GPM_METRICS
    _fields_ = [("version", c_uint), ("numMetrics", c_uint), ("sample1", c_void_p), ("sample2", c_void_p),
                ("metrics", GpmMetric * len(GPM_METRICS))]

# nvmlClocksEventReason bits that mean "held back", most serious first. The others (idle,
# application clocks, sync boost, display) are normal operation and stay hidden.
SLOWDOWNS = [(0x20 | 0x40, "SLOWED: too hot", ALERT),   # software / hardware thermal slowdown
             (0x08 | 0x80, "SLOWED: hw brake", ALERT),     # hardware slowdown / power brake
             (0x04, "SLOWED: power cap", SLOW)]          # software power cap

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
            nv = ctypes.CDLL(path)  # CDLL releases the GIL during calls: a slow one only holds up the Poller
        except OSError:
            continue
        nv.nvmlErrorString.restype = ctypes.c_char_p
        check(nv, nv.nvmlInit_v2())
        return nv
    raise NvmlError("NVML library not found - is the NVIDIA driver installed?")


def check(nv, rc):
    if rc != NVML_SUCCESS:
        raise NvmlError(nv.nvmlErrorString(rc).decode())


class Process(NamedTuple):
    name: str                     # script name for Python, else the executable
    mem: Optional[float]          # GPU memory, GiB
    owner: Optional[str]          # account; None for our own processes
    env: Optional[str]            # conda environment
    started: Optional[float]      # start time, seconds since the epoch


class Sample(NamedTuple):
    util: Optional[int]           # %
    temp: Optional[int]           # °C
    fan: Optional[int]            # %
    power: Optional[float]        # W
    power_limit: Optional[float]  # W
    clock: Optional[int]          # graphics clock, MHz
    mem_used: Optional[float]     # GiB
    mem_total: Optional[float]    # GiB
    tx: Optional[float]           # PCIe GPU -> CPU, bytes/s
    rx: Optional[float]           # PCIe CPU -> GPU, bytes/s
    pcie_width: Optional[int]     # current link width (lanes)
    cores: Optional[float]        # % of SMs busy (GPM, Hopper and newer)
    tensor: Optional[float]       # % Tensor Core activity (GPM)
    slowdown: Optional[tuple]     # (label, colour) when the clock is held back
    processes: list               # [Process]


class Gpu:
    def __init__(self, nv, index, handle):
        self.nv, self.index, self.handle = nv, index, handle
        self._unsupported = set()  # queries this GPU or driver cannot answer, see _ok
        name = ctypes.create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
        nv.nvmlDeviceGetName(handle, name, NVML_DEVICE_NAME_BUFFER_SIZE)
        name = name.value.decode("utf-8", "replace")
        self.name = name[len("NVIDIA "):] if name.startswith("NVIDIA ") else name
        self.history = deque(maxlen=HISTORY)
        self.sample = None  # the newest reading, written by poll()
        self.now = None     # the reading on screen, taken from `sample` by frame()
        self.pcie_max_gen = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        self.pcie_max_width = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        self._gpm = self._gpm_samples()
        self.has_activity = self._gpm is not None
        self._facts = {}  # pid -> (name, owner, env, started): fixed for a process's life, so read once
        self._listed, self._listed_mem = [], None  # process list, and the memory in use when it was read

    def _ok(self, fn, *args):
        """Run the device query `fn`; True on success. A query this GPU or driver cannot answer is not
        asked again: the answer costs time too (0.3 ms of CPU for a fanless H100 to say it has no fan)."""
        if fn in self._unsupported:
            return False
        query = getattr(self.nv, fn, None)  # None: a driver older than this function
        rc = query(self.handle, *args) if query else NVML_ERROR_FUNCTION_NOT_FOUND
        if rc in (NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_FUNCTION_NOT_FOUND, NVML_ERROR_ARGUMENT_VERSION_MISMATCH):
            self._unsupported.add(fn)
        return rc == NVML_SUCCESS

    def _uint(self, fn, *args):
        """Scalar query; None when this GPU does not support it (e.g. fan on an H100)."""
        value = c_uint()
        return value.value if self._ok(fn, *args, byref(value)) else None

    def _memory(self):
        """(used, total) bytes. v2 (R510+) leaves out driver-reserved memory, like nvidia-smi."""
        mem = MemoryV2(version=MEMORY_V2)
        if self._ok("nvmlDeviceGetMemoryInfo_v2", byref(mem)):
            return mem.used, mem.total
        mem = Memory()
        return (mem.used, mem.total) if self._ok("nvmlDeviceGetMemoryInfo", byref(mem)) else (None, None)

    def _temperature(self):
        """nvmlDeviceGetTemperature is deprecated; its replacement only exists on R565+ drivers."""
        temp = TemperatureV1(version=TEMPERATURE_V1, sensorType=NVML_TEMPERATURE_GPU)
        if self._ok("nvmlDeviceGetTemperatureV", byref(temp)):
            return temp.temperature
        return self._uint("nvmlDeviceGetTemperature", NVML_TEMPERATURE_GPU)

    def _gpm_samples(self):
        """Two GPM sample buffers, or None where GPM is unavailable (it needs Hopper or newer)."""
        support = GpmSupport(version=1)
        if not (self._ok("nvmlGpmQueryDeviceSupport", byref(support)) and support.isSupportedDevice):
            return None
        samples = [c_void_p(), c_void_p()]
        if any(self.nv.nvmlGpmSampleAlloc(byref(sample)) != NVML_SUCCESS for sample in samples):
            return None
        # The first sample, as the baseline for the first poll. (NVML crashes when a device's
        # first GPM samples are taken from several threads at once.)
        if not self._ok("nvmlGpmSampleGet", samples[0]):
            return None
        return samples  # [previous, current], kept for the whole run

    def _activity(self):
        """GPM_METRICS values between this poll and the previous one (Nones without GPM)."""
        none = (None,) * len(GPM_METRICS)
        if self._gpm is None or not self._ok("nvmlGpmSampleGet", self._gpm[1]):
            return none
        previous, current = self._gpm
        self._gpm.reverse()  # this sample is the baseline for the next poll
        query = GpmMetricsGet(version=1, numMetrics=len(GPM_METRICS), sample1=previous, sample2=current)
        for metric, metric_id in zip(query.metrics, GPM_METRICS):
            metric.metricId = metric_id
        if self.nv.nvmlGpmMetricsGet(byref(query)) != NVML_SUCCESS:
            return none
        return tuple(None if m.nvmlReturn != NVML_SUCCESS else m.value for m in query.metrics)

    def _pcie(self, counter):
        """PCIe bytes/s; this query samples a counter for 20 ms, so it is only a fallback for GPM."""
        kb = self._uint("nvmlDeviceGetPcieThroughput", counter)
        return None if kb is None else kb * 1024

    def _slowdown(self):
        """(label, colour) for why the clock is held back, or None."""
        reasons = c_ulonglong()
        if not (self._ok("nvmlDeviceGetCurrentClocksEventReasons", byref(reasons))
                or self._ok("nvmlDeviceGetCurrentClocksThrottleReasons", byref(reasons))):  # the pre-R535 name
            return None
        return next(((label, colour) for bits, label, colour in SLOWDOWNS if reasons.value & bits), None)

    def _processes(self):
        infos, count = (ProcessInfo * MAX_PROCESSES)(), c_uint(MAX_PROCESSES)
        if not self._ok("nvmlDeviceGetComputeRunningProcesses_v3", byref(count), infos):
            return []
        running = infos[:count.value]
        self._facts = {p.pid: self._facts.get(p.pid) or (self._process_name(p.pid),) + process_facts(p.pid)
                       for p in running}
        procs = []
        for p in running:
            name, owner, env, started = self._facts[p.pid]
            mem = None if p.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE else p.usedGpuMemory / 2**30
            procs.append(Process(name, mem, owner, env, started))
        return procs

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
        """Read everything shown into `sample`."""
        util = Utilization()
        util = util.gpu if self._ok("nvmlDeviceGetUtilizationRates", byref(util)) else None
        used, total = self._memory()
        # The process list is the costliest query on a busy GPU. Processes only come and go with memory
        # being allocated or freed, so the list is read again only when the memory in use has changed.
        if used is None or used != self._listed_mem:
            self._listed, self._listed_mem = self._processes(), used
        power = self._uint("nvmlDeviceGetPowerUsage")            # mW
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit")    # mW
        cores, tensor, gpm_tx, gpm_rx = self._activity()
        self.sample = Sample(
            util=util,
            temp=self._temperature(),
            fan=self._uint("nvmlDeviceGetFanSpeed"),
            power=None if power is None else power / 1000,
            power_limit=None if limit is None else limit / 1000,
            clock=self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_GRAPHICS),
            mem_used=None if used is None else used / 2**30,
            mem_total=None if total is None else total / 2**30,
            tx=self._pcie(NVML_PCIE_UTIL_TX_BYTES) if gpm_tx is None else gpm_tx * 2**20,
            rx=self._pcie(NVML_PCIE_UTIL_RX_BYTES) if gpm_rx is None else gpm_rx * 2**20,
            pcie_width=self._uint("nvmlDeviceGetCurrPcieLinkWidth"),
            cores=cores,
            tensor=tensor,
            slowdown=self._slowdown(),
            processes=self._listed,
        )

    def frame(self):
        """Put the newest reading on screen for the next frame; the graph gains one column per frame."""
        self.now = self.sample
        self.history.append(self.now.util or 0)


class Poller(threading.Thread):
    """Polls every GPU on a thread of its own, so a busy driver delays the numbers, not the screen.

    The driver serves one program at a time: while someone runs nvidia-smi (half a second for eight
    H100s), a query can wait 0.4 s, and polling on the main thread made the screen stutter.
    """

    def __init__(self, gpus):
        super().__init__(daemon=True)  # a driver call that never returns does not keep nvmon open
        self.gpus, self.error, self.started = gpus, None, None
        self.wanted, self.done = threading.Event(), threading.Event()
        self.done.set()
        self.start()

    def run(self):
        while True:
            self.wanted.wait()
            self.wanted.clear()
            try:
                for gpu in self.gpus:
                    gpu.poll()
            except Exception as e:  # a bug: refresh() raises it rather than leaving the numbers frozen
                self.error = e
            self.done.set()

    def refresh(self):
        """Start a round of polls, unless the last one is still running."""
        if self.error:
            raise self.error
        if self.done.is_set():
            self.done.clear()
            self.started = time.monotonic()
            self.wanted.set()

    def waiting(self):
        """How long the running round has waited on the driver so far; 0 when none is running."""
        return 0 if self.done.is_set() else time.monotonic() - self.started


def versions(nv):
    """'driver 580.173.02  CUDA 13.0'; fixed while the program runs, so read once."""
    driver, cuda, parts = ctypes.create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), c_int(), []
    if nv.nvmlSystemGetDriverVersion(driver, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE) == NVML_SUCCESS:
        parts.append("driver " + driver.value.decode())
    if nv.nvmlSystemGetCudaDriverVersion_v2(byref(cuda)) == NVML_SUCCESS:  # e.g. 13000 -> 13.0
        parts.append("CUDA {}.{}".format(cuda.value // 1000, cuda.value % 1000 // 10))
    return "  ".join(parts)


def process_facts(pid):
    """(owner, conda environment, start time) of `pid` from /proc; Nones where unknown (e.g. Windows)."""
    return process_owner(pid), conda_env(pid), start_time(pid)


def conda_env(pid):
    """Name of the conda environment whose Python runs `pid`; None outside conda."""
    try:
        exe = os.readlink("/proc/{}/exe".format(pid))
    except OSError:
        return None
    return conda_name(os.path.dirname(os.path.dirname(exe)))  # .../envs/NAME/bin/python


def conda_name(prefix):
    """The conda environment installed at `prefix`: ".../envs/NAME" -> NAME, the conda root itself
    -> "base"; None when `prefix` is not a conda environment."""
    if not os.path.isdir(os.path.join(prefix, "conda-meta")):
        return None
    return os.path.basename(prefix) if os.path.basename(os.path.dirname(prefix)) == "envs" else "base"


def start_time(pid):
    """When `pid` started, in seconds since the epoch."""
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            ticks = int(f.read().rsplit(")", 1)[1].split()[19])  # field 22, counted after "(name)"
        with open("/proc/stat") as f:
            boot = next(int(line.split()[1]) for line in f if line.startswith("btime"))
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def process_owner(pid):
    """Account running `pid`; None for our own processes or where there is no /proc (Windows)."""
    try:
        import pwd
        with open("/proc/{}/status".format(pid)) as f:
            uid = next(int(line.split()[1]) for line in f if line.startswith("Uid:"))
    except (ImportError, OSError, StopIteration):
        return None
    if uid == os.getuid():
        return None
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:  # no passwd entry, e.g. inside a container
        return str(uid)


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


# ── new releases ─────────────────────────────────────────────────────────────

class UpdateCheck(threading.Thread):
    """Asks PyPI for the latest release in the background. Afterwards `newer` is (version, how to
    update) when that release is newer than this one; it stays None otherwise, and on any failure."""

    def __init__(self):
        super().__init__(daemon=True)  # a slow network never holds up quitting
        self.newer = None

    def run(self):
        try:
            import json
            import urllib.request
            with urllib.request.urlopen(PYPI_JSON, timeout=5) as response:
                latest = json.load(response)["info"]["version"]
            ours = tuple(map(int, __version__.split("+")[0].split(".")))  # a "+dev" build counts as its release
            # Nothing from the network reaches the screen but a plain release number such as 0.2.2.
            if re.fullmatch(r"[0-9]+(\.[0-9]+)*", latest) and tuple(map(int, latest.split("."))) > ours:
                self.newer = latest, update_command()
        except Exception:  # offline, firewalled, a proxy in the way, an odd answer: no notice, no noise
            pass


def update_command():
    """How to update this copy of nvmon, judged from where it runs; None for a copied nvmon.py."""
    prefix = sys.prefix

    def has(name):
        return os.path.exists(os.path.join(prefix, name))
    if has("uv-receipt.toml"):  # `uv tool install`; `uvx nvmon` runs that same install too
        return "uv tool upgrade nvmon"
    if has("pipx_metadata.json"):
        return "pipx upgrade nvmon"
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) not in ("site-packages", "dist-packages"):
        return None
    if os.path.basename(os.path.dirname(prefix)).startswith("archive-v"):  # uvx: a throwaway env in uv's cache
        return "uvx nvmon@latest"
    env = conda_name(prefix)
    if env:
        return "in env {}: pip install -U nvmon".format(env)
    try:
        with open(os.path.join(prefix, "pyvenv.cfg")) as f:
            made_by_uv = any(line.startswith("uv =") for line in f)
    except OSError:  # not a virtual environment
        made_by_uv = False
    return "uv pip install -U nvmon" if made_by_uv else "pip install -U nvmon"  # uv's venvs have no pip


# ── rendering ────────────────────────────────────────────────────────────────
# A line is a list of (text, style) segments; every character is one cell wide.

# A horizontal rule across the stats column, joined to the borders.
RULE = [(" ├" + "─" * (INFO_W + 2) + "┤", DIM)]


def width_of(line):
    return sum(len(text) for text, _ in line)


def clip(line, width):
    """`line` cut to at most `width` cells."""
    out = []
    for text, style in line:
        if width <= 0:
            break
        out.append((text[:width], style))
        width -= len(text)
    return out


def pad(line, width):
    """`line` padded with spaces to `width` cells."""
    return line + [(" " * (width - width_of(line)), "")]


@lru_cache(maxsize=None)
def graph_row(r, height):
    """The (block, colour) cell that row `r` (0 = bottom) of a `height`-row graph shows for each
    level 0 .. 8 * height.

    Each cell takes the colour of the level its top reaches: full cells shade row by row,
    and the ragged top edge shows the exact colour of each value.
    """
    steps = height * 8
    fills = (min(8, max(0, level - r * 8)) for level in range(steps + 1))
    # Blank and full cells share the row colour so they join into long runs.
    return [(BLOCKS[f], heat((r * 8 + (f or 8)) / steps)) for f in fills]


def graph(history, width, height):
    """Stacked block chart, newest sample on the right, top row first."""
    vals = list(history)[-width:] if width > 0 else []
    steps = height * 8
    # Any non-zero value gets at least one sub-level so light load stays visible.
    levels = [0] * (width - len(vals)) + [max(round(v * steps / 100), 1 if v else 0) for v in vals]
    text_of, style_of = operator.itemgetter(0), operator.itemgetter(1)
    rows = []
    for r in reversed(range(height)):
        cells = map(graph_row(r, height).__getitem__, levels)
        rows.append([("".join(map(text_of, run)), style) for style, run in itertools.groupby(cells, key=style_of)])
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
    total = num(s.mem_total, "{:.1f}")
    busy = [("GPU ", "")] + share(s.util, 100)
    if gpu.has_activity:  # "cores" = share of SMs at work, "tensor" = Tensor Core activity
        busy += [("  cores " + share(s.cores, 100)[0][0] + " tensor " + share(s.tensor, 100)[0][0], FAINT)]
    # A link's top speed: its generation's per-lane rate times the lanes it runs on now.
    lanes = s.pcie_width or gpu.pcie_max_width
    top = PCIE_LANE_GBS.get(gpu.pcie_max_gen, 0) * (lanes or 0)
    cap = (" / {:.0f}".format(top), WARN if degraded(gpu) else "") if top else ("", "")

    def link(label, rate):  # " CPU -> GPU ... 0.16 / 63 GB/s", flush right like MEM's used / total
        return row([(label, "")], [("-" if rate is None else "{:.2f}".format(rate / 1e9), ""), cap, (" GB/s", "")])

    gpu_row = row(busy)
    mem_row = row([("MEM ", "")] + share(s.mem_used, s.mem_total),
                  [("{} / {} GiB".format(num(s.mem_used, "{:.1f}"), total), "")])
    title, to_gpu, to_cpu = row([("PCIe transfer", "")]), link(" CPU -> GPU", s.rx), link(" GPU -> CPU", s.tx)
    # The rule (None) after MEM and the PCIe title only appear when there is room.
    if height >= 6:
        return [gpu_row, mem_row, None, title, to_gpu, to_cpu] + [row([])] * (height - 6)
    if height == 5:
        return [gpu_row, mem_row, title, to_gpu, to_cpu]
    return [gpu_row, mem_row, to_gpu, to_cpu][:height]


def degraded(gpu):
    """True when the PCIe link runs on fewer lanes than card and slot allow, e.g. a loose card."""
    s = gpu.now
    return bool(s.pcie_width and gpu.pcie_max_width and s.pcie_width < gpu.pcie_max_width)


def panel(gpu, width, height):
    graph_w = max(0, width - CHROME_W - INFO_W)
    s = gpu.now
    split = 2 + graph_w + 1  # column of the graph | stats divider
    # Both sides as (segments, rank); when space runs out the lowest rank goes first:
    # fan, name, power, clock, temperature, PCIe warning, slowdown warning. The GPU number stays.
    # Fixed widths keep things from shifting.
    limit = num(s.power_limit, "{:.0f}")  # power padded to the limit's width
    label = [([("GPU {}".format(gpu.index), BOLD)], None), ([("  " + gpu.name, "")], 2),
             ([("  │ ", DIM), ("{} / {} W".format(num(s.power, "{:.0f}").rjust(len(limit)), limit), "")], 3)]
    parts = [] if s.slowdown is None else [([s.slowdown], 9)]
    if degraded(gpu):
        parts += [([("PCIe DEGRADED: x{} -> x{}".format(gpu.pcie_max_width, s.pcie_width), WARN)], 8)]
    parts += [] if s.temp is None else [([("{:>3}°C".format(s.temp), TEMP[min(max(s.temp, 0), 100)])], 5)]
    parts += [] if s.fan is None else [([("FAN {:>3}%".format(s.fan), "")], 1)]
    parts += [] if s.clock is None else [([("{:>4} MHz".format(s.clock), s.slowdown[1] if s.slowdown else "")], 4)]
    top = edge(width, label, parts)
    bottom = bottom_edge(width, split, process_label(s.processes, split - 4))
    body = [[("│ ", DIM)] + g + (RULE if i is None
                                  else [(" │ ", DIM)] + i + [(" │", DIM)])
            for g, i in zip(graph(gpu.history, graph_w, height), info(gpu, height))]
    return [top] + body + [bottom]


def edge(width, label, parts):
    """Top border, "╭─ label ───── parts ─╮", fitted to `width`.

    `label` and `parts` are (segments, rank) groups. While they do not fit, the group with the
    lowest rank goes, whichever side it is on, keeping the others in order; a group shows whole
    or not at all. The first label group (rank None: the GPU number) always stays, cut if need be.
    """
    label, parts = list(label), list(parts)
    while True:
        note = [seg for i, (group, _) in enumerate(parts) for seg in ([("  ", "")] if i else []) + group]
        tail = ([(" ", "")] + note + [(" ", "")] if note else []) + [("─╮", DIM)]
        kept = [seg for group, _ in label for seg in group]
        if width_of(kept) + width_of(tail) + 5 <= width:  # 5 = "╭─", a space either side of the label, one "─"
            break
        droppable = [(rank, side, group) for side in (label, parts) for group, rank in side if rank is not None]
        if not droppable:
            break
        _, side, group = min(droppable, key=lambda item: item[0])
        side.remove(next(item for item in side if item[0] is group))
    kept = clip(kept, width - 5 - width_of(tail))
    head = [("╭─", DIM)] + ([(" ", "")] + kept + [(" ", "")] if kept else [])
    return head + [("─" * max(0, width - width_of(head) - width_of(tail)), DIM)] + tail


def elapsed(started):
    """Time since `started`, compact: 42s, 5m, 2h13m, 1d4h."""
    s = max(0, int(time.time() - started))
    if s < 60:
        return "{}s".format(s)
    if s < 3600:
        return "{}m".format(s // 60)
    if s < 86400:
        return "{}h{}m".format(s // 3600, s % 3600 // 60)
    return "{}d{}h".format(s // 86400, s % 86400 // 3600)


def process_label(processes, room):
    """Processes grouped by owner and conda environment, groups and processes by memory, biggest
    first: "migi: train.py 2h13m 15.1G · eval.py 5m 0.5G   kim/torch: a.py 1d4h 9.0G" (our own
    processes carry no owner). As many whole entries as fit in `room` cells, then "+N" for the rest."""
    def mem(process):
        return process.mem or 0
    groups = {}
    for process in sorted(processes, key=mem, reverse=True):
        groups.setdefault((process.owner, process.env), []).append(process)
    entries = []
    for (owner, env), group in sorted(groups.items(), key=lambda item: sum(map(mem, item[1])), reverse=True):
        tag = "/".join(part for part in (owner, env) if part)
        for j, process in enumerate(group):
            gap = [] if not entries else [(" · ", DIM)] if j else [("   ", "")]
            details = [elapsed(process.started)] if process.started is not None else []
            details += [] if process.mem is None else ["{:.1f}G".format(process.mem)]
            entries.append(gap + ([(tag + ": ", DIM)] if tag and not j else []) + [(process.name, PROC)]
                           + ([(" " + " ".join(details), DIM)] if details else []))
    label = []
    for i, entry in enumerate(entries):
        rest = len(entries) - i - 1
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
    cols = 2 if two_fit and len(gpus) * (MAX_INNER_H + 2) > height else 1
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


def header(width, interval, driver, waiting=0):
    """Top line: name, version, host, local time | driver (or how long it has kept us waiting), refresh."""
    now = time.time()
    stamp = time.strftime("%Y-%m-%d %a %H:%M:%S", time.localtime(now)) + ".{:02d}".format(int(now % 1 * 100))
    left = [("nvmon", BOLD), (" " + __version__, DIM), ("  " + socket.gethostname(), ""), ("  " + stamp, "")]
    right = [("refresh ", DIM), ("{:g}s".format(interval), "")]
    if waiting >= 1:  # the numbers below are that old: something is keeping the driver busy, or it hangs
        right = [("driver: no answer for {:.0f}s   ".format(waiting), WARN)] + right
    elif driver and width_of(left) + 2 + len(driver) + 3 + width_of(right) <= width:  # dropped when narrow
        right = [(driver + "   ", DIM)] + right
    return spread(width, left, right)


def footer(width, newer=None):
    """Bottom line: a newer release when there is one, with how to update | how to quit."""
    right, left = [("Esc / q quit", DIM)], []
    if newer:
        version, command = newer
        notice = [("update available: " + version, NEWS)]
        how = [(" ({})".format(command or "new nvmon.py: " + RELEASES), "")]
        link = [("   what's new: " + RELEASES, DIM)] if command else []  # a copied nvmon.py already links there
        # As much as fits beside "Esc / q quit": the link goes first, then how to update.
        left = next((option for option in (notice + how + link, notice + how)
                     if width_of(option) + 2 + width_of(right) <= width), notice)
    return spread(width, left, right)


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
            time.sleep(max(0, seconds))
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


DEFAULT_INTERVAL, MIN_INTERVAL = 0.5, 0.1  # seconds


def interval(text):
    """argparse type for -i: seconds between updates. The GPU's own readings change every 0.1-0.2 s, so
    anything faster would only keep the driver busier and counts as MIN_INTERVAL; nan or inf, the default."""
    seconds = float(text)
    return max(MIN_INTERVAL, seconds) if math.isfinite(seconds) else DEFAULT_INTERVAL


def main():
    parser = argparse.ArgumentParser(
        prog="nvmon", description="A fancy NVIDIA GPU monitor for the terminal.",
        epilog="At start nvmon asks PyPI whether a newer release is out; NVMON_NO_UPDATE_CHECK=1 turns that "
               "off. " + REPO)
    parser.add_argument("-i", "--interval", type=interval, default=DEFAULT_INTERVAL, metavar="SEC",
                        help="seconds between updates, at least 0.1 (default: 0.5)")
    parser.add_argument("-g", "--gpus", type=gpu_list, metavar="LIST",
                        help="only these GPUs, e.g. 0,2,4-7 (default: all)")
    parser.add_argument("-V", "--version", action="version", version="nvmon {} {}".format(__version__, REPO))
    args = parser.parse_args()
    if not sys.stdout.isatty():
        sys.exit("nvmon: output is not a terminal (over ssh, use: ssh -t HOST nvmon)")

    try:
        nv = load_nvml()
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))
    poller = None
    try:
        gpus = open_gpus(nv, args.gpus)
        missing = sorted((args.gpus or set()) - {g.index for g in gpus})
        if missing:
            sys.exit("nvmon: GPU {} not found or not accessible".format(", ".join(map(str, missing))))
        if not gpus:
            sys.exit("nvmon: no accessible NVIDIA GPU")
        driver = versions(nv)
        update = UpdateCheck()
        if not os.environ.get("NVMON_NO_UPDATE_CHECK"):
            update.start()
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        for g in gpus:
            g.poll()  # the first numbers, before the screen opens
        poller = Poller(gpus)
        with Screen() as screen:
            deadline = time.monotonic()
            while True:
                for g in gpus:
                    g.frame()
                width, height = os.get_terminal_size()
                screen.draw([header(width, args.interval, driver, poller.waiting())]
                            + render(gpus, width, height - 2) + [footer(width, update.newer)])
                # Fixed-rate ticks; a late tick restarts the schedule instead of bursting.
                deadline = max(deadline + args.interval, time.monotonic())
                # The next frame's polls start ahead of it: half a tick, at most 0.5 s (a busy driver was seen
                # to take 0.4 s). Should the driver answer later still, the frame keeps the last numbers but
                # comes on time.
                if screen.wait(deadline - min(args.interval / 2, 0.5) - time.monotonic()):
                    break
                poller.refresh()
                if screen.wait(deadline - time.monotonic()):
                    break
    except KeyboardInterrupt:
        pass
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))
    finally:
        if poller is None or poller.done.is_set():  # leave NVML be while a call is still inside it
            nv.nvmlShutdown()


if __name__ == "__main__":
    main()
