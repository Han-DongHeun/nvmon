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
import json
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
MAX_INNER_H = 5                      # tallest panel: the full stats column (graph: 5 x 8 levels)
HISTORY = 1024                       # samples kept per GPU; wider than any terminal
BLOCKS = " ▁▂▃▄▅▆▇█"                 # a character cell filled in 1/8 steps


def _fg(n):
    return "\x1b[38;5;{}m".format(n)


def _rgb(rgb):
    return "\x1b[38;2;{};{};{}m".format(*rgb)


def _oklab(rgb):
    """Color `rgb` in OKLab (Björn Ottosson, 2020): a space where equal distances look equally different."""
    def linear(c):  # sRGB's gamma undone
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(linear, rgb)
    l, m, s = (v ** (1 / 3) for v in (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
                                       0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
                                       0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def _palette_rgb(code):
    """The color of code 16-255 of the 256-color palette: a 6 x 6 x 6 cube, then 24 greys."""
    if code >= 232:
        return (8 + 10 * (code - 232),) * 3
    levels, i = (0, 95, 135, 175, 215, 255), code - 16
    return levels[i // 36], levels[i // 6 % 6], levels[i % 6]


PALETTE = [(code, _oklab(_palette_rgb(code))) for code in range(16, 256)]  # (code, OKLab)


def _nearest256(rgb):
    """The palette code that looks nearest to color `rgb`. Nearest in plain RGB would, for a green between
    two of the palette's, as soon take a dull olive as the next green."""
    lab = _oklab(rgb)
    return min(PALETTE, key=lambda entry: sum((a - b) ** 2 for a, b in zip(entry[1], lab)))[0]


# Styles are written in 24-bit color and turned, as they are sent (restyle), into what the terminal shows:
# 24-bit color or 256 colors, as color_mode finds; in gray, as the black and white theme has it.
COLOR_ARG = re.compile(r"\x1b\[([34])8;(?:2;(\d+);(\d+);(\d+)|5;(\d+))m")


@lru_cache(maxsize=None)
def restyle(style, mode, gray=False):
    """`style` in color `mode`: "256" turns 24-bit colors into the palette's nearest. With `gray`, every color
    first becomes the gray as light as it (OKLab lightness): the hue goes, the lightness stays."""
    if mode == "truecolor" and not gray:
        return style

    def swap(m):
        layer, code = m.group(1), m.group(5)  # layer 3: the foreground, 4: the background
        if code and not gray:  # a palette color already
            return m.group(0)
        rgb = _palette_rgb(int(code)) if code else tuple(int(v) for v in m.group(2, 3, 4))
        if gray:
            rgb = _from_oklab((_oklab(rgb)[0], 0, 0))
        if mode == "256":
            return "\x1b[{}8;5;{}m".format(layer, _nearest256(rgb))
        return "\x1b[{}8;2;{};{};{}m".format(layer, *rgb)
    return COLOR_ARG.sub(swap, style)


def color_mode(truecolor):
    """"truecolor" where the terminal is known to show 24-bit color (it said so, see Screen.probe, or
    COLORTERM does), else "256": every terminal of today shows those, and they look nearly the same, while
    24-bit color where it is not understood comes out in odd colors."""
    return "truecolor" if truecolor or os.environ.get("COLORTERM") in ("truecolor", "24bit") else "256"


def _gradient(stops):
    """Colors (r, g, b) for 0..100, interpolated through (position, (r, g, b)) `stops`."""
    colors = []
    for i in range(101):
        x = min(max(i, stops[0][0]), stops[-1][0])
        (x0, c0), (x1, c1) = next(pair for pair in zip(stops, stops[1:]) if x <= pair[1][0])
        colors.append(tuple(round(a + (b - a) * (x - x0) / (x1 - x0)) for a, b in zip(c0, c1)))
    return colors


def _from_oklab(lab):
    """The color (r, g, b) at OKLab `lab`: _oklab the other way."""
    L, a, b = lab
    l, m, s = ((L + x * a + y * b) ** 3 for x, y in ((0.3963377774, 0.2158037573), (-0.1055613458, -0.0638541728),
                                                    (-0.0894841775, -1.2914855480)))

    def gamma(c):  # sRGB's gamma again
        c = 12.92 * c if c <= 0.0031308 else 1.055 * max(c, 0) ** (1 / 2.4) - 0.055
        return min(255, max(0, round(c * 255)))
    return (gamma(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            gamma(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            gamma(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))


def _readable(rgb, floor=0.62):
    """`rgb` as text on a dark background: when darker than `floor` (OKLab lightness), just that light, and
    a little greyer where a color that light would be beyond what a screen shows."""
    L, a, b = _oklab(rgb)
    if L >= floor:
        return tuple(rgb)
    for k in (1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        lifted = _from_oklab((floor, a * k, b * k))
        if _oklab(lifted)[0] >= floor - 0.005:  # not cut short by the screen's limits
            return lifted
    return _from_oklab((floor, 0, 0))


class Theme(NamedTuple):
    """A look: the colors utilization and shares take on in the graph (graph) and as numbers (text: the
    same, but never too dark to read), those of temperature, and the color of process names (accent)."""
    name: str
    graph: tuple      # style codes for 0 .. 100 %
    text: tuple
    temp: tuple       # style codes for 0 .. 100 °C
    accent: str
    selected: str     # the picked job: reverse video on the accent
    gray: bool        # black and white: paint turns every color gray, as light as it was


def theme(name, stops, accent, temp=None, gray=False):
    """A Theme. `stops`, low to high, are (position, (r, g, b)) pairs, or "#rrggbb ..." colors, which are
    placed by how different they look (OKLab distance) so that equal steps in % look about equally big.
    Temperature takes the same colors over 30-88 °C, unless `temp` has stops of its own."""
    if isinstance(stops, str):
        colors = [tuple(bytes.fromhex(code[1:])) for code in stops.split()]
        labs = [_oklab(c) for c in colors]
        gaps = [math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q))) for p, q in zip(labs, labs[1:])]
        stops = [(100 * sum(gaps[:i]) / sum(gaps), c) for i, c in enumerate(colors)]
    scale = _gradient(stops)
    heat_of_temp = _gradient(temp or [(30 + 0.58 * x, c) for x, c in stops])
    accent = accent if accent.startswith("\x1b") else _rgb(tuple(bytes.fromhex(accent[1:])))
    return Theme(name, tuple(map(_rgb, scale)), tuple(_rgb(_readable(c)) for c in scale),
                 tuple(_rgb(_readable(c)) for c in heat_of_temp), accent, "\x1b[7m" + accent, gray)


GREEN, YELLOW, ORANGE, RED = (95, 175, 95), (215, 215, 95), (215, 135, 95), (215, 95, 95)
# The themes c goes through. First nvmon's own: utilization and shares green -> yellow -> orange -> red, the
# yellow-green at 30 %, not 20, as from green to it is a long way to the eye; temperature in °C, blue at 30 to
# red at 88, as idle GPUs sit at 30-45, busy ones at 60-80, and most throttle from about 85-90. Then palettes
# made to look good, low to high, the darkest end dropped where it would vanish into the background: from
# works at the Metropolitan Museum, as MetBrewer (Blake R. Mills) takes them, Hiroshige's "Sailing Boats
# Returning to Yabase", Hokusai's "Yoro Waterfall", O'Keeffe's "Red and Yellow Cliffs", Van Gogh's "First
# Steps", the pinks of Benedictus's "Relais" and the lilacs of Cassatt's "Lilacs in a Window"; and the Rose
# Pine editor theme.
THEMES = [
    theme("nvmon", [(0, GREEN), (30, (175, 215, 95)), (40, YELLOW), (60, (215, 175, 95)), (80, ORANGE), (100, RED)],
          _fg(110), [(30, (95, 135, 215)), (45, (95, 175, 175)), (60, GREEN), (72, YELLOW), (80, ORANGE), (88, RED)]),
    theme("Hiroshige", "#376795 #528fad #72bcd5 #aadce0 #ffe6b7 #ffd06f #f7aa58 #ef8a47 #e76254", "#aadce0"),
    theme("Hokusai", "#295384 #5a97c1 #74c8c3 #95c36e #d8d97a", "#74c8c3"),
    theme("O'Keeffe", "#92351e #b9563f #d37750 #e69c6b #ecb27d #f2c88f #fbe3c2", "#f2c88f"),
    theme("Van Gogh", "#1f5b25 #3c7c3d #669d62 #9cc184 #c2d6a4 #e7e5cc", "#c2d6a4"),
    theme("Benedictus", "#9a133d #b93961 #d8527c #f28aaa #f9b4c9 #f9e0e8", "#f28aaa"),
    theme("Cassatt", "#574571 #90719f #b695bc #dec5da", "#b695bc"),
    theme("Rose Pine", "#3e8fb0 #9ccfd8 #c4a7e7 #ea9a97 #eb6f92", "#c4a7e7"),
]
# No hue, only lightness: dark gray when idle to near white when busy.
THEMES.append(theme("black and white", "#505050 #f0f0f0", "#d0d0d0", gray=True))
THEME = THEMES[0]  # the one in use, set by View.screen before it draws
SHOWS = "truecolor"  # what the terminal shows (color_mode), set with it
# Warnings on the top edge, the same in every theme: yellow = worth a look, orange = slowed, red = act.
WARN, SLOW, ALERT = _rgb(YELLOW), _rgb(ORANGE), _rgb(RED)
DIM, FAINT = _fg(240), _fg(238)
BOLD, RESET = "\x1b[1m", "\x1b[0m"


def heat(t):
    """The theme's color for a number at t in [0, 1] of the 0-100 % scale."""
    return THEME.text[round(min(max(t, 0), 1) * 100)]


# ── NVML (libnvidia-ml / nvml.dll, part of the NVIDIA driver) ───────────────

NVML_SUCCESS = 0
NVML_ERROR_NOT_FOUND, NVML_ERROR_INSUFFICIENT_SIZE = 6, 7
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
    pid: int
    name: str                     # script name for Python, else the executable
    command: str                  # the command line from `name` on, e.g. "train.py --lr 3e-4"
    mem: Optional[float]          # GPU memory, GiB
    owner: Optional[str]          # account; None for our own processes
    env: Optional[str]            # conda environment
    started: Optional[float]      # start time, seconds since the epoch
    job: int                      # the job it is part of: its launcher's PID (torchrun...), else its own
    launcher: Optional[str]       # the launcher's name; None when the job is just this process


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
    slowdown: Optional[tuple]     # (label, color) when the clock is held back
    processes: list               # [Process]
    process_util: Optional[dict]  # pid -> % of time its kernels ran; None unless the process list is open


class ProcessUtil(ctypes.Structure):  # nvmlProcessUtilizationSample_t
    _fields_ = [("pid", c_uint), ("timeStamp", c_ulonglong), ("smUtil", c_uint), ("memUtil", c_uint),
                ("encUtil", c_uint), ("decUtil", c_uint)]


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
        self._facts = {}  # pid -> what _facts_of says: fixed for a process's life, so read once
        self._listed, self._listed_mem = [], None  # process list, and the memory in use when it was read
        # Per-process utilization: the driver's samples since `seen` (µs), read at most every second.
        self._util, self._util_due, self._util_seen, self._util_buf = None, 0, 0, (ProcessUtil * 128)()

    def _call(self, fn, *args):
        """Run the device query `fn`; its NVML return code. A query this GPU or driver cannot answer is not
        asked again: the answer costs time too (0.3 ms of CPU for a fanless H100 to say it has no fan)."""
        if fn in self._unsupported:
            return NVML_ERROR_NOT_SUPPORTED
        query = getattr(self.nv, fn, None)  # None: a driver older than this function
        rc = query(self.handle, *args) if query else NVML_ERROR_FUNCTION_NOT_FOUND
        if rc in (NVML_ERROR_NOT_SUPPORTED, NVML_ERROR_FUNCTION_NOT_FOUND, NVML_ERROR_ARGUMENT_VERSION_MISMATCH):
            self._unsupported.add(fn)
        return rc

    def _ok(self, fn, *args):
        return self._call(fn, *args) == NVML_SUCCESS

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
        """(label, color) for why the clock is held back, or None."""
        reasons = c_ulonglong()
        if not (self._ok("nvmlDeviceGetCurrentClocksEventReasons", byref(reasons))
                or self._ok("nvmlDeviceGetCurrentClocksThrottleReasons", byref(reasons))):  # the pre-R535 name
            return None
        return next(((label, color) for bits, label, color in SLOWDOWNS if reasons.value & bits), None)

    def _processes(self):
        infos, count = (ProcessInfo * MAX_PROCESSES)(), c_uint(MAX_PROCESSES)
        if not self._ok("nvmlDeviceGetComputeRunningProcesses_v3", byref(count), infos):
            return []
        running = infos[:count.value]
        self._facts = {p.pid: self._facts.get(p.pid) or self._facts_of(p.pid) for p in running}
        procs = []
        for p in running:
            mem = None if p.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE else p.usedGpuMemory / 2**30
            name, command, owner, env, started, job, launcher = self._facts[p.pid]
            procs.append(Process(p.pid, name, command, mem, owner, env, started, job, launcher))
        return procs

    def _facts_of(self, pid):
        """(name, command, owner, conda environment, start time, job, launcher name) of `pid`."""
        launcher = launcher_of(pid)
        return (self._command(pid) + process_facts(pid)
                + ((launcher, self._command(launcher)[0]) if launcher else (pid, None)))

    def _command(self, pid):
        """(short name, command line from it on) of `pid`. For a Python interpreter the name is the script
        or module it runs: ("train.py", "train.py --lr 3e-4"), ("torch.distributed.run", ...)."""
        try:
            with open("/proc/{}/cmdline".format(pid), "rb") as f:
                argv = [a for a in f.read().decode("utf-8", "replace").split("\0") if a]
        except OSError:  # not Linux, or the process already exited
            argv = []
        if not argv:
            name = ctypes.create_string_buffer(256)
            if self.nv.nvmlSystemGetProcessName(pid, name, 256) != NVML_SUCCESS:
                return str(pid), ""
            argv = [name.value.decode("utf-8", "replace")]
        exe, rest = os.path.basename(argv[0]), argv[1:]
        if exe.startswith("python"):
            for i, arg in enumerate(rest):
                if arg == "-m" and i + 1 < len(rest):
                    return rest[i + 1], " ".join(rest[i + 1:])
                if not arg.startswith("-"):
                    name = os.path.basename(arg)
                    return name, " ".join([name] + rest[i + 1:])
        return exe, " ".join([exe] + rest)

    def _process_util(self):
        """{pid: % of the time its kernels ran} since the last call; None where the GPU cannot tell
        (with MIG, say). The driver takes some 2 ms to answer, so this is asked only while it is shown."""
        for _ in range(3):
            count = c_uint(len(self._util_buf))
            rc = self._call("nvmlDeviceGetProcessUtilization", self._util_buf, byref(count),
                            c_ulonglong(self._util_seen))
            if rc != NVML_ERROR_INSUFFICIENT_SIZE:
                break
            self._util_buf = (ProcessUtil * (2 * count.value))()
        if rc == NVML_ERROR_NOT_FOUND:  # no new samples: nothing ran
            return {}
        if rc != NVML_SUCCESS:
            return None
        samples = self._util_buf[:count.value]
        self._util_seen = max([self._util_seen] + [s.timeStamp for s in samples])
        runs = {}
        for s in samples:
            runs.setdefault(s.pid, []).append(s.smUtil)
        return {pid: sum(values) / len(values) for pid, values in runs.items()}

    def poll(self, detail=False):
        """Read everything shown into `sample`; with `detail`, the processes' utilization too."""
        util = Utilization()
        util = util.gpu if self._ok("nvmlDeviceGetUtilizationRates", byref(util)) else None
        used, total = self._memory()
        # The process list is the costliest query on a busy GPU. Processes only come and go with memory
        # being allocated or freed, so the list is read again only when the memory in use has changed.
        if used is None or used != self._listed_mem:
            self._listed, self._listed_mem = self._processes(), used
        if not detail:
            self._util, self._util_due = None, 0  # shown again, it is read at once
        elif time.monotonic() >= self._util_due:
            self._util = self._process_util() if self._listed else {}
            self._util_due = time.monotonic() + 1
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
            process_util=self._util,
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
        self.detail = set()  # the GPUs to read the processes' utilization of, see View.detail
        self.rounds = 0      # rounds of polls done
        self.wanted, self.done = threading.Event(), threading.Event()
        self.done.set()
        self.start()

    def run(self):
        while True:
            self.wanted.wait()
            self.wanted.clear()
            try:
                for gpu in self.gpus:
                    gpu.poll(gpu.index in self.detail)
            except Exception as e:  # a bug: refresh() raises it rather than leaving the numbers frozen
                self.error = e
            self.rounds += 1
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
    uid = uid_of(pid)
    if uid is None or uid == os.getuid():
        return None
    import pwd
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:  # no passwd entry, e.g. inside a container
        return str(uid)


def uid_of(pid):
    """The uid of the account running `pid`; None where unknown."""
    try:
        with open("/proc/{}/status".format(pid)) as f:
            return next(int(line.split()[1]) for line in f if line.startswith("Uid:"))
    except (OSError, StopIteration, ValueError):
        return None


def parent_of(pid):
    """The PID of `pid`'s parent; None where unknown."""
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            return int(f.read().rsplit(")", 1)[1].split()[1])  # field 4, counted after "(name)"
    except (OSError, ValueError, IndexError):
        return None


# Parents that start each program on its own, rather than as the parts of one job.
NOT_LAUNCHERS = {"bash", "sh", "dash", "zsh", "fish", "ksh", "tcsh", "csh", "tmux: server", "screen", "SCREEN",
                 "sshd", "sudo", "su", "login", "systemd", "init", "script", "nohup"}


def launcher_of(pid):
    """The PID of what launched `pid` as part of a job, such as torchrun for its workers: its parent, unless
    that is PID 1, a shell or the like, or someone else's process. None when `pid` stands alone."""
    parent = parent_of(pid)
    if not parent or parent <= 1 or uid_of(parent) != uid_of(pid):
        return None
    try:
        with open("/proc/{}/comm".format(parent)) as f:
            return None if f.read().strip() in NOT_LAUNCHERS else parent
    except OSError:
        return None


def holds_gpu(pid):
    """True when `pid` has an NVIDIA device open, as a GPU process in our PID namespace does."""
    try:
        fds = os.listdir("/proc/{}/fd".format(pid))
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink("/proc/{}/fd/{}".format(pid, fd)).startswith("/dev/nvidia"):
                return True
        except OSError:  # closed meanwhile
            pass
    return False


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


# ── jobs ─────────────────────────────────────────────────────────────────────

class Job(NamedTuple):
    """GPU processes that run as one: a launcher's workers (torchrun, deepspeed...), or a single process."""
    pid: int                      # where a stop goes: the launcher, else the process
    name: str                     # what runs, e.g. train.py
    launcher: Optional[str]       # e.g. torchrun
    members: tuple                # (pid, start time) of its GPU processes
    gpus: tuple                   # the GPUs it uses
    util: Optional[float]         # % of the time its kernels ran, over its GPUs; None when unknown
    mem: float                    # GiB, on all of them
    started: Optional[float]
    owner: Optional[str]
    env: Optional[str]
    command: str                  # its biggest process's command line


def jobs(gpus):
    """The jobs on `gpus`, in the order of the GPUs they use."""
    parts = {}
    for gpu in gpus:
        for p in gpu.now.processes:
            parts.setdefault(p.job, []).append((gpu, p))
    found = []
    for pid, entries in parts.items():
        procs = [p for _, p in entries]
        main = max(procs, key=lambda p: p.mem or 0)
        starts = [p.started for p in procs if p.started is not None]
        # Its share of each of its GPUs, averaged: a DDP job reads like one of its GPUs.
        shares = {}
        for gpu, p in entries:
            if gpu.now.process_util is not None:
                shares[gpu.index] = shares.get(gpu.index, 0) + gpu.now.process_util.get(p.pid, 0)
        found.append(Job(pid, main.name, main.launcher, tuple(sorted({(p.pid, p.started) for p in procs})),
                         tuple(sorted({gpu.index for gpu, _ in entries})),
                         sum(shares.values()) / len(shares) if shares else None, sum(p.mem or 0 for p in procs),
                         min(starts) if starts else None, main.owner, main.env, main.command))
    return sorted(found, key=lambda job: (job.gpus, job.pid))


def ranges(numbers):
    """(0, 2, 4, 5, 6, 7) -> "0,2,4-7", as -g takes them."""
    spans = []
    for n in numbers:
        if spans and spans[-1][1] == n - 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ",".join(str(a) if a == b else "{}-{}".format(a, b) for a, b in spans)


def account(job):
    """"kim/torch": whose it is, and the conda environment; our own go without an account."""
    return "/".join(part for part in (job.owner, job.env) if part)


def about(job):
    """"kim/torch · GPUs 4-5 · util 86% · 30.2G · 2h13m · torchrun job" as segments, util in its color;
    the launcher, often a long name, comes last, where a narrow line cuts first."""
    parts = [(account(job), DIM)] if account(job) else []
    parts.append(("GPU{} {}".format("s" if len(job.gpus) > 1 else "", ranges(job.gpus)), DIM))
    if job.util is not None:
        parts.append(("util ", DIM, "{:.0f}%".format(job.util), heat(job.util / 100)))
    parts.append(("{:.1f}G".format(job.mem), DIM))
    if job.started is not None:
        parts.append((elapsed(job.started), DIM))
    parts.append(("{} job".format(job.launcher) if job.launcher else "PID {}".format(job.pid), DIM))
    out = []
    for part in parts:
        out += ([(" · ", DIM)] if out else []) + [part[i:i + 2] for i in range(0, len(part), 2)]
    return out


def stop(job, name):
    """Send signal `name` ("SIGTERM" or "SIGKILL") to `job`, once /proc confirms its PIDs still are the
    processes we saw: started when we saw them, holding a GPU open (inside a container, NVML can report
    PIDs of the host), and ours. SIGTERM goes to the launcher, which then ends its workers; SIGKILL, which
    it cannot pass on, goes to each worker as well. Returns (a note on what happened, whether it was sent)."""
    if os.name == "nt":
        return [("stopping processes works on Linux", WARN)], False
    for pid, started in job.members:
        if start_time(pid) != started or not holds_gpu(pid) or (job.launcher and parent_of(pid) != job.pid):
            return [("{} has changed meanwhile: nothing sent".format(job.name), WARN)], False
    if uid_of(job.pid) != os.getuid():
        return [("only your own processes can be stopped", WARN)], False
    targets = [job.pid] + ([pid for pid, _ in job.members] if name == "SIGKILL" and job.launcher else [])
    try:
        for pid in targets:
            os.kill(pid, getattr(signal, name))
    except ProcessLookupError:
        return [("{} has already ended".format(job.name), "")], False
    except PermissionError:
        return [("not allowed to signal PID {}".format(job.pid), WARN)], False
    return [("sent {} to {} (PID {})".format(name, job.launcher or job.name, job.pid), "")], True


# ── new releases ─────────────────────────────────────────────────────────────

class UpdateCheck(threading.Thread):
    """Asks PyPI for the latest release in the background. Afterwards `newer` is (version, how to
    update) when that release is newer than this one; it stays None otherwise, and on any failure."""

    def __init__(self):
        super().__init__(daemon=True)  # a slow network never holds up quitting
        self.newer = None

    def run(self):
        try:
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
# A line is a list of (text, style) segments; every character is one cell wide. A segment that can be
# clicked carries a third item, what the click means: a job's PID, or ("key", name) and the like.

FADED = "\x1b[2m"            # "faint": GPUs the selected job does not use


def width_of(line):
    return sum(len(seg[0]) for seg in line)


def clip(line, width):
    """`line` cut to at most `width` cells."""
    out = []
    for seg in line:
        if width <= 0:
            break
        out.append((seg[0][:width],) + seg[1:])
        width -= len(seg[0])
    return out


def pad(line, width):
    """`line` padded with spaces to `width` cells."""
    return line + [(" " * (width - width_of(line)), "")]


@lru_cache(maxsize=None)
def bands(height, scale, shows):
    """The colors of a `height`-row graph's bands, bottom up, in the colors `scale` (a theme's graph), and how
    many bands a cell holds: two (see graph_row), each its middle's color. In 256 colors (`shows`) two bands
    side by side may come out as the same palette color, and the bands were one cell high here and half a
    cell there; then a band is a cell high, and one that would come out as the band below takes the nearest
    color that does not."""
    def rgb(code):
        return tuple(int(v) for v in COLOR_ARG.match(code).group(2, 3, 4))
    halves = [scale[round((4 * k + 2) * 100 / (height * 8))] for k in range(2 * height)]
    if shows == "truecolor":
        return tuple(halves), 2
    codes = [_nearest256(rgb(c)) for c in halves]
    if all(a != b for a, b in zip(codes, codes[1:])):
        return tuple(map(_fg, codes)), 2
    picked = []
    for k in range(height):
        lab = _oklab(rgb(scale[round((8 * k + 4) * 100 / (height * 8))]))
        ranked = sorted(PALETTE, key=lambda entry: sum((a - b) ** 2 for a, b in zip(entry[1], lab)))
        picked.append(next(code for code, _ in ranked if not picked or code != picked[-1]))
    return tuple(map(_fg, picked)), 1


@lru_cache(maxsize=None)
def graph_row(r, height, two_tone, scale, shows="truecolor"):
    """The (block, style) cell that row `r` (0 = bottom) of a `height`-row graph shows for each level
    0 .. 8 * height, in the colors `scale` (a theme's graph), as a terminal that `shows` 24-bit or 256 colors.

    The color goes with the height, as a gradient behind the bars would. A cell holds one character in one
    color on one background, so two colors at most: a full cell is "▀" in its upper half's color on its
    lower half's, which makes twice as many bands as rows (see bands); no other split of a cell comes closer
    to a smooth gradient. The ragged top is a block from the bottom in its lower half's color, the band
    beside it, so that every height has the one color from bar to bar. A top in the upper half would need
    three (the two bands, and the empty rest), so there it goes by halves: 5/8 shows as 4/8, 6/8 and 7/8 as
    a full cell; the number beside the graph has the exact value. With a band to a cell (bands), and
    without `two_tone` (faded boxes, as faint leaves backgrounds bright), a cell is one color and the top
    keeps its eighths.
    """
    colors, per_cell = bands(height, scale, shows)
    halves = two_tone and per_cell == 2
    steps = height * 8
    cells = []
    for level in range(steps + 1):
        fill = min(8, max(0, level - r * 8))
        if halves and fill > 4:
            fill = 4 if fill == 5 else 8
        if not fill:
            cells.append((" ", ""))
        elif halves and fill == 8:
            cells.append(("▀", colors[2 * r + 1] + colors[2 * r].replace("\x1b[38;", "\x1b[48;")))  # 48: background
        elif halves:
            cells.append((BLOCKS[fill], colors[2 * r]))
        else:
            cells.append((BLOCKS[fill], colors[r] if per_cell == 1 else scale[round((8 * r + 4) * 100 / steps)]))
    return cells


def graph(history, width, height, two_tone=True):
    """Stacked block chart, newest sample on the right, top row first."""
    vals = list(history)[-width:] if width > 0 else []
    steps = height * 8
    # Any non-zero value gets at least one sub-level so light load stays visible.
    levels = [0] * (width - len(vals)) + [max(round(v * steps / 100), 1 if v else 0) for v in vals]
    text_of, style_of = operator.itemgetter(0), operator.itemgetter(1)
    rows = []
    for r in reversed(range(height)):
        cells = map(graph_row(r, height, two_tone, THEME.graph, SHOWS).__getitem__, levels)
        rows.append([("".join(map(text_of, run)), style) for style, run in itertools.groupby(cells, key=style_of)])
    return rows


def row(left, right=(), width=INFO_W):
    """One stats line: `left`, then the `right` segments flush right; exactly `width` cells."""
    room = width - width_of(right)
    return pad(clip(left, room - 1 if right else room), room) + list(right)


def share(part, whole):
    """A 4-cell percentage, colored by its value."""
    if part is None or not whole:
        return [("   -", "")]
    return [("{:.0f}%".format(100 * part / whole).rjust(4), heat(part / whole))]


def num(value, fmt):
    return "-" if value is None else fmt.format(value)


def info(gpu, height, width=INFO_W):
    """The stats column, `width` cells wide; None for the rule between MEM and the PCIe traffic."""
    s = gpu.now
    total = num(s.mem_total, "{:.1f}")
    busy = [("GPU ", "")] + share(s.util, 100)
    if gpu.has_activity:  # "cores" = share of SMs at work, "tensor" = Tensor Core activity
        busy += [("  cores " + share(s.cores, 100)[0][0] + " tensor " + share(s.tensor, 100)[0][0], FAINT)]
    # A link's top speed: its generation's per-lane rate times the lanes it runs on now.
    lanes = s.pcie_width or gpu.pcie_max_width
    top = PCIE_LANE_GBS.get(gpu.pcie_max_gen, 0) * (lanes or 0)
    cap = (" / {:.0f}".format(top), WARN if degraded(gpu) else "") if top else ("", "")

    def link(label, rate):  # "CPU -> GPU ... 0.16 / 63 GB/s", flush right like MEM's used / total
        return row([(label, "")], [("-" if rate is None else "{:.2f}".format(rate / 1e9), ""), cap, (" GB/s", "")],
                   width)

    gpu_row = row(busy, (), width)
    mem_row = row([("MEM ", "")] + share(s.mem_used, s.mem_total),
                  [("{} / {} GiB".format(num(s.mem_used, "{:.1f}"), total), "")], width)
    to_gpu, to_cpu = link("CPU -> GPU", s.rx), link("GPU -> CPU", s.tx)
    # The rule (None) between MEM and the PCIe traffic only appears when there is room.
    if height >= 5:
        return [gpu_row, mem_row, None, to_gpu, to_cpu] + [row([], (), width)] * (height - 5)
    return [gpu_row, mem_row, to_gpu, to_cpu][:height]


def degraded(gpu):
    """True when the PCIe link runs on fewer lanes than card and slot allow, e.g. a loose card."""
    s = gpu.now
    return bool(s.pcie_width and gpu.pcie_max_width and s.pcie_width < gpu.pcie_max_width)


def panel(gpu, width, height, selected=None, graphs=True):
    """One GPU's box, the utilization graph beside the stats or, without `graphs`, a card of the stats
    alone; faded when a job is `selected` and it does not run here."""
    s = gpu.now
    faded = selected is not None and not any(p.job == selected for p in s.processes)
    # Both sides as (segments, rank); when space runs out the lowest rank goes first:
    # fan, name, power, clock, temperature, PCIe warning, slowdown warning. The GPU number stays.
    # Fixed widths keep things from shifting.
    limit = num(s.power_limit, "{:.0f}")  # power padded to the limit's width
    label = [([("GPU {}".format(gpu.index), BOLD)], None), ([("  " + gpu.name, "")], 2),
             ([("  │ ", DIM), ("{} / {} W".format(num(s.power, "{:.0f}").rjust(len(limit)), limit), "")], 3)]
    parts = [] if s.slowdown is None else [([s.slowdown], 9)]
    if degraded(gpu):
        parts += [([("PCIe DEGRADED: x{} -> x{}".format(gpu.pcie_max_width, s.pcie_width), WARN)], 8)]
    parts += [] if s.temp is None else [([("{:>3}°C".format(s.temp), THEME.temp[min(max(s.temp, 0), 100)])], 5)]
    parts += [] if s.fan is None else [([("FAN {:>3}%".format(s.fan), "")], 1)]
    parts += [] if s.clock is None else [([("{:>4} MHz".format(s.clock), s.slowdown[1] if s.slowdown else "")], 4)]
    top = edge(width, label, parts)
    if graphs:
        graph_w = max(0, width - CHROME_W - INFO_W)
        split = 2 + graph_w + 1  # column of the graph | stats divider
        rule = [(" ├" + "─" * (INFO_W + 2) + "┤", DIM)]  # across the stats, joined to the borders
        body = [[("│ ", DIM)] + g + (rule if i is None else [(" │ ", DIM)] + i + [(" │", DIM)])
                for g, i in zip(graph(gpu.history, graph_w, height, not faded), info(gpu, height))]
    else:
        split = width - 1  # the processes take the whole bottom edge
        rule = [("├" + "─" * (width - 2) + "┤", DIM)]
        body = [rule if i is None else [("│ ", DIM)] + i + [(" │", DIM)] for i in info(gpu, height, width - 4)]
    bottom = bottom_edge(width, split, process_label(s.processes, split - 4, selected, gpu.index))
    lines = [top] + body + [bottom]
    if not faded:
        return lines
    return [[(seg[0], FADED + (seg[1] if seg[1] != BOLD else "")) + seg[2:] for seg in line] for line in lines]


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


def process_label(processes, room, selected=None, index=None):
    """Processes grouped by owner and conda environment, groups and processes by memory, biggest
    first: "migi: train.py 2h13m 15.1G · eval.py 5m 0.5G   kim/torch: a.py 1d4h 9.0G" (our own
    processes carry no owner). As many whole entries as fit in `room` cells, then "+N" for the rest, or
    "+" when none is left out: clicked, it opens the process list at GPU `index`. Each entry, when
    clicked, means its job; the `selected` job's names are reversed."""
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
            name = (process.name, THEME.selected if process.job == selected else THEME.accent, process.job)
            entries.append(gap + ([(tag + ": ", DIM)] if tag and not j else []) + [name]
                           + ([(" " + " ".join(details), DIM, process.job)] if details else []))
    def more(n):
        return ("  +{}".format(n) if n else "  +", DIM, ("list", index))
    label = []
    for i, entry in enumerate(entries):
        rest = len(entries) - i - 1
        if width_of(label + entry) + len(more(rest)[0]) > room:
            # The first entry is always shown, cut if need be; later ones collapse into the count.
            return clip(entry, room - len(more(rest)[0])) + [more(rest)] if not label else label + [more(rest + 1)]
        label += entry
    return label + [more(0)] if label else []


def bottom_edge(width, split, label):
    """Bottom border with `label` flush right against column `split`, the graph | stats divider."""
    label = clip(label, split - 4)  # keep "╰─" and a space on each side
    middle = [(" ", "")] + label + [(" ", "")] if label else []
    return ([("╰" + "─" * max(0, split - 1 - width_of(middle)), DIM)] + middle
            + [("─" * max(0, width - split - 1) + "╯", DIM)])


MIN_INNER_H = 2  # the shortest a box gets; when even that leaves GPUs out, the boxes scroll


CARD_W = 48  # the narrowest a box without graph gets: room for the name, power, temperature and clock


def layout(count, width, height, graphs=True):
    """(columns, inner height) for `count` GPU boxes in `width` x `height` cells."""
    if graphs:  # two columns only when one column cannot show every GPU at full height
        two_fit = width // 2 >= CHROME_W + INFO_W + MIN_GRAPH_W
        cols = 2 if two_fit and count * (MAX_INNER_H + 2) > height else 1
    else:  # cards: as many side by side as fit
        cols = max(1, width // CARD_W)
    rows = max(1, math.ceil(count / cols))
    return cols, max(MIN_INNER_H, min(MAX_INNER_H, height // rows - 2))


def render(gpus, width, height, selected=None, shape=None, graphs=True):
    """The GPU boxes, in `shape` = (columns, inner height) or else the layout that fits: exactly `height`
    lines of exactly `width` cells."""
    cols, inner = shape or layout(len(gpus), width, height, graphs)
    rows = math.ceil(len(gpus) / cols)
    lines = []
    for r in range(rows):
        panels = [panel(g, width // cols, inner, selected, graphs) for g in gpus[r * cols:(r + 1) * cols]]
        lines += [[seg for part in parts for seg in part] for parts in zip(*panels)]
    lines = [pad(clip(line, width), width) for line in lines[:height]]
    return lines + [[(" " * width, "")]] * (height - len(lines))


# The process list's columns: heading, width (">" when flush right), the cell for a job as (text, style),
# and the key it sorts by. "command" takes whatever width is left.
COLUMNS = [
    ("PID", 8, lambda j: (str(j.pid), ""), lambda j: j.pid),
    ("account/env", 14, lambda j: (account(j), ""), account),
    ("process", 22, lambda j: (j.name + ("  {} workers".format(len(j.members)) if len(j.members) > 1 else ""),
                               THEME.accent), lambda j: j.name.lower()),
    ("GPUs", 8, lambda j: (ranges(j.gpus), ""), lambda j: j.gpus),
    (">util", 5, lambda j: share(j.util, 100)[0] if j.util is not None else ("-", ""),
     lambda j: -1 if j.util is None else j.util),
    (">memory", 7, lambda j: ("{:.1f}G".format(j.mem), ""), lambda j: j.mem),
    (">time", 7, lambda j: (elapsed(j.started) if j.started is not None else "-", ""),
     lambda j: -(j.started or 0)),
    ("command", 0, lambda j: (j.command, DIM), lambda j: j.command),
]
SORTS = {heading.lstrip(">"): key for heading, _, _, key in COLUMNS}
COMMAND_AT = 1 + sum(size + 2 for _, size, _, _ in COLUMNS[:-1])  # the column where commands start


def cells(values, width, fill=("",)):
    """One list line of `width` cells from (text, style, ...) values, one per column, two spaces apart;
    `fill` = (style, ...) of the spaces between, so a selected row stays one bar and clicks anywhere."""
    line, used = [(" ",) + fill], 1
    for (heading, size, _, _), (text, *rest) in zip(COLUMNS, values):
        size = size or max(0, width - used - 1)
        text = text[:size].rjust(size) if heading.startswith(">") else text[:size].ljust(size)
        line += [(text,) + tuple(rest), (" " if heading == "command" else "  ",) + fill]
        used += size + 2
    return line


def process_list(jobs, width, rows, selected, top, sort, shift=0):
    """The list p opens, `rows` lines: the headings (clicking one sorts by it; `sort` = (column, descending)),
    the jobs from index `top` on (clicking one means it; the `selected` one reversed, its command moved
    `shift` characters on), how many more. A command too long to show ends in "…"."""
    room = max(0, width - COMMAND_AT - 1)
    column, descending = sort
    head = []
    for heading, _, _, _ in COLUMNS:
        name = heading.lstrip(">")
        head.append((name + ("▼" if descending else "▲") if name == column else name,
                     "" if name == column else DIM, ("sort", name)))
    lines = [cells(head, width)]
    shown = jobs[top:top + rows - 2]
    for job in shown:
        chosen = job.pid == selected
        values = [cell(job) for _, _, cell, _ in COLUMNS]
        command = "…" + job.command[shift:] if chosen and shift else job.command
        values[-1] = (command if len(command) <= room else command[:max(0, room - 1)] + "…", DIM)
        lines.append(cells([(text, THEME.selected if chosen else color, job.pid) for text, color in values],
                           width, (THEME.selected if chosen else "", job.pid)))
    if not jobs:
        lines.append([(" no processes on these GPUs", DIM)])
    above, below = top, len(jobs) - top - len(shown)
    more = ", ".join(n for n in ("{} more above".format(above) if above else "",
                                 "{} more below".format(below) if below else "") if n)
    lines.append(spread(width, [], [(more + " · wheel or ↑↓ to scroll", DIM)]) if more else [])
    lines = [pad(clip(line, width), width) for line in lines[:rows]]
    return lines + [[(" " * width, "")]] * (rows - len(lines))


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


# Keys by what the terminal sends: a control character, or how an escape sequence ends ("ESC [ A" is up).
# Apart, so that a typed "C" is a C, not the right arrow.
CONTROL_KEYS = {"\r": "enter", "\n": "enter", "\t": "tab", "\x1b": "esc"}
SEQUENCE_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end", "Z": "backtab",
                 "1~": "home", "4~": "end", "5~": "pgup", "6~": "pgdn"}
HINT_KEYS = {"Esc": "esc", "Esc / q": "q"}  # what clicking a key hint presses

# A Korean keyboard in Hangul mode types letters of Hangul: the key of q gives ㅂ. Taken back to the keys of
# the common layout (2-beolsik), the keys work without switching to English; a syllable the input method has
# put together, such as 사, counts as the keys typed for it, t and k.
JAMO_KEYS = dict(zip("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡㅃㅉㄸㄲㅆㅒㅖ", "qwertyuiopasdfghjklzxcvbnmQWERTOP"))
JAMO_KEYS.update({"ㅘ": "hk", "ㅙ": "ho", "ㅚ": "hl", "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅢ": "ml", "ㄳ": "rt",
                  "ㄵ": "sw", "ㄶ": "sg", "ㄺ": "fr", "ㄻ": "fa", "ㄼ": "fq", "ㄽ": "ft", "ㄾ": "fx", "ㄿ": "fv",
                  "ㅀ": "fg", "ㅄ": "qt"})
INITIALS, MEDIALS, FINALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ", "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ", \
    " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def typed(char):
    """The keys `char` was typed with: "q" for ㅂ, "tk" for 사, `char` itself for anything not Hangul."""
    syllable = ord(char) - 0xAC00
    if 0 <= syllable < 19 * 21 * 28:
        parts = INITIALS[syllable // 588], MEDIALS[syllable // 28 % 21], FINALS[syllable % 28].strip()
        return "".join(JAMO_KEYS[part] for part in parts if part)
    return JAMO_KEYS.get(char, char)


def keys(*hints):
    """Key hints for the bottom line, keys(("p", "processes"), ...); clicking one presses its key."""
    out = []
    for key, label in hints:
        press = ("key", HINT_KEYS.get(key, key))
        out += ([("   ", "")] if out else []) + [(key, "", press), (" " + label, DIM, press)]
    return out


def footer(width, newer=None, picker=()):
    """Bottom line: the GPU `picker`, a newer release when there is one with how to update | the keys."""
    picker, notice, how, link = list(picker), [], [], []
    if newer:
        version, command = newer
        # A newer release is news, not trouble: the theme's accent, not warning yellow.
        notice = ([("   ", "")] if picker else []) + [("update available: " + version, THEME.accent)]
        how = [(" ({})".format(command or "new nvmon.py: " + RELEASES), "")]
        link = [("   what's new: " + RELEASES, DIM)] if command else []  # a copied nvmon.py already links there
    every = keys(("g", "graphs"), ("c", "theme"), ("p", "processes"), ("Esc / q", "quit"))
    few = keys(("p", "processes"), ("Esc / q", "quit"))
    # As much as fits: the link goes first, then how to update, the keys for graphs and theme, the picker.
    options = [(picker + notice + how + link, every), (picker + notice + how, every), (picker + notice + how, few),
               (picker + notice, few), (notice[1:] if picker else notice, few)]
    left, right = next((option for option in options if width_of(option[0]) + 2 + width_of(option[1]) <= width),
                       options[-1])
    return spread(width, left, right)


MOVES = {"tab": 1, "down": 1, "up": -1, "backtab": -1, "pgdn": 10, "pgup": -10, "home": -10**6, "end": 10**6}


class View:
    """What the screen shows around the numbers, and what keys and clicks do to it: the selected job,
    only its GPUs, the process list, a question before stopping, a note on how that went."""

    def __init__(self):
        self.shows = "truecolor"          # the colors the terminal shows, see color_mode
        self.theme = THEMES[0]            # the look, see THEMES (c)
        self.graphs = True                # the boxes show the utilization graph, else they are cards
        self.hidden, self.indices = set(), []  # the GPUs left out, and all there are
        self.job = None                   # the selected job's PID
        self.only = False                 # show only the GPUs it uses
        self.listing = False              # the process list is open
        self.top, self.follow = 0, True   # the list's first job; keep the selected one in view
        self.sort = ("GPUs", False)       # the list's order, and the order Tab and the arrows go in
        self.asking = None                # "SIGTERM" or "SIGKILL" awaiting y or n
        self.note = None                  # (segments, until): what the last stop did
        self.sent = None                  # (PID, when) of a SIGTERM, to point at k if the job lives on
        self.lines, self.jobs = [], []    # the last screen drawn, and the jobs on it
        self.gpu_top, self.gpu_page = 0, 1  # the first row of boxes shown, and how many rows fit
        self.list_rows = range(0)         # the screen rows the list takes, when it is open
        self.shift, self.shifted = 0, None  # how far the selected job's command is scrolled, and whose
        self.clicked = (0, None)          # (when, what): the last click, to tell a double click

    def restore(self, saved, indices):
        """Take up what an earlier run kept (Settings.load); of the GPUs `indices`, one always stays shown."""
        self.graphs = saved.get("graphs", self.graphs)
        self.theme = next((kept for kept in THEMES if kept.name == saved.get("theme")), self.theme)
        self.sort = saved.get("sort", self.sort)
        hidden = saved.get("hidden", set()) & set(indices)
        self.hidden = hidden if len(hidden) < len(indices) else set()

    def kept(self):
        """What is kept for the next run: see Settings."""
        return {"graphs": self.graphs, "theme": self.theme.name, "sort": self.sort, "hidden": set(self.hidden)}

    def selected(self):
        return next((job for job in self.jobs if job.pid == self.job), None)

    def screen(self, gpus, width, height, top_line, newer):
        """The whole screen: `top_line`, the GPU boxes (and the list, when open), the bottom line."""
        global THEME, SHOWS
        THEME, SHOWS = self.theme, self.shows
        column, descending = self.sort
        self.jobs = sorted(jobs(gpus), key=lambda job: (SORTS[column](job), job.pid), reverse=descending)
        job = self.selected()
        if job is None:  # nothing selected, or its job ended
            self.job, self.only, self.asking = None, False, None
        self.remind()
        self.indices = [g.index for g in gpus]
        shown = [g for g in gpus if (g.index in job.gpus if self.only else g.index not in self.hidden)]
        rows = min(len(self.jobs) + 2, max(4, (height - 2) // 2)) if self.listing else 0
        body = self.boxes(shown, width, height - 2 - rows)
        self.list_rows = range(1 + len(body), 1 + len(body) + rows)
        if rows:
            if self.shifted != self.job:
                self.shift, self.shifted = 0, self.job
            room = max(0, width - COMMAND_AT - 1)
            self.shift = max(0, min(self.shift, len(job.command) + 1 - room)) if job else 0
            body += process_list(self.jobs, width, rows, self.job, self.scrolled(rows - 2), self.sort, self.shift)
        self.lines = [top_line] + body + [self.bottom_line(width, job, newer, gpus)]
        return self.lines

    def boxes(self, gpus, width, height):
        """The GPU boxes in `height` lines; when not all fit, the rows from gpu_top on and a line on that."""
        cols, inner = layout(len(gpus), width, height, self.graphs)
        rows = math.ceil(len(gpus) / cols)
        if rows * (inner + 2) <= height:
            self.gpu_top, self.gpu_page = 0, rows
            return render(gpus, width, height, self.job, (cols, inner), self.graphs)
        self.gpu_page = max(1, (height - 1) // (inner + 2))
        self.gpu_top = max(0, min(self.gpu_top, rows - self.gpu_page))
        part = gpus[self.gpu_top * cols:(self.gpu_top + self.gpu_page) * cols]
        note = "GPUs {} of {} shown · wheel or PgUp/PgDn for the rest".format(
            ranges([g.index for g in part]), len(gpus))
        return (render(part, width, height - 1, self.job, (cols, inner), self.graphs)
                + [spread(width, [], [(note, DIM)])])

    def detail(self, gpus):
        """The GPUs whose processes' utilization is shown: all with the list open, else the selected
        job's. Only those are asked, as the driver takes some 2 ms a GPU to answer."""
        if self.listing:
            return {g.index for g in gpus}
        job = self.selected()
        return set(job.gpus) if job else set()

    def remind(self):
        """A job still there 5 s after its SIGTERM may be stuck, or slow to save: say that k ends it."""
        if not self.sent:
            return
        pid, when = self.sent
        job = next((job for job in self.jobs if job.pid == pid), None)
        if job and time.monotonic() - when > 5:
            self.note = ([("{} is still running 5 s after SIGTERM; k kills it at once".format(job.name), WARN)],
                         time.monotonic() + 6)
        if not job or time.monotonic() - when > 5:
            self.sent = None

    def scrolled(self, room):
        """The list's first job, kept in range and, after the selection moved, on the selected job."""
        order = [job.pid for job in self.jobs]
        if self.follow and self.job in order:
            i = order.index(self.job)
            self.top = min(max(self.top, i - room + 1), i)
        self.top = max(0, min(self.top, len(order) - room))
        return self.top

    def picker(self, gpus):
        """"GPUs 0 1 2 3" for the bottom line, the hidden ones faint; a click on a number (or the key)
        hides that GPU or shows it again."""
        if len(gpus) < 2:
            return []
        return [("GPUs", DIM)] + [(" {}".format(g.index), FAINT if g.index in self.hidden else BOLD, ("gpu", g.index))
                                  for g in gpus]

    def bottom_line(self, width, job, newer, gpus):
        if self.asking:
            return spread(width, [(self.question(job), WARN)], keys(("y", "yes"), ("n", "no")))
        note = self.note[0] if self.note and time.monotonic() < self.note[1] else None
        if job:
            hints = [("f", "all GPUs" if self.only else "only these GPUs")]
            hints += [("t", "stop"), ("k", "kill")] if job.owner is None and os.name != "nt" else []
            left = [(job.name, THEME.accent, job.pid), ("  ", DIM)] + about(job)
            return spread(width, note or left, keys(*hints, ("Esc", "back")))
        if self.listing:
            return spread(width, note or [("click a job or a heading", DIM)],
                          keys(("s", "sort by next"), ("r", "reverse"), ("Esc", "close")))
        if note:
            return spread(width, note, keys(("p", "processes"), ("Esc / q", "quit")))
        return footer(width, newer, self.picker(gpus))

    def question(self, job):
        """"stop train.py: SIGTERM to its torchrun (PID 48213), which ends its 2 workers?" and the like."""
        what = "stop" if self.asking == "SIGTERM" else "kill"
        if not job.launcher:
            return "{} {} (PID {}) with {}?".format(what, job.name, job.pid, self.asking)
        passed = ", which ends its" if self.asking == "SIGTERM" else " and to its"
        return "{} {}: {} to its {} (PID {}){} {} workers?".format(what, job.name, self.asking, job.launcher,
                                                                  job.pid, passed, len(job.members))

    def target(self, row, col):
        """What a click at (row, col) of the last screen means; None for nothing."""
        if 0 <= row < len(self.lines):
            x = 0
            for seg in self.lines[row]:
                if x <= col < x + len(seg[0]):
                    return seg[2] if len(seg) > 2 else None
                x += len(seg[0])
        return None

    def handle(self, event):
        """Act on one key or mouse event; False means quit."""
        kind, value = event[0], event[1:]
        if kind == "wheel":  # over the list it scrolls the list, or the selected job's command sideways;
            step, row, col = value  # anywhere else, the GPU boxes
            if row in self.list_rows:
                if col >= COMMAND_AT and self.target(row, col) == self.job:
                    self.shift = max(0, self.shift + 8 * step)
                else:
                    self.top, self.follow = self.top + 3 * step, False
            else:
                self.gpu_top += step
            return True
        if kind == "list":  # "+" under a GPU: the list, from that GPU's first job
            self.listing, self.follow = True, False
            self.top = next((i for i, job in enumerate(self.jobs) if value[0] in job.gpus), self.top)
            return True
        if kind == "gpu":  # a GPU's number at the bottom: hide that GPU, or show it again; one always stays
            index = value[0]
            if index in self.hidden:
                self.hidden.discard(index)
            elif len(self.hidden) + 1 < len(self.indices):
                self.hidden.add(index)
            return True
        if kind == "sort":  # a heading clicked: sort by it; again, the other way round
            column, descending = self.sort
            self.sort, self.follow = (value[0], not descending if value[0] == column else False), True
            return True
        if kind == "click":
            target = self.target(*value)
            if isinstance(target, tuple):  # a key hint, a heading of the list, "+" under a GPU
                return self.handle(target)
            # A job picks it; the picked job again, or a place meaning nothing, lets it go, and outside the
            # list closes that too. A double click on a job picks it and shows only its GPUs, or all again.
            now = time.monotonic()
            if target is None and value[0] not in self.list_rows:
                self.listing = False
            if target is not None and self.clicked[1] == target and now - self.clicked[0] < 0.4:
                self.job, self.only = target, not self.only
            else:
                self.job = target if target is not None and target != self.job else None
            self.asking, self.follow, self.clicked = None, True, (now, target)
            return True
        key = value[0].lower() if len(value[0]) == 1 else value[0]
        if self.asking:  # y stops; anything else, n and Esc among them, lets it be
            if key == "y":
                note, sent = stop(self.selected(), self.asking)
                self.note = (note, time.monotonic() + 4)
                self.sent = (self.job, time.monotonic()) if sent and self.asking == "SIGTERM" else None
            self.asking = None
            return key != "q"
        if key == "q":
            return False
        if key == "esc":  # one step back; from the plain screen, quit
            if self.only:
                self.only = False
            elif self.listing:
                self.listing = False
            elif self.job is not None:
                self.job = None
            else:
                return False
        elif key == "p":
            self.listing = not self.listing
        elif key == "g":
            self.graphs = not self.graphs
        elif key == "c":  # the next theme; C the one before
            self.theme = THEMES[(THEMES.index(self.theme) + (-1 if value[0] == "C" else 1)) % len(THEMES)]
            shows = "" if self.shows == "truecolor" else ", in 256 colors: all this terminal shows"
            self.note = ([("theme: " + self.theme.name + shows, "")], time.monotonic() + 3)
        elif key.isdigit() and int(key) in self.indices:
            self.handle(("gpu", int(key)))
        elif self.listing and key in ("s", "r"):  # the next column, or the other way round
            names, (column, descending) = list(SORTS), self.sort
            if key == "s":
                self.sort = (names[(names.index(column) + 1) % len(names)], False)
            else:
                self.sort = (column, not descending)
            self.follow = True
        elif key in ("pgup", "pgdn") and not self.listing:  # a page of GPU boxes
            self.gpu_top += self.gpu_page * (1 if key == "pgdn" else -1)
        elif key in ("left", "right") and self.listing:  # the selected job's command, sideways
            self.shift = max(0, self.shift + (8 if key == "right" else -8))
        elif key in MOVES and self.jobs:
            order = [job.pid for job in self.jobs]
            i = order.index(self.job) if self.job in order else (-1 if MOVES[key] > 0 else len(order))
            self.job, self.follow = order[min(max(i + MOVES[key], 0), len(order) - 1)], True
        elif self.job is not None and key in ("f", "enter"):
            self.only = not self.only
        elif self.job is not None and key in ("t", "k"):
            if self.selected().owner is None:
                self.asking = "SIGTERM" if key == "t" else "SIGKILL"
            else:
                self.note = ([("only your own processes can be stopped", WARN)], time.monotonic() + 3)
        return True


# For terminals that draw the line, block and other characters above two cells wide, as they may for Korean,
# Chinese and Japanese (Unicode calls their width "ambiguous"): ASCII look-alikes, one for one.
ASCII = str.maketrans({"─": "-", "│": "|", "├": "+", "┤": "+", "╭": "+", "╮": "+", "╰": "+", "╯": "+",
                       "▁": "_", "▂": "_", "▃": "_", "▄": "=", "▅": "=", "▆": "#", "▇": "#", "█": "#",
                       "▀": " ",  # the graph's full cells: their background alone, a solid block
                       "°": " ", "·": "-", "…": "~", "↑": "^", "↓": "v", "▲": "^", "▼": "v"})


def paint(lines, mode="truecolor", ascii=False, gray=False):
    """Join lines into one string in color `mode` (with `gray`, grays only; see restyle), and with `ascii` in
    ASCII, sending a style code only where the style changes."""
    out, current = [], ""
    for i, line in enumerate(lines):
        if i:
            out.append("\r\n")
        for text, style, *_ in line:
            if ascii:
                text = text.translate(ASCII)
            if mode != "truecolor" or gray:
                style = restyle(style, mode, gray)
            if style != current:
                # One color replaces another directly; anything else (bold, a background, plain) needs a
                # reset first.
                colors = (style.startswith("\x1b[38") and current.startswith("\x1b[38")
                          and "\x1b[48" not in current)
                out.append(style if colors else RESET + style)
                current = style
            out.append(text)
    return "".join(out) + RESET


# ── settings ─────────────────────────────────────────────────────────────────

def client():
    """Where the person looking sits: the address their ssh session comes from, else "local". Settings go by
    it, as one account is often used from several computers, each with a terminal of its own."""
    for name in ("SSH_CONNECTION", "SSH_CLIENT"):
        address = os.environ.get(name, "").split()
        if address:
            return address[0]
    return "local"


class Settings:
    """What stays from one run to the next, for each client (see client): graphs on or off, the theme, the
    list's order, and the GPUs hidden, those for each machine, as several often share a home directory.
    Kept in ~/.config/nvmon/settings.json (%APPDATA%\\nvmon on Windows); a file that cannot be read or written
    only means that nothing is kept."""

    def __init__(self):
        base = (os.environ.get("APPDATA") if os.name == "nt" else None) or os.environ.get("XDG_CONFIG_HOME")
        base = base or os.path.join(os.path.expanduser("~"), ".config")
        self.path, self.client = os.path.join(base, "nvmon", "settings.json"), client()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def load(self, host):
        """This client's settings as View.restore takes them, the GPUs hidden those on `host`."""
        mine = self._read().get(self.client)
        mine = mine if isinstance(mine, dict) else {}
        out = {"graphs": mine["graphs"]} if isinstance(mine.get("graphs"), bool) else {}
        if isinstance(mine.get("theme"), str):
            out["theme"] = mine["theme"]
        sort = mine.get("sort")
        if isinstance(sort, list) and len(sort) == 2 and sort[0] in SORTS and isinstance(sort[1], bool):
            out["sort"] = tuple(sort)
        hidden = mine.get("hidden")
        if isinstance(hidden, dict) and isinstance(hidden.get(host), list):
            out["hidden"] = {i for i in hidden[host] if isinstance(i, int)}
        return out

    def save(self, kept, indices, host):
        """Keep `kept` (View.kept) for this client; for GPUs that are not `indices` (nvmon -g), what was
        kept for them. Everyone else's settings stay as the file has them now."""
        data = self._read()
        mine = data.get(self.client) if isinstance(data.get(self.client), dict) else {}
        hidden = mine.get("hidden") if isinstance(mine.get("hidden"), dict) else {}
        earlier = hidden.get(host) if isinstance(hidden.get(host), list) else []
        hidden[host] = sorted({i for i in earlier if isinstance(i, int) and i not in indices} | kept["hidden"])
        mine.update(graphs=kept["graphs"], theme=kept["theme"], sort=list(kept["sort"]),
                    hidden={name: gpus for name, gpus in hidden.items() if gpus})
        data[self.client] = mine
        temp = "{}.{}".format(self.path, os.getpid())
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, sort_keys=True)
            os.replace(temp, self.path)  # at once: a run stopped halfway leaves the old file whole
        except OSError:  # a home that is read-only, full, not there
            try:
                os.remove(temp)
            except OSError:
                pass


# ── terminal ─────────────────────────────────────────────────────────────────

INPUT = re.compile(rb"\x1b\[<(?P<button>\d+);(?P<x>\d+);(?P<y>\d+)(?P<act>[Mm])"  # mouse, SGR encoding
                   rb"|\x1b\[M(?P<old>...)"                                     # mouse, the old encoding
                   rb"|\x1b[\[O](?P<args>[?>0-9;]*)(?P<end>[A-Za-z~])"           # arrows, Home, PgUp and such
                   rb"|(?P<report>\x1bP[^\x1b]*\x1b\\)"                           # a late answer to Screen.probe
                   rb"|(?P<text>[\xc2-\xf4][\x80-\xbf]+)"                          # a letter beyond ASCII, as UTF-8
                   rb"|(?P<char>.)", re.DOTALL)                                 # any other key; a lone ESC is Esc


def events(data):
    """The input in `data`, the bytes a terminal sends: ("key", name) events, ("click", row, column) for
    a left-button press, ("wheel", +1 down or -1 up, row, column)."""
    out = []
    for m in INPUT.finditer(data):
        if m.group("button") or m.group("old"):
            if m.group("button"):
                button, x, y = (int(m.group(g)) for g in ("button", "x", "y"))
                press = m.group("act") == b"M"
            else:
                button, x, y = (c - 32 for c in m.group("old"))
                press = button & 3 != 3
            if button & 64:
                out.append(("wheel", 1 if button & 1 else -1, y - 1, x - 1))
            elif press and button & 3 == 0 and not button & 32:  # left button down, not a drag
                out.append(("click", y - 1, x - 1))
        elif m.group("report"):
            continue
        elif m.group("end"):
            args, end = m.group("args").decode(), m.group("end").decode()
            name = SEQUENCE_KEYS.get(args + end) or SEQUENCE_KEYS.get(end)
            if name:
                out.append(("key", name))
        elif m.group("text"):  # Hangul among them, see typed
            out += [("key", key) for char in m.group("text").decode("utf-8", "replace") for key in typed(char)]
        else:
            char = m.group("char").decode("latin-1")
            out.append(("key", CONTROL_KEYS.get(char, char)))
    return out


WINDOWS_KEYS = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "I": "pgup", "Q": "pgdn", "\x0f": "backtab"}


class Screen:
    """Alternate screen, hidden cursor, no auto-wrap, unbuffered keys, mouse clicks and wheel; all
    restored on exit."""

    def __enter__(self):
        self._keys = sys.stdin.isatty()
        self._mouse = self._keys and os.name != "nt"  # Windows reads keys with msvcrt, which sees no mouse
        self._restore = [enable_vt(), raw_keys() if self._keys else (lambda: None)]
        # 1000 + 1006: clicks and the wheel come in as SGR-encoded sequences (Shift+drag still selects text).
        self._write("\x1b[?1049h\x1b[?25l\x1b[?7l" + ("\x1b[?1000h\x1b[?1006h" if self._mouse else ""))
        # Windows Terminal and the console of Windows 10 on show 24-bit color and draw lines one cell wide.
        self.wide, self.truecolor = self.probe() if self._mouse else (False, os.name == "nt")
        return self

    def probe(self):
        """(wide, truecolor), as the terminal answers: whether it draws a line character two cells wide, and
        whether it keeps a 24-bit color. Every terminal answers the last question, DA1, so that answer ends
        the wait, a round trip even over ssh; but for the answer on the color, which may come after it: the
        ConPTY between Windows's ssh and Windows Terminal answers DA1 itself at once and passes DECRQSS on.
        xterm.js (Tabby, VS Code) shows 24-bit color but answers DECRQSS with "0m" whatever is set; it is
        told by that together with its own DA2 answer, ">0;276;0". No answer within a second means neither."""
        self._write("\x1b[H─\x1b[6n"                  # where the cursor is after one line character
                    "\x1b[38;2;1;2;3m\x1bP$qm\x1b\\"  # DECRQSS: the color set now, as the terminal kept it
                    + RESET + "\x1b[>c\x1b[c")          # DA2: which terminal; DA1: what kind, answered last
        import select
        fd, data, end = sys.stdin.fileno(), b"", time.monotonic() + 1
        answered = False  # DA1 is in
        while True:
            if not answered and re.search(rb"\x1b\[\?[0-9;]*c", data):
                answered, end = True, min(end, time.monotonic() + 0.25)  # a quarter second more for DECRQSS
            if answered and re.search(rb"\x1bP[01]\$r[^\x1b]*\x1b\\", data):
                break
            left = end - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                break
            more = os.read(fd, 1024)
            if not more:
                break
            data += more
        cursor = re.search(rb"\x1b\[\d+;(\d+)R", data)
        color = re.search(rb"\x1bP1\$r([0-9;:]*)m", data)
        kept = bool(color) and bool(re.search(rb"38[;:]2[;:]+1[;:]2[;:]3", color.group(1)))
        xterm_js = b"\x1b[>0;276;0c" in data and bool(color) and color.group(1) == b"0"
        return bool(cursor) and int(cursor.group(1)) > 2, kept or xterm_js

    def __exit__(self, *exc):
        self._write(RESET + ("\x1b[?1006l\x1b[?1000l" if self._mouse else "") + "\x1b[?7h\x1b[?25h\x1b[?1049l")
        for undo in self._restore:
            undo()

    def read(self, seconds):
        """Key and mouse events as soon as there are some; [] once `seconds` pass without any."""
        if not self._keys:
            time.sleep(max(0, seconds))
            return []
        if os.name == "nt":
            import msvcrt
            end = time.monotonic() + seconds
            while True:
                found = []
                while msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char in ("\x00", "\xe0"):  # arrows and the like come as two characters
                        name = WINDOWS_KEYS.get(msvcrt.getwch())
                        found += [("key", name)] if name else []
                    else:
                        found += [("key", CONTROL_KEYS.get(char) or key) for key in typed(char)]
                left = end - time.monotonic()
                if found or left <= 0:
                    return found
                time.sleep(min(left, 0.02))
        import select
        fd = sys.stdin.fileno()
        if not select.select([fd], [], [], max(0, seconds))[0]:
            return []
        data = os.read(fd, 1024)
        if not data:  # stdin closed: nothing more will come
            self._keys = False
        return events(data)

    def draw(self, lines, mode, gray):
        # 2026 = synchronized output: terminals that know it swap the frame in at once.
        self._write("\x1b[?2026h\x1b[H" + paint(lines, mode, self.wide, gray) + "\x1b[?2026l")

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
    anything faster would only keep the driver busier and counts as MIN_INTERVAL; what is not a number of
    seconds (abc, nan, inf) means the default."""
    try:
        seconds = float(text)
    except ValueError:
        return DEFAULT_INTERVAL
    return max(MIN_INTERVAL, seconds) if math.isfinite(seconds) else DEFAULT_INTERVAL


def main():
    parser = argparse.ArgumentParser(
        prog="nvmon", description="A fancy NVIDIA GPU monitor for the terminal.",
        epilog="At start nvmon asks PyPI whether a newer release is out; NVMON_NO_UPDATE_CHECK=1 turns that "
               "off. Colors are 24-bit where the terminal says it shows them, else 256. Graphs on or off, the theme, "
               "the list's order and hidden GPUs are kept for each computer you connect from, in "
               "~/.config/nvmon/settings.json. " + REPO)
    parser.add_argument("-i", "--interval", type=interval, default=DEFAULT_INTERVAL, metavar="SEC",
                        help="seconds between updates, at least 0.1 (default: 0.5)")
    parser.add_argument("-g", "--gpus", type=gpu_list, metavar="LIST",
                        help="only these GPUs, e.g. 0,2,4-7 (default: all)")
    parser.add_argument("-V", "--version", action="version", version="nvmon {} {}".format(__version__, REPO))
    args = parser.parse_args()
    if not sys.stdout.isatty():
        sys.exit("nvmon: output is not a terminal (over ssh, use: ssh -t HOST nvmon)")

    # NVML is never shut down: exiting frees it as well, while nvmlShutdown can keep a busy driver, and so
    # the shell prompt, waiting for a second.
    try:
        nv = load_nvml()
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
        host, settings, indices = socket.gethostname(), Settings(), [g.index for g in gpus]
        poller, view = Poller(gpus), View()
        view.restore(settings.load(host), indices)
        kept = view.kept()
        with Screen() as screen:
            view.shows = color_mode(screen.truecolor)
            # peek = (GPUs, round): when more GPUs' processes are to show their utilization, as after a click on
            # a job, the screen takes the newest numbers as soon as those GPUs have been read, not at the next
            # frame; or once that round of polls is over, for a GPU that cannot tell (with MIG, say).
            frame_at, poll_at, redraw, peek = time.monotonic(), None, True, None
            while True:
                now = time.monotonic()
                if now >= frame_at:  # a new frame: the newest numbers, one more graph column each
                    for g in gpus:
                        g.frame()
                    frame_at = max(frame_at + args.interval, now)  # a late frame restarts the schedule
                    # The polls for the next frame start ahead of it: half a tick, at most 0.5 s (a busy
                    # driver was seen to take 0.4 s). Should the driver answer later still, that frame keeps
                    # the last numbers but comes on time.
                    poll_at, redraw = frame_at - min(args.interval / 2, 0.5), True
                if poll_at is not None and now >= poll_at:
                    poller.refresh()
                    poll_at = None
                if peek is not None:
                    wanted, last = peek
                    if poller.rounds >= last or all(g.sample.process_util is not None for g in gpus
                                                    if g.index in wanted):
                        for g in gpus:  # the graphs move on at the next frame, as ever
                            g.now = g.sample
                        peek, redraw = None, True
                    else:
                        poller.refresh()  # after the round running, should it have passed those GPUs
                if redraw:
                    width, height = os.get_terminal_size()
                    screen.draw(view.screen(gpus, width, height,
                                            header(width, args.interval, driver, poller.waiting()), update.newer),
                                view.shows, view.theme.gray)
                # Keys and clicks redraw at once; the numbers move on at the next frame.
                events = screen.read(0.01 if peek is not None else (poll_at or frame_at) - time.monotonic())
                going = all(map(view.handle, events))
                if view.kept() != kept:  # a setting changed: keep it for next time
                    kept = view.kept()
                    settings.save(kept, indices, host)
                if not going:
                    break
                redraw, detail = bool(events), view.detail(gpus)
                if detail - poller.detail:  # at the latest the next round, or the one after the round running
                    busy = not poller.done.is_set()
                    peek = (detail - poller.detail | (peek[0] if peek else set()), poller.rounds + 1 + busy)
                poller.detail = detail
    except KeyboardInterrupt:
        pass
    except NvmlError as e:
        sys.exit("nvmon: {}".format(e))


if __name__ == "__main__":
    main()
