#!/usr/bin/env python3
"""nvmon: a fancy NVIDIA GPU monitor for the terminal. One Python file, no dependencies.

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

__version__ = "0.3.4+dev"  # "+dev": work past this release; the release commit sets the next number
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
# 24-bit color or 256 colors, as color_mode finds; in gray, as the black and white themes have it, and for
# black on white the GPU boxes on white (panel marks their styles with ON_PAPER).
PAPER, INK = (242, 242, 242), (30, 30, 30)
ON_PAPER = "\x1b[48;2;242;242;242m"
COLOR_ARG = re.compile(r"\x1b\[([34])8;(?:2;(\d+);(\d+);(\d+)|5;(\d+))m")


@lru_cache(maxsize=None)
def restyle(style, mode, gray=False):
    """`style` in color `mode`: "256" turns 24-bit colors into the palette's nearest. With `gray`, every
    color first the gray as light as it is; or, for a style marked ON_PAPER, on PAPER in INK, as dark as it
    was light (OKLab lightness turned around): light text on the dark screen becomes dark on the white one,
    and the busiest bars the darkest. FADED on PAPER is drawn as text halfway to the paper, since many
    terminals draw faint text darker, which on white stands out rather than fades."""
    if mode == "truecolor" and not gray:
        return style
    paper = gray and style.startswith(ON_PAPER)
    faint = paper and style.startswith(FADED, len(ON_PAPER))
    style = style[len(ON_PAPER) + len(FADED) * faint:] if paper else style

    def swap(m):
        layer, code = m.group(1), m.group(5)  # layer 3: the foreground, 4: the background
        if code and not gray:  # a palette color already
            return m.group(0)
        rgb = _palette_rgb(int(code)) if code else tuple(int(v) for v in m.group(2, 3, 4))
        if gray:
            lightness = _oklab(rgb)[0]
            rgb = ink(layer, 1 - lightness) if paper else _from_oklab((lightness, 0, 0))
        return color(layer, rgb)

    def ink(layer, lightness):
        if faint and layer == "3":
            lightness = (lightness + _oklab(PAPER)[0]) / 2
        return _from_oklab((lightness, 0, 0))

    def color(layer, rgb):
        if mode == "256":
            return "\x1b[{}8;5;{}m".format(layer, _nearest256(rgb))
        return "\x1b[{}8;2;{};{};{}m".format(layer, *rgb)
    style = COLOR_ARG.sub(swap, style)
    if not paper:
        return style
    return color("4", PAPER) + color("3", ink("3", _oklab(INK)[0])) + style


def color_mode(truecolor):
    """"truecolor" where the terminal is known to show 24-bit color (it said so, see Screen.probe, or
    COLORTERM does, or LC_TERMINAL, which iTerm2 sets and ssh passes on), else "256": every terminal of today
    shows those, and they look nearly the same, while 24-bit color where it is not understood comes out in
    odd colors."""
    known = os.environ.get("COLORTERM") in ("truecolor", "24bit") or os.environ.get("LC_TERMINAL") == "iTerm2"
    return "truecolor" if truecolor or known else "256"


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
    gray: bool        # every color the gray of its lightness: see restyle
    paper: bool       # and the GPU boxes black on white


def theme(name, stops, accent, temp=None, gray=False, paper=False):
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
                 tuple(_rgb(_readable(c)) for c in heat_of_temp), accent, "\x1b[7m" + accent,
                 gray or paper, paper)


GREEN, YELLOW, ORANGE, RED = (95, 175, 95), (215, 215, 95), (215, 135, 95), (215, 95, 95)
# The themes c goes through. First nvmon's own: utilization and shares green -> yellow -> orange -> red, the
# yellow-green at 30 %, not 20, as from green to it is a long way to the eye; temperature in °C, blue at 30 to
# red at 88, as idle GPUs sit at 30-45, busy ones at 60-80, and most throttle from about 85-90. Then palettes
# made to look good, low to high, the darkest end dropped where it would vanish into the background: from
# works at the Metropolitan Museum, as MetBrewer (Blake R. Mills) takes them, Hiroshige's "Sailing Boats
# Returning to Yabase", Hokusai's "Yoro Waterfall", O'Keeffe's "Red and Yellow Cliffs", Van Gogh's "First
# Steps", the pinks of Benedictus's "Relais", the lilacs of Cassatt's "Lilacs in a Window", a Greek
# terracotta neck-amphora of about 550 B.C., Vivienne Tam's "Dragon Robe", Winslow Homer's "The Veteran in a
# New Field" and Demuth's "I Saw the Figure 5 in Gold"; and the Rose Pine editor theme.
THEMES = [
    theme("nvmon", [(0, GREEN), (30, (175, 215, 95)), (40, YELLOW), (60, (215, 175, 95)), (80, ORANGE), (100, RED)],
          _fg(110), [(30, (95, 135, 215)), (45, (95, 175, 175)), (60, GREEN), (72, YELLOW), (80, ORANGE), (88, RED)]),
    theme("Hiroshige", "#376795 #528fad #72bcd5 #aadce0 #ffe6b7 #ffd06f #f7aa58 #ef8a47 #e76254", "#aadce0"),
    theme("Hokusai", "#295384 #5a97c1 #74c8c3 #95c36e #d8d97a", "#74c8c3"),
    theme("O'Keeffe", "#92351e #b9563f #d37750 #e69c6b #ecb27d #f2c88f #fbe3c2", "#f2c88f"),
    theme("Van Gogh", "#1f5b25 #3c7c3d #669d62 #9cc184 #c2d6a4 #e7e5cc", "#c2d6a4"),
    theme("Benedictus", "#9a133d #b93961 #d8527c #f28aaa #f9b4c9 #f9e0e8", "#f28aaa"),
    theme("Cassatt", "#574571 #90719f #b695bc #dec5da", "#b695bc"),
    theme("Greek", "#8d1c06 #e67424 #ed9b49 #f5c34d", "#ed9b49"),
    theme("Tam", "#9f2d55 #bb292c #de4f33 #ef8737 #ffb242 #ffd353", "#ffb242"),
    theme("Homer", "#a62f00 #df7700 #f5b642 #fff179 #c3f4f6 #6ad5e8 #32b2da", "#f5b642"),
    theme("Demuth", "#41485f #5d6174 #8b8b99 #b9b9b8 #f7c267 #d39a2d #b64f32 #9b332b #591c19", "#f7c267"),
    theme("Rose Pine", "#3e8fb0 #9ccfd8 #c4a7e7 #ea9a97 #eb6f92", "#c4a7e7"),
]
# No hue, only lightness (see restyle): dark gray when idle to near white when busy; and the same black on
# white, the lightest when idle to near black when busy.
THEMES += [theme("black and white", "#505050 #f0f0f0", "#d0d0d0", gray=True),
           theme("black on white", "#505050 #f0f0f0", "#d0d0d0", paper=True)]
THEME = THEMES[0]  # the one in use, set by View.screen before it draws
SHOWS = "truecolor"  # what the terminal shows (color_mode), set with it
# Warnings on the top edge, the same in every theme: yellow = worth a look, orange = slowed, red = act.
WARN, SLOW, ALERT = _rgb(YELLOW), _rgb(ORANGE), _rgb(RED)
DIM, FAINT = _fg(240), _fg(238)
UNDERLINE = "\x1b[4m"  # first in a style, so that the next one starts with a reset (see paint)
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
    owner: Optional[str]          # account; None where unknown
    mine: bool                    # ours: its account underlined, and only ours can be stopped
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
        self.history = deque(maxlen=HISTORY)  # utilization, one a frame: the graph
        self.memory = deque(maxlen=HISTORY)   # and the memory in use then (GiB), for the chip (View.hover)
        self.sample = None  # the newest reading, written by poll()
        self.now = None     # the reading on screen, taken from `sample` by frame()
        self.pcie_max_gen = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        self.pcie_max_width = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        self._gpm = self._gpm_samples()
        self.has_activity = self._gpm is not None
        self._facts = {}  # pid -> what _facts_of says: fixed for a process's life, so read once
        self._listed, self._listed_mem = [], None  # process list, and the memory in use when it was read
        # Per-process utilization: the driver's samples since `seen` (µs), read at most every second.
        self._util, self._util_due, self._util_seen = None, 0, 0
        self._sizes = {}  # entry type -> how many the last array of them held, see _entries

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

    def _entries(self, kind, query):
        """(return code, entries) of `query(array, count)`, a query that fills an array of `kind` and sets
        `count`. The array is made bigger while the driver says it is too small (more may come between its
        answer and asking again), and starts that big the next time."""
        size = self._sizes.get(kind, 64)
        for _ in range(3):
            array, count = (kind * size)(), c_uint(size)
            rc = query(array, count)
            if rc != NVML_ERROR_INSUFFICIENT_SIZE:
                break
            size = 2 * max(count.value, size)
        self._sizes[kind] = size
        return rc, array[:count.value] if rc == NVML_SUCCESS else []

    def _processes(self):
        # _v3 from driver R510 on; before it (R450 on) _v2, with the same entries.
        for query in ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses_v2"):
            rc, running = self._entries(ProcessInfo, lambda array, count: self._call(query, byref(count), array))
            if query not in self._unsupported:
                break
        if rc != NVML_SUCCESS:
            return []
        self._facts = {p.pid: self._facts.get(p.pid) or self._facts_of(p.pid) for p in running}
        procs = []
        for p in running:
            mem = None if p.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE else p.usedGpuMemory / 2**30
            name, command, owner, mine, env, started, job, launcher = self._facts[p.pid]
            procs.append(Process(p.pid, name, command, mem, owner, mine, env, started, job, launcher))
        return procs

    def _facts_of(self, pid):
        """(name, command, owner, ours, conda environment, start time, job, launcher name) of `pid`. A
        notebook's kernel is a job of its own: what started it, Jupyter or an editor, runs every notebook, and
        stopping that would stop them all."""
        command = self._command(pid)
        launcher = None if command[0].startswith("ipykernel") else launcher_of(pid)
        return (command + process_facts(pid)
                + ((launcher, self._command(launcher)[0]) if launcher else (pid, None)))

    def _command(self, pid):
        """(short name, command line from it on) of `pid`. For a Python interpreter the name is the script
        or module it runs: ("train.py", "train.py --lr 3e-4"), ("torch.distributed.run", ...)."""
        argv = [a for a in (proc("{}/cmdline".format(pid)) or "").split("\0") if a]
        if not argv:  # not Linux, or the process already exited
            name = ctypes.create_string_buffer(256)
            if self.nv.nvmlSystemGetProcessName(pid, name, 256) != NVML_SUCCESS:
                return str(pid), ""
            argv = [name.value.decode("utf-8", "replace")]
        exe, rest = os.path.basename(argv[0]), argv[1:]
        if exe.startswith("python"):
            value = False  # the argument before was an option that takes one, as -X faulthandler
            for i, arg in enumerate(rest):
                if arg == "-c":  # code, no script: the interpreter it is
                    break
                if arg == "-m" and i + 1 < len(rest):
                    return rest[i + 1], " ".join(rest[i + 1:])
                if not arg.startswith("-") and not value:
                    name = os.path.basename(arg)
                    return name, " ".join([name] + rest[i + 1:])
                value = arg in ("-X", "-W", "--check-hash-based-pycs")
        return exe, " ".join([exe] + rest)

    def _process_util(self):
        """{pid: % of the time its kernels ran} since the last call; None where the GPU cannot tell
        (with MIG, say). The driver takes some 2 ms to answer, so this is asked only while it is shown."""
        rc, samples = self._entries(ProcessUtil, lambda array, count: self._call(
            "nvmlDeviceGetProcessUtilization", array, byref(count), c_ulonglong(self._util_seen)))
        if rc == NVML_ERROR_NOT_FOUND:  # no new samples: nothing ran
            return {}
        if rc != NVML_SUCCESS:
            return None
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
        self.memory.append(self.now.mem_used)


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
    """(owner, whether it is ours, conda environment, start time) of `pid`, from /proc or Windows; Nones,
    and not ours, where unknown."""
    if os.name == "nt":
        sid = windows_sid(pid)
        return sid and sid_name(sid), ours(pid), None, start_time(pid)
    uid = uid_of(pid)
    return account_name(uid), ours(pid), conda_env(pid), start_time(pid)


def ours(pid):
    """Whether `pid` runs as our account; False where unknown."""
    if os.name == "nt":
        sid = windows_sid(pid)
        return sid is not None and sid == windows_sid(os.getpid())
    uid = uid_of(pid)
    return uid is not None and uid == os.getuid()


def windows_process(pid):
    """A handle to query `pid` with, on Windows; None where that is not allowed (the system's processes, or
    another account's). To be closed (CloseHandle)."""
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    return c_void_p(handle) if handle else None


def windows_sid(pid):
    """The security identifier (SID) of the account running `pid`, on Windows, as bytes; None where unknown."""
    kernel32, advapi32 = ctypes.windll.kernel32, ctypes.windll.advapi32
    process = windows_process(pid)
    if process is None:
        return None
    token, info, size = c_void_p(), ctypes.create_string_buffer(256), ctypes.c_ulong()
    try:
        if not advapi32.OpenProcessToken(process, 0x0008, byref(token)):  # TOKEN_QUERY
            return None
        found = advapi32.GetTokenInformation(token, 1, info, len(info), byref(size))  # TokenUser
        kernel32.CloseHandle(token)
        if not found:
            return None
        sid = c_void_p.from_buffer(info)  # TOKEN_USER starts with the SID's address, within `info`
        return ctypes.string_at(sid.value, advapi32.GetLengthSid(sid))
    finally:
        kernel32.CloseHandle(process)


@lru_cache(maxsize=None)
def sid_name(sid):
    """The account name of `sid` (windows_sid); looked up once each, as it may ask a domain controller."""
    name, domain = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
    sizes, kind = (ctypes.c_ulong(256), ctypes.c_ulong(256)), ctypes.c_ulong()
    found = ctypes.windll.advapi32.LookupAccountSidW(None, sid, name, byref(sizes[0]), domain, byref(sizes[1]),
                                                     byref(kind))
    return name.value if found else None


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


def proc(path):
    """The text of /proc/`path`; None where there is none. UTF-8 whatever the locale, and a letter that is
    not (a process's name is cut at 15 bytes, inside one maybe) does not lose the rest."""
    try:
        with open("/proc/" + path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def stat_field(pid, n):
    """Field `n` of /proc/`pid`/stat, counted from 1 as `man proc` does, as a number; None where unknown.
    Counted after "(name)", as a name may hold spaces and parentheses."""
    try:
        return int(proc("{}/stat".format(pid)).rsplit(")", 1)[1].split()[n - 3])
    except (AttributeError, IndexError, ValueError):
        return None


@lru_cache(maxsize=None)
def boot_time():
    """When the machine started, in seconds since the epoch. Read once: the kernel's figure moves when the
    clock is set, and a process's start time is to stay the same all run long, as stop compares it."""
    lines = (proc("stat") or "").splitlines()
    return next((int(line.split()[1]) for line in lines if line.startswith("btime")), None)


def start_time(pid):
    """When `pid` started, in seconds since the epoch; None where unknown."""
    if os.name == "nt":
        process = windows_process(pid)
        if process is None:
            return None
        created, exited, kernel, user = (c_ulonglong() for _ in range(4))  # FILETIMEs: 100 ns since 1601
        found = ctypes.windll.kernel32.GetProcessTimes(process, *map(byref, (created, exited, kernel, user)))
        ctypes.windll.kernel32.CloseHandle(process)
        return created.value / 1e7 - 11644473600 if found else None
    ticks = stat_field(pid, 22)
    return None if ticks is None or boot_time() is None else boot_time() + ticks / os.sysconf("SC_CLK_TCK")


def account_name(uid):
    """The name of the account `uid`; None for None."""
    if uid is None:
        return None
    import pwd
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:  # no passwd entry, e.g. inside a container
        return str(uid)


def uid_of(pid):
    """The uid of the account running `pid`; None where unknown."""
    lines = (proc("{}/status".format(pid)) or "").splitlines()
    return next((int(line.split()[1]) for line in lines if line.startswith("Uid:")), None)


def parent_of(pid):
    """The PID of `pid`'s parent; None where unknown."""
    return stat_field(pid, 4)


# Parents that start each program on its own, rather than as the parts of one job (a notebook's kernel stands
# alone too, see Gpu._facts_of).
NOT_LAUNCHERS = {"bash", "sh", "dash", "zsh", "fish", "ksh", "tcsh", "csh", "tmux: server", "screen", "SCREEN",
                 "sshd", "sudo", "su", "login", "systemd", "init", "script", "nohup", "raylet"}


def launcher_of(pid):
    """The PID of what launched `pid` as part of a job, such as torchrun for its workers: its parent, unless
    that is PID 1, a shell or the like, or someone else's process. None when `pid` stands alone."""
    parent = parent_of(pid)
    if not parent or parent <= 1 or uid_of(parent) != uid_of(pid):
        return None
    name = proc("{}/comm".format(parent))
    return None if name is None or name.strip() in NOT_LAUNCHERS else parent


def holds_gpu(pid):
    """True when `pid` has an NVIDIA device open, as a GPU process in our PID namespace does. Windows does not
    tell, nor has it PID namespaces: True there, the start time telling it is the same process."""
    if os.name == "nt":
        return True
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
    mine: bool
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
        # The launcher's name from any of them: a launcher on a GPU itself has none of its own. The members by
        # PID, once each though seen on several GPUs, which may have read it at different times.
        launcher = next((p.launcher for p in procs if p.launcher), None)
        found.append(Job(pid, main.name, launcher, tuple(sorted({p.pid: p.started for p in procs}.items())),
                         tuple(sorted({gpu.index for gpu, _ in entries})),
                         sum(shares.values()) / len(shares) if shares else None, sum(p.mem or 0 for p in procs),
                         min(starts) if starts else None, main.owner, main.mine, main.env, main.command))
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


def gpu_numbers(indices):
    """"GPU 3", "GPUs 4-5,7"."""
    indices = sorted(set(indices))
    return "GPU{} {}".format("s" if len(indices) > 1 else "", ranges(indices))


def account(who):
    """"kim/torch": whose a process or job (`who`) is, and its conda environment; "" where neither is known."""
    return "/".join(part for part in (who.owner, who.env) if part)


def tagged(who):
    """account as segments: dim, and ours underlined, so that they stand apart without standing out."""
    text = account(who)
    return [(text, UNDERLINE + DIM if who.mine else DIM)] if text else []


def about(job):
    """"kim/torch · GPUs 4-5 · util 86% · 30.2G · 2h13m · torchrun job" as segments, util in its color;
    the launcher, often a long name, comes last, where a narrow line cuts first."""
    parts = [sum(tagged(job), ())] if account(job) else []
    parts.append((gpu_numbers(job.gpus), DIM))
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


def about_all(jobs):
    """"GPUs 4-7 · 61.2G · 2 yours" for several jobs picked together, as segments."""
    parts = [gpu_numbers(index for job in jobs for index in job.gpus), "{:.1f}G".format(sum(j.mem for j in jobs))]
    ours = sum(job.mine for job in jobs)
    parts += ["{} yours".format(ours)] if 0 < ours < len(jobs) else []
    return [(" · ".join(parts), DIM)]


def stop(job, name):
    """Send signal `name` ("SIGTERM" or "SIGKILL") to `job`, once /proc or Windows confirms its PIDs still
    are the processes we saw: started when we saw them, holding a GPU open (inside a container, NVML can
    report PIDs of the host), and ours. While its launcher still leads them (their parent, ours), SIGTERM
    goes to the launcher, which then ends its workers, and SIGKILL, which it cannot pass on, to each worker
    as well; with no launcher, or one gone (its workers left to PID 1), to each process. On Windows, which
    has no SIGTERM and no launchers, SIGKILL only (see View.handle): each process is ended at once. Returns
    (a note on what happened, whether any was sent)."""
    for pid, started in job.members:
        if start_time(pid) != started or not holds_gpu(pid):
            return [("{} has changed meanwhile: nothing sent".format(job.name), WARN)], False
        if not ours(pid):
            return [("only your own processes can be stopped", WARN)], False
    workers = [pid for pid, _ in job.members if pid != job.pid]  # the launcher may be on a GPU too
    led = bool(job.launcher) and ours(job.pid) and all(parent_of(pid) == job.pid for pid in workers)
    targets = ([job.pid] + (workers if name == "SIGKILL" else [])) if led else [pid for pid, _ in job.members]
    sent = []
    for pid in targets:  # each on its own: one ended meanwhile keeps none of the others from theirs
        try:
            # Windows: any signal but Ctrl+C's ends the process at once (TerminateProcess)
            os.kill(pid, signal.SIGTERM if os.name == "nt" else getattr(signal, name))
            sent.append(pid)
        except PermissionError:
            return [("not allowed to signal PID {}".format(pid), WARN)], bool(sent)
        except OSError:  # ended meanwhile: no such process (on Windows, an invalid PID)
            pass
    if not sent:
        return [("{} has already ended".format(job.name), "")], False
    if led or len(sent) == 1:
        to = "{} (PID {})".format(job.launcher if led else job.name, job.pid if led else sent[0])
    else:
        to = "the {} processes of {}".format(len(sent), job.name)
    return [("{} {}".format("ended" if os.name == "nt" else "sent {} to".format(name), to), "")], True


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


# The pointer over a graph (View.hover): a guide up the empty part of its column, and a chip with that moment.
GUIDE = _rgb((150, 150, 150))
CHIP = "\x1b[48;2;235;235;235m" + _rgb((24, 24, 27))
COLOR_24 = re.compile(r"(\x1b\[[34]8;2;)(\d+);(\d+);(\d+)m")


def lit(style):
    """`style` lit up, for the graph column under the pointer: its colors halfway to white; on PAPER, where
    restyle turns lightness around, halfway to black."""
    paper = ON_PAPER if style.startswith(ON_PAPER) else ""
    to = 0 if paper else 255
    return paper + COLOR_24.sub(lambda m: m.group(1) + ";".join(
        str((int(v) + to) // 2) for v in m.group(2, 3, 4)) + "m", style[len(paper):])


def ago(seconds):
    """How long ago, as short as a run time: now, 14.5s ago, 3m05s ago, 1h02m ago."""
    if seconds < 0.05:
        return "now"
    if seconds < 60:
        return "{:g}s ago".format(round(seconds, 1))
    m, s = divmod(int(round(seconds)), 60)
    return "{}m{:02d}s ago".format(m, s) if m < 60 else "{}h{:02d}m ago".format(m // 60, m % 60)


def row(left, right=()):
    """One stats line: `left`, then the `right` segments flush right; exactly INFO_W cells."""
    room = INFO_W - width_of(right)
    return pad(clip(left, room - 1 if right else room), room) + list(right)


def share(part, whole):
    """A 4-cell percentage, colored by its value."""
    if part is None or not whole:
        return [("   -", "")]
    return [("{:.0f}%".format(100 * part / whole).rjust(4), heat(part / whole))]


def num(value, fmt):
    return "-" if value is None else fmt.format(value)


def info(gpu, height):
    """The stats column; None for the rule between MEM and the PCIe traffic."""
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
        return row([(label, "")], [("-" if rate is None else "{:.2f}".format(rate / 1e9), ""), cap, (" GB/s", "")])

    gpu_row = row(busy)
    mem_row = row([("MEM ", "")] + share(s.mem_used, s.mem_total),
                  [("{} / {} GiB".format(num(s.mem_used, "{:.1f}"), total), "")])
    to_gpu, to_cpu = link("CPU -> GPU", s.rx), link("GPU -> CPU", s.tx)
    # The rule (None) between MEM and the PCIe traffic only appears when there is room.
    if height >= MAX_INNER_H:
        return [gpu_row, mem_row, None, to_gpu, to_cpu]
    return [gpu_row, mem_row, to_gpu, to_cpu][:height]


def degraded(gpu):
    """True when the PCIe link runs on fewer lanes than card and slot allow, e.g. a loose card."""
    s = gpu.now
    return bool(s.pcie_width and gpu.pcie_max_width and s.pcie_width < gpu.pcie_max_width)


def panel(gpu, width, height, selected=frozenset(), left="graph", scroll=(0, 0)):
    """One GPU's box: beside the stats, as `left` says (see LEFTS), the utilization graph, or the GPU's
    processes one a line, moved `scroll` = (lines, cells) on (see process_lines); or the stats alone. Faded
    when jobs are `selected` (their PIDs) and none runs here."""
    s = gpu.now
    faded = bool(selected) and not any(p.job in selected for p in s.processes)
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
    if left in CARDS:  # the numbers only, no processes
        split = width - 1
        body = [[("├" + "─" * (width - 2) + "┤", DIM)] if i is None else [("│ ", DIM)] + i + [(" │", DIM)]
                for i in info(gpu, height)]
        label = []
    else:
        graph_w = max(0, width - CHROME_W - INFO_W)
        split = 2 + graph_w + 1  # column of the graph | stats divider
        rule = [(" ├" + "─" * (INFO_W + 2) + "┤", DIM)]  # across the stats, joined to the borders
        if left == "graph":
            beside = graph(gpu.history, graph_w, height, not faded)
            label = process_label(s.processes, split - 4, selected)
        else:
            beside, label = process_lines(s.processes, graph_w, height, selected, *scroll)
        body = [[("│ ", DIM)] + g + (rule if i is None else [(" │ ", DIM)] + i + [(" │", DIM)])
                for g, i in zip(beside, info(gpu, height))]
    lines = [top] + body + [bottom_edge(width, split, label)]
    if faded:
        lines = [[(seg[0], FADED + (seg[1] if seg[1] != BOLD else "")) + seg[2:] for seg in line] for line in lines]
    if THEME.paper:  # black on white, see restyle
        lines = [[(seg[0], ON_PAPER + seg[1]) + seg[2:] for seg in line] for line in lines]
    return lines


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


def process_entries(processes):
    """Processes grouped by owner and conda environment, groups and processes by memory, biggest first, as
    (tag, process): the tag its account, "kim/torch"."""
    def mem(process):
        return process.mem or 0
    groups = {}
    for process in sorted(processes, key=mem, reverse=True):
        groups.setdefault(account(process), []).append(process)
    ordered = sorted(groups.items(), key=lambda item: sum(map(mem, item[1])), reverse=True)
    return [(tag, process) for tag, group in ordered for process in group]


def process_name(process, selected=frozenset()):
    """The segment of `process`'s name: clicked, it means its job; those of `selected` jobs reversed."""
    return process.name, THEME.selected if process.job in selected else THEME.accent, process.job


def run_time(process):
    return elapsed(process.started) if process.started is not None else "-"


def memory(process):
    return "-" if process.mem is None else "{:.1f}G".format(process.mem)


def process_label(processes, room, selected=frozenset()):
    """The processes (process_entries) on one line: "migi: train.py 2h13m 15.1G · eval.py 5m 0.5G   kim/torch:
    a.py 1d4h 9.0G". As many whole entries as fit in `room` cells, then "+N" for the rest."""
    entries, last = [], None
    for tag, process in process_entries(processes):
        gap = [] if not entries else [(" · ", DIM)] if tag == last else [("   ", "")]
        details = " ".join(part for part in (run_time(process), memory(process)) if part != "-")
        entries.append(gap + (tagged(process) + [(": ", DIM)] if tag and tag != last else [])
                       + [process_name(process, selected)] + ([(" " + details, DIM, process.job)] if details else []))
        last = tag

    def more(n):
        return [("  +{}".format(n), DIM)] if n else []
    label = []
    for i, entry in enumerate(entries):
        rest = len(entries) - i - 1
        if width_of(label + entry + more(rest)) > room:
            # The first entry is always shown, cut if need be; later ones collapse into the count.
            return clip(entry, room - width_of(more(rest))) + more(rest) if not label else label + more(rest + 1)
        label += entry
    return label


def process_lines(processes, room, height, selected=frozenset(), top=0, shift=0):
    """The processes (process_listing) one a line, `height` lines of `room` cells, from
    line `top` on and `shift` cells in (see listing_extent), and the label for the box's bottom edge: which
    lines show of how many, when not all do."""
    lines = process_listing(processes, selected)
    # Sideways the command moves; account, run time and memory stay.
    shown = [pad(clip(fixed + skip(moving, shift), room), room) for fixed, moving in lines[top:top + height]]
    label = [("{}-{} of {}".format(top + 1, top + len(shown), len(lines)), DIM)] if len(lines) > height else []
    return shown + [[(" " * room, "")]] * (height - len(shown)), label


def process_listing(processes, selected=frozenset()):
    """process_lines's lines, whole, as (what stays, what moves sideways): account and environment (see
    tagged), run time and memory in columns; then the command line, the name in the process names' color and
    the arguments dim: "kim/torch:  2d5h  26.1G  train.py --lr 3". The columns are as wide as the GPU's
    longest."""
    entries = process_entries(processes)
    tag_w = max([0] + [len(tag) + 1 for tag, _ in entries])  # the longest "kim/torch:", when any has one
    time_w = max([0] + [len(run_time(process)) for _, process in entries])
    mem_w = max([0] + [len(memory(process)) for _, process in entries])
    lines = []
    for tag, process in entries:
        command = process.command
        rest = command[len(process.name):] if command.startswith(process.name) else " " + command
        who = tagged(process) + [(":", DIM)] if tag else []
        numbers = " " * (tag_w + 2 - width_of(who) if tag_w else 0) + "{:>{}}  {:>{}}  ".format(
            run_time(process), time_w, memory(process), mem_w)
        lines.append(([seg[:2] + (process.job,) for seg in who] + [(numbers, DIM, process.job)],
                      [process_name(process, selected), (rest, DIM, process.job)]))
    return lines


def listing_extent(processes, room, height):
    """How far process_lines can move on for these processes: (lines, cells)."""
    lines = process_listing(processes)
    return max(0, len(lines) - height), max([0] + [width_of(fixed + moving) - room for fixed, moving in lines])


def skip(line, cells):
    """`line` without its first `cells` cells."""
    out = []
    for seg in line:
        if cells >= len(seg[0]):
            cells -= len(seg[0])
            continue
        out.append((seg[0][cells:],) + seg[1:])
        cells = 0
    return out


def bottom_edge(width, split, label):
    """Bottom border with `label` flush right against column `split`, the graph | stats divider."""
    label = clip(label, split - 4)  # keep "╰─" and a space on each side
    middle = [(" ", "")] + label + [(" ", "")] if label else []
    return ([("╰" + "─" * max(0, split - 1 - width_of(middle)), DIM)] + middle
            + [("─" * max(0, width - split - 1) + "╯", DIM)])


MIN_INNER_H = 2  # the shortest a box gets; when even that leaves GPUs out, the boxes scroll


# What v goes through (V back): the graph beside the stats, the processes there, the stats alone (a box the stats and
# their borders wide), and of them GPU and MEM alone ("brief", two lines).
LEFTS = ("graph", "processes", "numbers", "brief")
CARDS = ("numbers", "brief")
CARD_W = INFO_W + 4


def layout(count, width, height, left="graph"):
    """(columns, inner height) for `count` GPU boxes in `width` x `height` cells: one column, two side by
    side only when that lets all show at full height, then shorter. Cards stay one column at their height,
    and scroll: shorter numbers would be the brief ones."""
    if left in CARDS:
        return 1, MIN_INNER_H if left == "brief" else MAX_INNER_H
    most = 2 if count > 1 and width // 2 >= CHROME_W + INFO_W + MIN_GRAPH_W else 1  # narrower is not worth it
    cols = next((c for c in range(1, most + 1) if math.ceil(count / c) * (MAX_INNER_H + 2) <= height), most)
    rows = max(1, math.ceil(count / cols))
    return cols, max(MIN_INNER_H, min(MAX_INNER_H, height // rows - 2))


def render(gpus, width, height, shape, selected=frozenset(), left="graph", scroll=None):
    """The GPU boxes, in `shape` = (columns, inner height) (see layout): exactly `height` lines of exactly
    `width` cells. `left`: see panel; the processes moved on as `scroll` has it for each GPU (its index:
    (lines, cells))."""
    cols, inner = shape
    box_w = CARD_W if left in CARDS else width // cols
    rows = math.ceil(len(gpus) / cols)
    lines = []
    for r in range(rows):
        panels = [panel(g, box_w, inner, selected, left, (scroll or {}).get(g.index, (0, 0)))
                  for g in gpus[r * cols:(r + 1) * cols]]
        lines += [[seg for part in parts for seg in part] for parts in zip(*panels)]
    lines = [pad(clip(line, width), width) for line in lines[:height]]
    return lines + [[(" " * width, "")]] * (height - len(lines))


# The process list's columns: heading, width (">" when flush right), the cell for a job as segments, and the
# key it sorts by. "command" takes whatever width is left.
COLUMNS = [
    ("PID", 8, lambda j: [(str(j.pid), "")], lambda j: j.pid),
    ("account/env", 14, lambda j: tagged(j), account),
    ("process", 22, lambda j: [(j.name + ("  {} workers".format(len(j.members)) if len(j.members) > 1 else ""),
                                THEME.accent)], lambda j: j.name.lower()),
    ("GPUs", 8, lambda j: [(ranges(j.gpus), "")], lambda j: j.gpus),
    (">util", 5, lambda j: share(j.util, 100), lambda j: -1 if j.util is None else j.util),
    (">memory", 7, lambda j: [("{:.1f}G".format(j.mem), "")], lambda j: j.mem),
    (">time", 7, lambda j: [(elapsed(j.started) if j.started is not None else "-", "")],
     lambda j: -(j.started or 0)),
    ("command", 0, lambda j: [(j.command, DIM)], lambda j: j.command),
]
SORTS = {heading.lstrip(">"): key for heading, _, _, key in COLUMNS}
COMMAND_AT = 1 + sum(size + 2 for _, size, _, _ in COLUMNS[:-1])  # the column where commands start


def cells(values, width, fill=("",)):
    """One list line of `width` cells from the columns' segments, two spaces apart; `fill` = (style, ...) of
    the spaces, so that a selected row stays one bar and clicks anywhere: those padding a column mean what
    its last segment does, but take nothing else of it, an underline, say."""
    line, used = [(" ",) + fill], 1
    for (heading, size, _, _), segments in zip(COLUMNS, values):
        size = size or max(0, width - used - 1)
        segments = clip(segments, size)
        padding = [(" " * (size - width_of(segments)),) + fill[:1] + (segments[-1][2:3] if segments else fill[1:])]
        line += padding + segments if heading.startswith(">") else segments + padding
        line.append((" " if heading == "command" else "  ",) + fill)
        used += size + 2
    return line


def process_list(jobs, width, rows, selected, focus, top, sort, shift=0):
    """The list p opens, `rows` lines: the headings (clicking one sorts by it; `sort` = (column, descending)),
    the jobs from index `top` on (clicking one means it; the `selected` ones reversed, the command of the one
    in `focus` moved `shift` characters on), how many more. A command too long to show ends in "…"."""
    room = max(0, width - COMMAND_AT - 1)
    column, descending = sort
    head = []
    for heading, _, _, _ in COLUMNS:
        name = heading.lstrip(">")
        head.append([(name + ("▼" if descending else "▲") if name == column else name,
                      "" if name == column else DIM, ("sort", name))])
    lines = [cells(head, width)]
    shown = jobs[top:top + rows - 2]
    for job in shown:
        chosen = job.pid in selected
        values = [cell(job) for _, _, cell, _ in COLUMNS]
        command = "…" + job.command[shift:] if job.pid == focus and shift else job.command
        values[-1] = [(command if len(command) <= room else command[:max(0, room - 1)] + "…", DIM)]
        lines.append(cells([[(seg[0], THEME.selected if chosen else seg[1], job.pid) for seg in column]
                            for column in values], width, (THEME.selected if chosen else "", job.pid)))
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
CONTROL_KEYS = {"\r": "enter", "\n": "enter", "\t": "tab", "\x1b": "esc",
                "\x03": "q"}  # Ctrl+C, a key on Windows (raw_keys); elsewhere the terminal makes it a signal
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
    every = keys(("v", "view"), ("c", "colors"), ("p", "processes"), ("Esc / q", "quit"))
    few = keys(("p", "processes"), ("Esc / q", "quit"))
    # As much as fits: the link goes first, then how to update, the keys for view and theme, the picker.
    options = [(picker + notice + how + link, every), (picker + notice + how, every), (picker + notice + how, few),
               (picker + notice, few), (notice[1:] if picker else notice, few)]
    left, right = next((option for option in options if width_of(option[0]) + 2 + width_of(option[1]) <= width),
                       options[-1])
    return spread(width, left, right)


MOVES = {"tab": 1, "down": 1, "up": -1, "backtab": -1, "pgdn": 10, "pgup": -10, "home": -10**6, "end": 10**6}


class View:
    """What the screen shows around the numbers, and what keys and clicks do to it: the picked jobs (one, or
    a GPU's), only their GPUs, the process list, a question before stopping, a note on how that went."""

    def __init__(self):
        self.shows = "truecolor"          # the colors the terminal shows, see color_mode
        self.theme = THEMES[0]            # the look, see THEMES (c)
        self.left = "graph"               # what the boxes have beside the stats, see LEFTS (v)
        self.hidden, self.indices = set(), []  # the GPUs left out, and all there are
        self.picked = set()               # the picked jobs' PIDs
        self.job = None                   # of them the one in focus: the arrows go on from it, ← → move its command
        self.only = False                 # show only the GPUs they use
        self.listing = False              # the process list is open
        self.opened = False               # by a pick, and so to close when the picks are let go
        self.top, self.follow = 0, True   # the list's first job; keep the one in focus in view
        self.sort = ("GPUs", False)       # the list's order, and the order Tab and the arrows go in
        self.asking = None                # "SIGTERM" or "SIGKILL" awaiting y or n
        self.note = None                  # (segments, until): what the last stop did
        self.sent = None                  # (PIDs, when) of a SIGTERM, to point at k if a job lives on
        self.lines, self.jobs = [], []    # the last screen drawn, and the jobs on it
        self.gpu_top, self.gpu_page = 0, 1  # the first row of boxes shown, and how many rows fit
        self.list_rows = range(0)         # the screen rows the list takes, when it is open
        self.shift, self.shifted = 0, None  # how far the selected job's command is scrolled, and whose
        self.clicked = (0, None)          # (when, what): the last click, to tell a double click
        self.scroll = {}                  # GPU index: (lines, cells) its processes are moved on
        self.areas = []                   # (rows, columns, GPU) where those processes are on screen
        self.graphs = []                  # (rows, columns, GPU) where the graphs are on screen
        self.panels = []                  # (rows, columns, GPU index) where the boxes are on screen
        self.pointer = None               # (row, column) of the mouse pointer, as the terminal last said
        self.interval = DEFAULT_INTERVAL  # seconds from one graph column to the next

    def restore(self, saved, indices):
        """Take up what an earlier run kept (Settings.load); of the GPUs `indices`, one always stays shown."""
        self.left = saved.get("left", self.left)
        self.theme = next((kept for kept in THEMES if kept.name == saved.get("theme")), self.theme)
        self.sort = saved.get("sort", self.sort)
        hidden = saved.get("hidden", set()) & set(indices)
        self.hidden = hidden if len(hidden) < len(indices) else set()

    def kept(self):
        """What is kept for the next run: see Settings."""
        return {"left": self.left, "theme": self.theme.name, "sort": self.sort, "hidden": set(self.hidden)}

    def pick(self, pids):
        """Pick the jobs `pids`, in the list's order, the first in focus; none lets all go, and closes the
        list if a pick opened it."""
        self.picked = set(pids)
        self.job = next((job.pid for job in self.jobs if job.pid in self.picked), None)
        if not self.picked:
            self.only = False
            self.listing = self.listing and not self.opened
            self.opened = False

    def selected(self):
        """The picked jobs, in the list's order."""
        return [job for job in self.jobs if job.pid in self.picked]

    def screen(self, gpus, width, height, top_line, newer):
        """The whole screen: `top_line`, the GPU boxes (and the list, when open), the bottom line."""
        global THEME, SHOWS
        THEME, SHOWS = self.theme, self.shows
        column, descending = self.sort
        self.jobs = sorted(jobs(gpus), key=lambda job: (SORTS[column](job), job.pid), reverse=descending)
        self.picked &= {job.pid for job in self.jobs}  # those that ended go
        picked = self.selected()
        if self.job not in self.picked:  # the one in focus ended: the next
            self.job = picked[0].pid if picked else None
        self.only = self.only and bool(picked)
        if not any(job.mine for job in picked):  # ours ended while asked about: nothing left to stop
            self.asking = None
        focus = next(job for job in picked if job.pid == self.job) if picked else None
        self.remind()
        self.indices = [g.index for g in gpus]
        used = {index for job in picked for index in job.gpus}
        shown = [g for g in gpus if (g.index in used if self.only else g.index not in self.hidden)]
        rows = min(len(self.jobs) + 2, max(4, (height - 2) // 2), max(0, height - 2)) if self.listing else 0
        body = self.boxes(shown, width, height - 2 - rows)
        self.list_rows = range(1 + len(body), 1 + len(body) + rows)
        if rows:
            if self.shifted != self.job:
                self.shift, self.shifted = 0, self.job
            room = max(0, width - COMMAND_AT - 1)
            self.shift = max(0, min(self.shift, len(focus.command) + 1 - room)) if focus else 0
            body += process_list(self.jobs, width, rows, self.picked, self.job, self.scrolled(rows - 2), self.sort,
                                 self.shift)
        self.lines = self.hover([top_line] + body + [self.bottom_line(width, picked, newer, gpus)])
        return self.lines

    def boxes(self, gpus, width, height):
        """The GPU boxes in `height` lines; when not all fit, the rows from gpu_top on and a line on that."""
        self.areas, self.graphs, self.panels = [], [], []
        if height <= 0:  # a window so short that the list takes it all
            return []
        cols, inner = layout(len(gpus), width, height, self.left)
        rows = math.ceil(len(gpus) / cols)
        note, count = [], len(gpus)
        if rows * (inner + 2) > height:
            self.gpu_page = max(1, (height - 1) // (inner + 2))
            self.gpu_top = max(0, min(self.gpu_top, rows - self.gpu_page))
            gpus = gpus[self.gpu_top * cols:(self.gpu_top + self.gpu_page) * cols]
            shown = "GPUs {} of {}".format(ranges([g.index for g in gpus]), count)
            text = next((t for t in (shown + " shown · wheel or PgUp/PgDn for the rest", shown + " · PgUp/PgDn")
                         if len(t) <= width), shown)
            note = [spread(width, [], [(text, DIM)])]
            height -= 1
        else:
            self.gpu_top, self.gpu_page = 0, rows
        box_w = CARD_W if self.left in CARDS else width // cols
        room = max(0, box_w - CHROME_W - INFO_W)

        def drawn(start, stop):  # screen rows start..stop, but no further than the boxes go: a window too
            return range(start, min(stop, 1 + height))  # short for a whole one shows part of it
        self.panels = [(drawn(1 + k // cols * (inner + 2), 1 + (k // cols + 1) * (inner + 2)),
                        range(k % cols * box_w, (k % cols + 1) * box_w), g.index) for k, g in enumerate(gpus)]
        for k, g in enumerate(gpus if self.left in ("graph", "processes") else []):
            y, x = 2 + k // cols * (inner + 2), k % cols * box_w + 2  # 2: the top line and the box's edge
            if self.left == "graph":
                self.graphs.append((drawn(y, y + inner), range(x, x + room), g))
                continue
            most = listing_extent(g.now.processes, room, inner)  # the processes' offsets in range
            top, shift = self.scroll.get(g.index, (0, 0))
            self.scroll[g.index] = min(max(top, 0), most[0]), min(max(shift, 0), most[1])
            self.areas.append((drawn(y, y + inner), range(x, x + room), g.index))
        return render(gpus, width, height, (cols, inner), self.picked, self.left, self.scroll) + note

    def hovered(self):
        """(GPU, graph rows, graph columns, the column pointed at, how many samples back from the newest that
        is) when the pointer is over a graph column that has a sample; else None."""
        if self.pointer is None:
            return None
        row, col = self.pointer
        for rows, cols, gpu in self.graphs:
            if row in rows and col in cols:
                back = cols.stop - 1 - col  # the newest is on the right
                if back < len(gpu.history):
                    return gpu, rows, cols, col, back
        return None

    def pointed(self):
        """(GPU index, column) of the graph column under the pointer, or None: what the chip is about."""
        hovered = self.hovered()
        return hovered and (hovered[0].index, hovered[3])

    def hover(self, lines):
        """`lines` with the column under the pointer lit up, and beside the pointer a chip with that moment:
        the utilization, the memory in use, and how long ago."""
        hovered = self.hovered()
        if hovered is None:
            return lines
        gpu, rows, cols, col, back = hovered
        mem = gpu.memory[-1 - back]
        parts = ["{}%".format(gpu.history[-1 - back]), None if mem is None else "{:.1f}G".format(mem),
                 ago(back * self.interval)]
        # As much as fits on the pointer's right, or else its left: all, without the memory, the utilization alone.
        chips = [" {} ".format(" · ".join(filter(None, shown))) for shown in (parts, parts[::2], parts[:1])]
        place = next(((chip, start) for chip in chips for start in (col + 2, col - 1 - len(chip))
                      if cols.start <= start and start + len(chip) <= cols.stop), None)
        lines = list(lines)
        for row in rows:
            cells = [(ch,) + tuple(seg[1:]) for seg in lines[row] for ch in seg[0]]
            ch, style = cells[col][:2]
            paper = ON_PAPER if style.startswith(ON_PAPER) else ""
            cells[col] = ("│", paper + GUIDE) if ch == " " else (ch, lit(style))
            if row == self.pointer[0] and place:
                chip, start = place
                bold = chip.index("%") + 1
                for k, c in enumerate(chip):
                    cells[start + k] = (c, paper + CHIP + (BOLD if k < bold else ""))
            lines[row] = [("".join(c[0] for c in run),) + key for key, run in
                          itertools.groupby(cells, key=lambda c: c[1:])]
        return lines

    def detail(self, gpus):
        """The GPUs whose processes' utilization is shown: all with the list open, else the picked jobs'.
        Only those are asked, as the driver takes some 2 ms a GPU to answer."""
        if self.listing:
            return {g.index for g in gpus}
        return {index for job in self.selected() for index in job.gpus}

    def remind(self):
        """A job still there 5 s after its SIGTERM may be stuck, or slow to save: say that k ends it."""
        if not self.sent:
            return
        pids, when = self.sent
        living = [job for job in self.jobs if job.pid in pids]
        if living and time.monotonic() - when > 5:
            what = living[0].name + " is" if len(living) == 1 else "{} jobs are".format(len(living))
            self.note = ([("{} still running 5 s after SIGTERM; k kills at once".format(what), WARN)],
                         time.monotonic() + 6)
        if not living or time.monotonic() - when > 5:
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

    def bottom_line(self, width, picked, newer, gpus):
        if self.asking:
            return spread(width, [(self.question(picked), WARN)], keys(("y", "yes"), ("n", "no")))
        note = self.note[0] if self.note and time.monotonic() < self.note[1] else None
        if picked:
            hints = [("f", "all GPUs" if self.only else "only these GPUs")]
            if any(job.mine for job in picked):
                hints += [("k", "kill")] if os.name == "nt" else [("t", "stop"), ("k", "kill")]
            if len(picked) == 1:
                left = [(picked[0].name, THEME.accent, picked[0].pid), ("  ", DIM)] + about(picked[0])
            else:
                left = [("{} jobs".format(len(picked)), THEME.accent), ("  ", DIM)] + about_all(picked)
            return spread(width, note or left, keys(*hints, ("Esc", "back")))
        if self.listing:
            return spread(width, note or [("click a job or a heading", DIM)],
                          keys(("s", "sort by next"), ("r", "reverse"), ("Esc", "close")))
        if note:
            return spread(width, note, keys(("p", "processes"), ("Esc / q", "quit")))
        return footer(width, newer, self.picker(gpus))

    def question(self, picked):
        """"stop train.py on GPUs 4-5: SIGTERM to its torchrun (PID 48213), which ends its 2 workers?" and the
        like; for several jobs "stop your 2 of 3 jobs on GPUs 4-5,7, SIGTERM to each: train.py, eval.py?". The
        GPUs are those the signal reaches, whichever was clicked. On Windows, "kill python.exe on GPU 0 (PID 4120)
        at once?", as no signal is sent there."""
        what = "stop" if self.asking == "SIGTERM" else "kill"
        ours = [job for job in picked if job.mine]
        where = "on " + gpu_numbers(index for job in ours for index in job.gpus)
        if len(picked) > 1:
            which = "{} jobs".format(len(ours)) if len(ours) == len(picked) else "your {} of {} jobs".format(
                len(ours), len(picked))
            how = " at once" if os.name == "nt" else ", {} to each".format(self.asking)
            return "{} {} {}{}: {}?".format(what, which, where, how, ", ".join(job.name for job in ours))
        job = ours[0]
        if not job.launcher:
            how = "at once" if os.name == "nt" else "with " + self.asking
            return "{} {} {} (PID {}) {}?".format(what, job.name, where, job.pid, how)
        passed = ", which ends its" if self.asking == "SIGTERM" else " and to its"
        return "{} {} {}: {} to its {} (PID {}){} {} workers?".format(what, job.name, where, self.asking,
                                                                     job.launcher, job.pid, passed, len(job.members))

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
        if kind == "move":
            self.pointer = value
            return True
        if kind in ("wheel", "hwheel"):  # over a GPU's processes, they scroll, either way;
            step, row, col = value     # over the list, the list, or the picked job's command sideways;
            area = next((index for rows, cols, index in self.areas  # anywhere else, the GPU boxes
                         if row in rows and col in cols), None)
            if area is not None:
                top, shift = self.scroll.get(area, (0, 0))
                self.scroll[area] = (top + step, shift) if kind == "wheel" else (top, shift + 8 * step)
            elif kind == "hwheel":
                if row in self.list_rows:
                    self.shift = max(0, self.shift + 8 * step)
            elif row in self.list_rows:
                if col >= COMMAND_AT and self.job is not None and self.target(row, col) == self.job:
                    self.shift = max(0, self.shift + 8 * step)
                else:
                    self.top, self.follow = self.top + 3 * step, False
            else:
                self.gpu_top += step
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
            row, col = value
            target = self.target(row, col)
            if isinstance(target, tuple):  # a key hint, a heading of the list, a GPU's number at the bottom
                return self.handle(target)
            if target is None:  # anywhere else in a GPU's box: that GPU
                target = next((("box", index) for rows, cols, index in self.panels if row in rows and col in cols),
                              None)
            now = time.monotonic()
            double = target is not None and self.clicked[1] == target and now - self.clicked[0] < 0.4
            self.asking, self.follow, self.clicked = None, True, (now, target)
            # A job picks it, a GPU the jobs on it, and outside the list opens that; the same again lets them
            # go, as does a place meaning nothing, which outside the list closes that too. A double click picks
            # them and shows only their GPUs, or all again.
            if target is None:
                self.pick(())
                self.listing = self.listing and row in self.list_rows
                return True
            pids = [job.pid for job in self.jobs if target[1] in job.gpus] if isinstance(target, tuple) else [target]
            if not pids:
                self.note = ([("no processes on GPU {}".format(target[1]), DIM)], now + 2)
            elif double:
                self.pick(pids)
                self.only = not self.only
            elif set(pids) == self.picked:
                self.pick(())
            else:
                self.pick(pids)
                self.opened = self.opened or not self.listing
                self.listing = True
            return True
        key = value[0].lower() if len(value[0]) == 1 else value[0]
        if self.asking:  # y stops; anything else, n and Esc among them, lets it be
            if key == "y":
                self.stop_picked()
            self.asking = None
            return key != "q"
        if key == "q":
            return False
        if key == "esc":  # one step back; from the plain screen, quit
            if self.only:
                self.only = False
            elif self.listing:
                self.listing = False
            elif self.picked:
                self.pick(())
            else:
                return False
        elif key == "p":  # opened or closed by hand, the list stays so when the picks go
            self.listing, self.opened = not self.listing, False
        elif key == "v":  # the graph, the processes, the stats alone, GPU and MEM alone; V the one before
            self.left = LEFTS[(LEFTS.index(self.left) + (-1 if value[0] == "V" else 1)) % len(LEFTS)]
            name = {"graph": "graphs", "processes": "processes", "numbers": "numbers only",
                    "brief": "GPU and MEM only"}[self.left]
            self.note = ([("view: " + name, "")], time.monotonic() + 2)
        elif key == "c":  # the next theme; C the one before
            self.theme = THEMES[(THEMES.index(self.theme) + (-1 if value[0] == "C" else 1)) % len(THEMES)]
            shows = "" if self.shows == "truecolor" else ", in 256 colors: all this terminal shows"
            self.note = ([("theme: " + self.theme.name + shows, "")], time.monotonic() + 3)
        elif key.isdecimal() and int(key) in self.indices:  # not isdigit: ² is a digit, but no number
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
        elif key in ("left", "right") and self.left == "processes":  # every GPU's processes, sideways
            step = 8 if key == "right" else -8
            self.scroll = {index: (top, shift + step) for index, (top, shift) in self.scroll.items()}
        elif key in MOVES and self.jobs:
            order = [job.pid for job in self.jobs]
            i = order.index(self.job) if self.job in order else (-1 if MOVES[key] > 0 else len(order))
            self.pick([order[min(max(i + MOVES[key], 0), len(order) - 1)]])
            self.follow = True
        elif self.picked and key in ("f", "enter"):
            self.only = not self.only
        elif self.picked and key in ("t", "k"):
            if os.name == "nt" and key == "t":  # no SIGTERM there, nothing that asks a process to end
                self.note = ([("Windows can only end a process at once: k", WARN)], time.monotonic() + 3)
            elif any(job.mine for job in self.selected()):
                self.asking = "SIGTERM" if key == "t" else "SIGKILL"
            else:
                self.note = ([("only your own processes can be stopped", WARN)], time.monotonic() + 3)
        return True

    def stop_picked(self):
        """Send the signal asked for to each picked job of ours (see stop), and say how that went."""
        ours = [job for job in self.selected() if job.mine]
        results = [stop(job, self.asking) for job in ours]
        sent = {job.pid for job, (_, ok) in zip(ours, results) if ok}
        failed = [note for note, ok in results if not ok]
        if len(ours) == 1:
            note = results[0][0]
        elif not failed:
            note = [("sent {} to {} jobs".format(self.asking, len(sent)), "")]
        else:
            note = [("sent {} to {} of {} jobs; ".format(self.asking, len(sent), len(ours)), WARN)] + failed[0]
        self.note = (note, time.monotonic() + 4)
        self.sent = (sent, time.monotonic()) if sent and self.asking == "SIGTERM" else None


# For terminals that draw the line, block and other characters above two cells wide, as they may for Korean,
# Chinese and Japanese (Unicode calls their width "ambiguous"): ASCII look-alikes, one for one.
ASCII = str.maketrans({"─": "-", "│": "|", "├": "+", "┤": "+", "╭": "+", "╮": "+", "╰": "+", "╯": "+",
                       "▁": "_", "▂": "_", "▃": "_", "▄": "=", "▅": "=", "▆": "#", "▇": "#", "█": "#",
                       "▀": " ",  # the graph's full cells: their background alone, a solid block
                       "°": " ", "·": "-", "…": "~", "↑": "^", "↓": "v", "▲": "^", "▼": "v"})
# Control characters, which would act on the terminal rather than show: anyone's command line can carry them
# (ESC, BEL, a new line), and with them clear the screens of all who watch, or set their window titles. A "?"
# each, a cell, as they are counted.
SHOWN = {code: "?" for code in itertools.chain(range(0x20), range(0x7f, 0xa0))}
ASCII.update(SHOWN)


def paint(line, mode="truecolor", ascii=False, gray=False):
    """One line as the terminal gets it, from plain to plain: in color `mode` (with `gray`, grays only; see
    restyle), with `ascii` in ASCII, control characters as "?" (SHOWN), a style code only where the style
    changes."""
    out, current = [], ""
    for text, style, *_ in line:
        text = text.translate(ASCII if ascii else SHOWN)
        if mode != "truecolor" or gray:
            style = restyle(style, mode, gray)
        if style != current:
            # One color replaces another directly; anything else (bold, a background, plain) needs a reset first.
            colors = style.startswith("\x1b[38") and current.startswith("\x1b[38") and "\x1b[48" not in current
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
    """What stays from one run to the next, for each client (see client): what is beside the stats, the theme, the
    list's order, and the GPUs hidden, those for each machine, as several often share a home directory.
    Kept in ~/.config/nvmon/settings.json (%APPDATA%\\nvmon on Windows); a file that cannot be read or written
    only means that nothing is kept."""

    def __init__(self):
        base = (os.environ.get("APPDATA") if os.name == "nt" else None) or os.environ.get("XDG_CONFIG_HOME")
        base = base or os.path.join(os.path.expanduser("~"), ".config")
        self.path, self.client = os.path.join(base, "nvmon", "settings.json"), client()

    def _read(self):
        """The file's settings; {} when there is no file, None when it is not JSON nvmon can keep to (edited
        by hand, say), and so not to be written over."""
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def load(self, host):
        """This client's settings as View.restore takes them, the GPUs hidden those on `host`; none when they
        are not as nvmon writes them (edited by hand, say)."""
        try:
            mine = self._read()[self.client]
            kept = {"left": mine["left"], "theme": mine["theme"], "sort": (mine["sort"][0], bool(mine["sort"][1])),
                    "hidden": set(map(int, mine["hidden"].get(host, [])))}
            return kept if kept["left"] in LEFTS and kept["sort"][0] in SORTS else {}
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError):  # Overflow: Infinity
            return {}

    def save(self, kept, indices, host):
        """Keep `kept` (View.kept) for this client; for GPUs that are not `indices` (nvmon -g), and on other
        hosts, what was kept for them. Everyone else's settings stay as the file has them now; a file nvmon
        cannot read stays as it is."""
        data = self._read()
        if data is None:
            return
        try:
            hidden = {name: set(map(int, gpus)) for name, gpus in data[self.client]["hidden"].items()}
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            hidden = {}
        hidden[host] = hidden.get(host, set()) - set(indices) | kept["hidden"]
        data[self.client] = {"left": kept["left"], "theme": kept["theme"], "sort": list(kept["sort"]),
                             "hidden": {name: sorted(gpus) for name, gpus in hidden.items() if gpus}}
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
# What INPUT would take for keys if the rest of it were there: input that ends in the start of an escape
# sequence, or of a letter's UTF-8 (see Screen.read).
CUT = re.compile(rb"\x1b(?:\[[<?>0-9;]*|\[M.{0,2}|O|P[^\x1b]*\x1b?)?\Z"
                 rb"|(?:[\xc2-\xdf]|[\xe0-\xef][\x80-\xbf]?|[\xf0-\xf4][\x80-\xbf]{0,2})\Z", re.DOTALL)


def events(data):
    """The input in `data`, the bytes a terminal sends: ("key", name) events, ("click", row, column) for
    a left-button press, ("wheel", +1 down or -1 up, row, column), ("hwheel", +1 right or -1 left, row,
    column), ("move", row, column) where the pointer went."""
    out = []
    for m in INPUT.finditer(data):
        if m.group("button") or m.group("old"):
            if m.group("button"):
                button, x, y = (int(m.group(g)) for g in ("button", "x", "y"))
                press = m.group("act") == b"M"
            else:
                button, x, y = (c - 32 for c in m.group("old"))
                press = button & 3 != 3
            if button & 64:  # the wheel: 64 up, 65 down, 66 left, 67 right; with Shift (+4) sideways too
                sideways = button & 2 or button & 4
                out.append(("hwheel" if sideways else "wheel", 1 if button & 1 else -1, y - 1, x - 1))
            elif button & 32:  # a move, a button held or not
                out.append(("move", y - 1, x - 1))
            elif press and button & ~(4 | 8 | 16) == 0:  # left button down, Shift, Alt or Ctrl held at most
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


# What a terminal that shows 24-bit color answers to Screen.probe; one is enough. Anything else, no answer
# included, means 256 colors, as for NetSarang Xshell (24-bit color is off there unless turned on), PuTTY
# before 0.71, Tera Term, macOS Terminal before macOS 26.
TRUECOLOR_ANSWERS = re.compile(
    # DECRQSS: the color set kept as it was, in any of the forms seen: 38;2;1;2;3, 38:2::1:2:3 (Windows
    # Terminal, VTE, Ghostty), 38:2:1:2:3 (kitty), 38:2:1:1:2:3 (iTerm2)
    rb"\x1bP1\$r[0-9;:]*38[;:]2[;:](?:\d*[;:])?1[;:]2[;:]3m"
    rb"|\x1bP1\+r(?:524742|5463)"  # XTGETTCAP: RGB or Tc in its terminfo
    rb"|\x1bP>\|(?:xterm\.js|kitty|WezTerm|iTerm2|VTE|Konsole|ghostty|foot|contour|Rio|mintty|tmux)"  # XTVERSION
    # DA2: xterm.js (Tabby, VS Code; it answers DECRQSS with 0m whatever is set), Windows Terminal and
    # Alacritty, Konsole, VTE, kitty, WezTerm, iTerm2, tmux
    rb"|\x1b\[>(?:0;276;0|0;\d+;1|[01];115;0|6[15];\d+;1|1;4\d{3};\d+|1;277;0|64;2500;0|84;0;0)c")


class Screen:
    """Alternate screen, hidden cursor, no auto-wrap, unbuffered keys, mouse clicks, wheel and moves; all
    restored on exit."""

    def __enter__(self):
        self._keys = self._mouse = sys.stdin.isatty()  # keys stop if stdin closes; the mouse goes off on exit
        self._restore = [enable_vt(), raw_keys() if self._keys else (lambda: None)]
        self._painted, self._size = [], None  # the lines last sent, and the window they were sent to
        # 1000 + 1003 + 1006: clicks, the wheel and every move of the pointer come in as SGR-encoded sequences
        # (Shift+drag still selects text). A terminal without 1003 sends clicks and the wheel all the same.
        self._write("\x1b[?1049h\x1b[?25l\x1b[?7l" + ("\x1b[?1000h\x1b[?1003h\x1b[?1006h" if self._mouse else ""))
        # Windows Terminal and the console of Windows 10 on show 24-bit color and draw lines one cell wide;
        # not asked there, as the console would answer nothing on color, and so get 256 colors.
        try:
            self.wide, self.truecolor = self.probe() if self._keys and os.name != "nt" else (False, os.name == "nt")
        except BaseException:  # Ctrl+C while waiting for the answers: the terminal as it was all the same
            self.__exit__()
            raise
        return self

    def probe(self):
        """(wide, truecolor), as the terminal answers: whether it draws a line character two cells wide, and
        whether it shows 24-bit color (TRUECOLOR_ANSWERS). Every terminal answers the last question, DA1, so
        that answer ends the wait, a round trip even over ssh; the answer on the color was seen to come after
        it (Windows Terminal over Windows's ssh), so for that a quarter second more. No answer within a second
        means neither."""
        self._write("\x1b[H─\x1b[6n"                  # where the cursor is after one line character
                    "\x1b[>c\x1b[>0q"                 # DA2 and XTVERSION: which terminal this is
                    "\x1bP+q524742;5463\x1b\\"        # XTGETTCAP: whether its terminfo has RGB or Tc
                    "\x1b[38;2;1;2;3m\x1bP$qm\x1b\\"  # DECRQSS: the color set now, as the terminal kept it
                    + RESET + "\x1b[c")                 # DA1, answered last
        data, end = b"", time.monotonic() + 1
        answered = False  # DA1 is in
        while True:
            if not answered and re.search(rb"\x1b\[\?[0-9;]*c", data):
                answered, end = True, min(end, time.monotonic() + 0.25)  # a quarter second more for DECRQSS
            if answered and re.search(rb"\x1bP[01]\$r[^\x1b]*\x1b\\", data):
                break
            left = end - time.monotonic()
            more = incoming(left) if left > 0 else None
            if not more:
                break
            data += more
        cursor = re.search(rb"\x1b\[\d+;(\d+)R", data)
        return bool(cursor) and int(cursor.group(1)) > 2, bool(TRUECOLOR_ANSWERS.search(data))

    def __exit__(self, *exc):
        mouse = "\x1b[?1006l\x1b[?1003l\x1b[?1000l" if self._mouse else ""
        self._write(RESET + mouse + "\x1b[?7h\x1b[?25h\x1b[?1049l")
        for undo in self._restore:
            undo()

    def read(self, seconds):
        """Key and mouse events as soon as there are some; [] once `seconds` pass without any."""
        if not self._keys:
            time.sleep(max(0, seconds))
            return []
        data = incoming(seconds)
        # A read can end inside a sequence, as it takes 1024 bytes at most (a slow ssh link delivers in
        # bursts). The rest is on its way: wait for it, up to 0.05 s (an Esc key waits as long), or the halves
        # come in as keys, a pointer's move ESC [<35;120;40M as Esc, [, <, 3, 5... which hide GPUs.
        while data and CUT.search(data):
            more = incoming(0.05)
            if not more:
                break
            data += more
        if data is None:  # stdin closed: nothing more will come
            self._keys = False
        return events(data or b"")

    def draw(self, lines, mode, gray, size):
        """The screen `lines`, in a window of `size`: only the lines that changed since the last time, so that
        a frame, or the pointer moving over a graph, sends a few hundred bytes rather than the whole screen."""
        painted = [paint(line, mode, self.wide, gray) for line in lines]
        last = self._painted if self._size == size else []  # a new size: all anew
        changed = "".join("\x1b[{};1H{}".format(row + 1, text) for row, text in enumerate(painted)
                          if row >= len(last) or last[row] != text)
        self._painted, self._size = painted, size
        # 2026 = synchronized output: terminals that know it swap the frame in at once.
        self._write("\x1b[?2026h" + changed + "\x1b[?2026l")

    @staticmethod
    def _write(text):
        # Raw UTF-8 bytes: works even where Python picked an ASCII stdout (LANG=C on 3.6).
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()


def incoming(seconds):
    """What the terminal sends within `seconds`, as soon as there is some: bytes, b"" when nothing came,
    None once stdin is closed."""
    if os.name == "nt":
        return console_input(seconds)
    import select
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], max(0, seconds))[0]:
        return b""
    return os.read(fd, 1024) or None


class KeyEventRecord(ctypes.Structure):  # Windows's INPUT_RECORD, read as the KEY_EVENT_RECORD it holds for a key
    _fields_ = [("EventType", ctypes.c_ushort), ("bKeyDown", c_int), ("wRepeatCount", ctypes.c_ushort),
                ("wVirtualKeyCode", ctypes.c_ushort), ("wVirtualScanCode", ctypes.c_ushort),
                ("UnicodeChar", ctypes.c_wchar), ("dwControlKeyState", ctypes.c_ulong)]


# Keys as the console may hand them over, by virtual-key code, rather than as the sequences raw_keys asks for:
# Windows's own ConPTY did so for the first ones it got from a terminal (as under Windows's ssh). What any
# other terminal sends for them.
CONSOLE_KEYS = {0x26: "\x1b[A", 0x28: "\x1b[B", 0x27: "\x1b[C", 0x25: "\x1b[D", 0x24: "\x1b[H", 0x23: "\x1b[F",
                0x21: "\x1b[5~", 0x22: "\x1b[6~"}


def key_text(record):
    """What a key-down record (KeyEventRecord) types: its character, or for a key handed over as itself, the
    sequence (CONSOLE_KEYS); Shift+Tab so handed over is ESC [ Z, as elsewhere."""
    if record.UnicodeChar == "\0":
        return CONSOLE_KEYS.get(record.wVirtualKeyCode, "")
    shift_tab = record.wVirtualKeyCode == 0x09 and record.dwControlKeyState & 0x0010  # VK_TAB, SHIFT_PRESSED
    return "\x1b[Z" if shift_tab else record.UnicodeChar


def console_input(seconds):
    """incoming, on Windows: the characters typed into the console, the arrows, the mouse and the like among
    them as the same sequences as from any other terminal (raw_keys sees to that). The console's other
    input, keys let go among it, is skipped, and the wait goes on."""
    import msvcrt
    kernel32, handle = ctypes.windll.kernel32, c_void_p(msvcrt.get_osfhandle(sys.stdin.fileno()))
    records, count, end = (KeyEventRecord * 1024)(), ctypes.c_ulong(), time.monotonic() + seconds
    while True:
        signaled = kernel32.WaitForSingleObject(handle, max(0, int((end - time.monotonic()) * 1000)))
        if signaled == 0x102:  # WAIT_TIMEOUT
            return b""
        if signaled != 0 or not kernel32.ReadConsoleInputW(handle, records, len(records), byref(count)):
            return None  # no console to read from after all: NUL, for one, passes for a terminal
        text = "".join(key_text(r) for r in records[:count.value] if r.EventType == 1 and r.bKeyDown)  # KEY_EVENT
        if text:
            return text.encode("utf-8", "ignore")  # "ignore": half of an emoji, the other half read next


def console_mode(stream, on, off=0):
    """Windows: the console mode of `stream` with the flags `on` set and `off` cleared; returns a function
    that undoes it."""
    import msvcrt
    kernel32, handle, mode = ctypes.windll.kernel32, c_void_p(msvcrt.get_osfhandle(stream.fileno())), ctypes.c_ulong()
    if not kernel32.GetConsoleMode(handle, byref(mode)):
        return lambda: None
    kernel32.SetConsoleMode(handle, mode.value & ~off | on)
    return lambda: kernel32.SetConsoleMode(handle, mode.value)


def enable_vt():
    """Make the Windows console interpret ANSI codes; returns a function that undoes it."""
    if os.name != "nt":
        return lambda: None
    # ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN (VT-style end of line)
    return console_mode(sys.stdout, 0x0004 | 0x0008)


def raw_keys():
    """Deliver key presses at once and without echo; returns a function that undoes it."""
    if os.name == "nt":
        # On: ENABLE_VIRTUAL_TERMINAL_INPUT, keys and the mouse as escape sequences, as elsewhere. Off:
        # ENABLE_PROCESSED_INPUT, so that Ctrl+C comes as a key (CONTROL_KEYS): as a signal, it would only
        # stop nvmon once console_input's wait is over; ENABLE_LINE_INPUT and ENABLE_ECHO_INPUT.
        return console_mode(sys.stdin, 0x0200, 0x0001 | 0x0002 | 0x0004)
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
    anything faster would only keep the driver busier and counts as MIN_INTERVAL."""
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not seconds >= 0 or math.isinf(seconds):  # abc, nan, -1, inf
        raise argparse.ArgumentTypeError("expected seconds such as 0.5 or 2, got {!r}".format(text))
    return max(MIN_INTERVAL, seconds)


def main():
    parser = argparse.ArgumentParser(
        prog="nvmon", description="A fancy NVIDIA GPU monitor for the terminal.",
        epilog="At start nvmon asks PyPI whether a newer release is out; NVMON_NO_UPDATE_CHECK=1 turns that "
               "off. Colors are 24-bit where the terminal says it shows them, else 256. The view, the theme, the "
               "list's order and hidden GPUs are kept for each computer you connect from, in "
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
        view.interval = args.interval
        kept = view.kept()
        with Screen() as screen:
            view.shows = color_mode(screen.truecolor)
            # peek = (GPUs, round): when more GPUs' processes are to show their utilization, as after a click on
            # a job, the screen takes the newest numbers as soon as those GPUs have been read, not at the next
            # frame; or once that round of polls is over, for a GPU that cannot tell (with MIG, say).
            frame_at, poll_at, redraw, peek, size = time.monotonic(), None, True, None, None
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
                if redraw or tuple(os.get_terminal_size()) != size:  # a window resized redraws too
                    width, height = size = tuple(os.get_terminal_size())
                    screen.draw(view.screen(gpus, width, height,
                                            header(width, args.interval, driver, poller.waiting()), update.newer),
                                view.shows, view.theme.gray, size)
                # Keys and clicks redraw at once, the pointer when it goes to another graph column; the
                # numbers move on at the next frame. A quarter second at most, to see a resized window soon
                # with a long interval too.
                wait = 0.01 if peek is not None else (poll_at or frame_at) - time.monotonic()
                events = screen.read(min(wait, 0.25))
                pointed = view.pointed()
                going = all(map(view.handle, events))
                if view.kept() != kept:  # a setting changed: keep it for next time
                    kept = view.kept()
                    settings.save(kept, indices, host)
                if not going:
                    break
                redraw = any(e[0] != "move" for e in events) or view.pointed() != pointed
                detail = view.detail(gpus)
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
