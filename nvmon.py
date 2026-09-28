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
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_int, c_uint, c_ulonglong, c_void_p
from typing import NamedTuple, Optional

__version__ = "0.1.0"

INFO_W = 36                          # width of the stats column
SUFFIX_W = 11                        # numbers after a bar: "  98%  52°C"
BAR_W = INFO_W - 4 - 1 - SUFFIX_W    # "GPU " + bar + " " + numbers
CHROME_W = 7                         # "│ " + " │ " + " │" around graph and stats
MIN_GRAPH_W = 30                     # a narrower graph is not worth a second column
MAX_INNER_H = 8                      # tallest graph: 8 rows x 8 sub-levels = 64 steps
HISTORY = 1024                       # samples kept per GPU; wider than any terminal
BLOCKS = " ▁▂▃▄▅▆▇█"                 # a character cell filled in 1/8 steps


def _fg(n):
    return "\x1b[38;5;{}m".format(n)


# Fixed 256-colour palette: identical over ssh, tmux and local terminals.
GREEN, YELLOW, RED, CYAN, BLUE, DIM = map(_fg, (101, 179, 131, 73, 68, 240))
BOLD, RESET = "\x1b[1m", "\x1b[0m"


def heat(t):
    """Colour for a position t in [0, 1] of the 0-100 % scale."""
    return GREEN if t < 0.5 else YELLOW if t < 0.8 else RED


# ── NVML (libnvidia-ml / nvml.dll, part of the NVIDIA driver) ───────────────

NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_GRAPHICS = 0
NVML_PCIE_UTIL_TX_BYTES, NVML_PCIE_UTIL_RX_BYTES = 0, 1
NVML_DEVICE_NAME_BUFFER_SIZE = 96


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class Memory(ctypes.Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class MemoryV2(ctypes.Structure):
    _fields_ = [("version", c_uint), ("total", c_ulonglong), ("reserved", c_ulonglong),
                ("free", c_ulonglong), ("used", c_ulonglong)]


class TemperatureV1(ctypes.Structure):
    _fields_ = [("version", c_uint), ("sensorType", c_int), ("temperature", c_int)]


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
    pstate: Optional[int]
    clock: Optional[int]          # graphics clock, MHz
    mem_used: Optional[float]     # GiB
    mem_total: Optional[float]    # GiB
    tx: Optional[int]             # PCIe, KiB/s
    rx: Optional[int]             # PCIe, KiB/s


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

    def _uint(self, fn, *args, ctype=c_uint):
        """Scalar query; None when this GPU does not support it (e.g. fan on an H100)."""
        value = ctype()
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

    def poll(self):
        util = Utilization()
        util = util.gpu if self._ok("nvmlDeviceGetUtilizationRates", byref(util)) else None
        used, total = self._memory()
        power = self._uint("nvmlDeviceGetPowerUsage")            # mW
        limit = self._uint("nvmlDeviceGetEnforcedPowerLimit")    # mW
        pstate = self._uint("nvmlDeviceGetPerformanceState", ctype=c_int)
        self.now = Sample(
            util=util,
            temp=self._temperature(),
            fan=self._uint("nvmlDeviceGetFanSpeed"),
            power=None if power is None else power / 1000,
            power_limit=None if limit is None else limit / 1000,
            pstate=pstate if pstate is not None and 0 <= pstate < 16 else None,  # 32 = unknown
            clock=self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_GRAPHICS),
            mem_used=None if used is None else used / 2**30,
            mem_total=None if total is None else total / 2**30,
            # Each PCIe query samples a counter for ~20 ms, hence the parallel polling.
            tx=self._uint("nvmlDeviceGetPcieThroughput", NVML_PCIE_UTIL_TX_BYTES),
            rx=self._uint("nvmlDeviceGetPcieThroughput", NVML_PCIE_UTIL_RX_BYTES),
        )
        self.history.append(util or 0)


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
    """Stacked block chart, newest sample on the right, top row first."""
    vals = list(history)[-width:] if width > 0 else []
    steps = height * 8
    # Any non-zero value gets at least one sub-level so light load stays visible.
    levels = [0] * (width - len(vals)) + [max(round(v * steps / 100), 1 if v else 0) for v in vals]
    return [[("".join(BLOCKS[min(8, max(0, lv - r * 8))] for lv in levels), heat((r + 0.5) / height))]
            for r in reversed(range(height))]


def bar(frac, color):
    """Horizontal meter; `color` is a style or a function of the position along the bar."""
    filled = round(min(max(frac or 0, 0), 1) * BAR_W)
    cells = [(color(i / BAR_W) if callable(color) else color) if i < filled else DIM for i in range(BAR_W)]
    return [("■" * len(list(run)), style) for style, run in itertools.groupby(cells)]


def row(left, right=""):
    """One stats line: `left`, then `right` flush right; exactly INFO_W cells."""
    room = INFO_W - len(right)
    return pad(clip(left, room - 1 if right else room), room) + [(right, "")]


def ratio(part, whole):
    return part / whole if part is not None and whole else None


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
    lines = [
        row([(gpu.name, BOLD)], num(s.clock, "{} MHz")),
        row([("GPU ", "")] + bar(ratio(s.util, 100), heat),
            num(s.util, "{}%").rjust(5) + num(s.temp, "{}°C").rjust(6)),
        row([("PWR ", "")] + bar(ratio(s.power, s.power_limit), CYAN),
            num(s.power, "{:.0f}W").rjust(5) + num(s.pstate, "P{}").rjust(6)),
        row([("MEM ", "")] + bar(ratio(s.mem_used, s.mem_total), BLUE),
            "{}/{}G".format(num(s.mem_used, "{:.1f}"), num(s.mem_total, "{:.0f}")).rjust(SUFFIX_W)),
        row([("TX {} RX {}".format(rate(s.tx), rate(s.rx)), "")], num(s.fan, "FAN {}%") if s.fan is not None else ""),
    ]
    # A short panel keeps the most important lines, in display order.
    keep = sorted([1, 3, 2, 0, 4][:height])
    return [lines[i] for i in keep] + [row([])] * (height - len(keep))


def panel(gpu, width, height):
    graph_w = max(0, width - CHROME_W - INFO_W)
    title = " GPU {} ".format(gpu.index)
    top = [("╭─", DIM), (title, ""), ("─" * max(0, width - 3 - len(title)) + "╮", DIM)]
    body = [[("│ ", DIM)] + g + [(" │ ", DIM)] + i + [(" │", DIM)]
            for g, i in zip(graph(gpu.history, graph_w, height), info(gpu, height))]
    bottom = [("╰" + "─" * max(0, width - 2) + "╯", DIM)]
    return [top] + body + [bottom]


def render(gpus, width, height):
    """The whole screen: exactly `height` lines of exactly `width` cells."""
    # Two columns only when one column cannot show every GPU at full height.
    two_fit = width // 2 >= CHROME_W + INFO_W + MIN_GRAPH_W
    cols = 2 if two_fit and len(gpus) * (5 + 2) > height else 1
    rows = math.ceil(len(gpus) / cols)
    inner = max(1, min(MAX_INNER_H, height // rows - 2))
    lines = []
    for r in range(rows):
        panels = [panel(g, width // cols, inner) for g in gpus[r * cols:(r + 1) * cols]]
        lines += [[seg for part in parts for seg in part] for parts in zip(*panels)]
    lines = [pad(clip(line, width), width) for line in lines[:height]]
    return lines + [[(" " * width, "")]] * (height - len(lines))


def paint(lines):
    return "\r\n".join("".join(style + text + RESET if style else text for text, style in line)
                       for line in lines)


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
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        with ThreadPoolExecutor(len(gpus)) as pool, Screen() as screen:
            deadline = time.monotonic()
            while True:
                list(pool.map(Gpu.poll, gpus))
                screen.draw(render(gpus, *os.get_terminal_size()))
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
