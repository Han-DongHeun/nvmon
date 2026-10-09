"""Write the README's pictures of nvmon on a made-up GPU server: a GIF and a still of it (for link previews and
tool directories), stills of its views and of the process list, and one of its themes:

    uv run --with pillow docs/demo.py

The server is a script: each GPU's utilization and memory over time and the processes on it, the rest as an H100
shows it. So the pictures come out the same each time, but for the clock, and need no GPU; only DejaVu Sans Mono
for the text.
"""
import itertools
import math
import os
import random
import re
import sys
import time
from collections import deque

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nvmon  # noqa: E402

# ── the server ───────────────────────────────────────────────────────────────

WARM, SECONDS = 240, 10  # half-second ticks that fill the graphs before the GIF; the GIF's length
TICKS = WARM + 2 * SECONDS


class Clock:
    """nvmon's time: as in the real world, but for time.time(), which goes on half a second a tick here."""
    now = time.time()

    def time(self):
        return Clock.now

    def __getattr__(self, name):
        return getattr(time, name)


# Utilization over TICKS ticks, by kind of work.

def phases(rng, segments):
    """By turns: (ticks, level, spread, gaps) segments, the last long enough to reach the end; `gaps`, the
    chance a tick drops near zero (a step waiting on something)."""
    out = [3 + rng.uniform(0, 8) if rng.random() < gaps else level + rng.gauss(0, spread)
           for ticks, level, spread, gaps in segments for _ in range(ticks)]
    return out[:TICKS]


def blocks(rng, high=95, low=18, busy=(12, 22), quiet=(5, 12)):
    """A sweep: run after run, each busy for a while, a pause between."""
    out = []
    while len(out) < TICKS:
        out += [high + rng.gauss(0, 4) for _ in range(rng.randint(*busy))]
        out += [low + rng.gauss(0, 8) for _ in range(rng.randint(*quiet))]
    return out[:TICKS]


def wavering(rng, center=45):
    """A job its data loader keeps waiting: never long busy, never long idle."""
    out, u = [], 40.0
    for _ in range(TICKS):
        u += (center - u) * 0.3 + rng.uniform(-24, 24)
        out.append(min(88, u))
    return out


def bursts(rng, low=6, high=84):
    """Serving requests: bursts of work, quiet between."""
    out = []
    while len(out) < TICKS:
        busy = rng.random() < 0.5
        out += [(high if busy else low) + rng.uniform(-10, 8)
                for _ in range(rng.randint(3, 12) if busy else rng.randint(2, 9))]
    return out[:TICKS]


def idle(rng):
    return [0] * TICKS


# The share of the utilization the SMs and the Tensor Cores are busy, by kind of work (cores, tensor)
KINDS = {phases: (0.7, 0.45), blocks: (0.74, 0.48), wavering: (0.55, 0.28), bursts: (0.62, 0.34), idle: (0, 0)}


class Scripted(nvmon.Gpu):
    """An H100 whose readings come from a script, not a driver: `kind` of work (with `shape`, its arguments)
    for `procs`, which hold their memory; `hot`: held back by heat."""

    def __init__(self, index, kind, procs, hot=False, **shape):
        self.index, self.name = index, "H100 80GB HBM3"
        self.history, self.memory = deque(maxlen=nvmon.HISTORY), deque(maxlen=nvmon.HISTORY)
        self.sample = self.now = None
        self.pcie_max_gen, self.pcie_max_width, self.has_activity = 5, 16, True
        self.utils = kind(random.Random(index), **shape)
        self.kind, self.procs, self.hot = kind, procs, hot
        self.rng, self.tick, self.temp = random.Random(index * 7 + 3), 0, 31.0

    def poll(self, detail=False):
        u = max(0, min(100, round(self.utils[min(self.tick, TICKS - 1)])))
        self.tick += 1
        # It warms and cools slowly; held back by heat, until it has cooled
        self.temp += (31 + (0.42 * u + 28 * self.hot) * (u > 0) - self.temp) * 0.06
        cores, tensor = KINDS[self.kind]
        held = sum(p.mem for p in self.procs)
        busy = bool(self.procs)
        self.sample = nvmon.Sample(
            util=u, temp=round(self.temp), fan=None, power_limit=700,
            power=min(700, 72 + 6.2 * u + self.rng.uniform(-8, 8)) if busy else 70 + self.rng.uniform(-3, 3),
            clock=(1350 + self.rng.randint(-60, 60) if self.hot else 1980) if busy else 345,
            mem_used=held + self.rng.uniform(-0.05, 0.05) if busy else 0.0, mem_total=79.6, pcie_width=16,
            tx=(0.02 + 0.06 * self.rng.random()) * 1e9 * (u > 0),
            rx=((6 if self.kind is wavering else 0.5) * u / 100 + 0.05 * self.rng.random()) * 1e9 * (u > 0),
            cores=cores * u + self.rng.uniform(0, 2) * (u > 0), tensor=tensor * u,
            slowdown=("SLOWED: too hot", nvmon.ALERT) if self.hot and busy else None,
            processes=self.procs,
            process_util={p.pid: u * p.mem / held for p in self.procs} if detail else None)  # the bigger, the busier


def proc(pid, name, owner, env, minutes, mem, args, job=None, launcher=None):
    """A process `minutes` old holding `mem` GiB; `owner` None for ours."""
    return nvmon.Process(pid, name, name + " " + args, mem, owner, env, Clock.now - 60 * minutes, job or pid,
                         launcher)


def server():
    """The eight GPUs, and the four the GIF shows: han's training on 0-1 (busy, then less so; low and broken),
    with a checkpoint being evaluated and yoon's notebook beside it; yoon's sweeps on 2-3 (one in big blocks,
    on a GPU held back by heat); han's model being served, our evaluation in shards waiting on its data loader,
    yoon's notebooks holding memory, and one idle. Only those two names, and ours."""
    han = [proc(241801 + k, "train.py", "han", "torch", 312, mem, "--config configs/train.yaml", 241800, "torchrun")
           for k, mem in enumerate((4.6, 5.2))]
    sweep = [proc(pid, "sweep.py", "yoon", "jax", minutes, mem, "--lr " + lr) for pid, minutes, mem, lr in
             ((287700, 97, 30.2, "1e-4"), (287731, 41, 6.9, "2e-4"), (287765, 12, 4.9, "5e-4"),
              (270500, 1800, 18.3, "3e-4"), (270544, 1680, 6.2, "3e-5"))]
    checkpoint = proc(330112, "eval_ckpt.py", "han", "torch", 3, 0.8, "--ckpt runs/step_42000")
    notebook = proc(301877, "ipykernel_launcher", "yoon", "jax", 95, 2.1, "-f kernel-2.json")
    shards = [proc(325017 + k, "eval.py", None, "torch", 28 - k // 2, 3.7, "--shard {}/6".format(k)) for k in range(6)]
    gpus = [Scripted(0, phases, han[:1] + [checkpoint], segments=[(190, 93, 6, 0.03), (200, 34, 9, 0.08)]),
            Scripted(1, phases, han[1:] + [notebook], segments=[(400, 20, 8, 0.15)]),
            Scripted(2, blocks, sweep[:3], hot=True),
            Scripted(3, wavering, sweep[3:], center=35),
            Scripted(4, bursts, [proc(198733, "serve.py", "han", "vllm", 1520, 71.2, "--model llama-3-8b")]),
            Scripted(5, wavering, shards),
            Scripted(6, idle, [proc(287390, "ipykernel_launcher", "yoon", "hf", 2900, 31.5, "-f kernel.json"),
                               proc(287455, "ipykernel_launcher", "yoon", "hf", 300, 2.4, "-f kernel-1.json")]),
            Scripted(7, idle, [])]
    return gpus, (0, 1, 2, 3)


# ── the GIF ──────────────────────────────────────────────────────────────────

GIF = ("demo-wide.gif", 120, 4 * (nvmon.MAX_INNER_H + 2) + 2)  # (file, columns, rows): four boxes one under the other
STEPS = 6  # frames a half second while the pointer is in, as nvmon redraws on each move
STILL_AT = 5.2  # seconds into the GIF for its still: the pointer resting on a graph, its chip shown


def pointer_path(view, cols):
    """Where the pointer goes in the GIF, as (seconds, (row, column), bow) keyframes: in from the right, onto
    the busiest GPU's graph, left along it, a rest, a little further, then out to the right again. Each way
    there bends by `bow` rows, as a hand's does."""
    rows, graph, _ = max(view.graphs, key=lambda shown: sum(shown[2].history))
    mid, right = rows.start + len(rows) // 2, graph.stop
    return [(1.5, (mid + 3.5, cols + 2), 0), (2.7, (mid + 0.4, right - 5), -1.5), (4.8, (mid - 0.3, right - 33), 0.6),
            (5.6, (mid - 0.3, right - 33), 0), (6.3, (mid + 0.5, right - 41), 0.4), (8.0, (mid - 2, cols + 3), -2)]


def along(path, t):
    """The pointer's (row, column) at `t` seconds on `path`, or None before and after: each way eased in and
    out, as a hand starts and stops, and bent by its bow."""
    if not path[0][0] <= t <= path[-1][0]:
        return None
    for (t0, a, _), (t1, b, bow) in zip(path, path[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1
            eased = u * u * (3 - 2 * u)
            return (a[0] + (b[0] - a[0]) * eased + bow * math.sin(math.pi * eased), a[1] + (b[1] - a[1]) * eased)
    return None


def record(gpus, shown, driver):
    """The GIF's screens, every half second and more often while the pointer moves: [(lines, pointer or None,
    milliseconds)], after WARM ticks have filled the graphs."""
    def tick():
        for gpu in gpus:
            gpu.poll(detail=True)
            gpu.frame()
    for _ in range(WARM):
        tick()
        Clock.now += 0.5
    _, cols, rows = GIF
    view = nvmon.View()
    view.hidden = {gpu.index for gpu in gpus if gpu.index not in shown}  # as a click on their numbers does
    frames, path = [], None
    for k in range(2 * SECONDS):
        tick()
        if path is None:
            view.screen(gpus, cols, rows, [], None)  # where the graphs are, for the pointer's path
            path = pointer_path(view, cols)
        steps = STEPS if any(along(path, k / 2 + s / 2 / STEPS) for s in range(STEPS)) else 1
        for s in range(steps):
            at = along(path, k / 2 + s / 2 / steps)
            cell = at and (round(at[0]), round(at[1]))  # outside the window the terminal says nothing
            view.pointer = cell if cell and 0 <= cell[0] < rows and 0 <= cell[1] < cols else None
            frames.append((view.screen(gpus, cols, rows, nvmon.header(cols, 0.5, driver), None), at, 500 // steps))
            Clock.now += 0.5 / steps
    return frames


# ── pictures ─────────────────────────────────────────────────────────────────

def biggest_job(view):
    """The job most worth showing: one with a launcher (torchrun), the most memory."""
    return max(view.jobs, key=lambda job: (job.launcher is not None, job.mem)).pid


def listing(view):
    view.listing, view.sort = True, ("util", True)
    view.pick([biggest_job(view)])


def focus(view):
    view.pick([biggest_job(view)])
    view.only = True


# (file, columns, rows, what is set on the view). The cards' windows are as narrow as the cards, as a window
# kept beside another; the numbers' window too short for all eight, the brief ones' just tall enough.
STILLS = [("view-graphs.png", 160, 31, lambda view: None),  # too short for eight one under the other
          ("view-processes.png", 160, 31, lambda view: setattr(view, "left", "processes")),
          ("view-numbers.png", nvmon.CARD_W, 31, lambda view: setattr(view, "left", "numbers")),
          ("view-brief.png", nvmon.CARD_W, 8 * (nvmon.MIN_INNER_H + 2) + 2,
           lambda view: setattr(view, "left", "brief")),
          ("list.png", 160, 48, listing),
          ("focus.png", 160, 31, focus)]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf"
CELL_W, CELL_H, PAD, SIZE = 9, 18, 14, 15
BG, FG = (24, 24, 27), (215, 215, 215)


def palette256(n):
    if n < 16:
        return [(0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0), (0, 0, 238), (205, 0, 205), (0, 205, 205),
                (229, 229, 229), (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0), (92, 92, 255),
                (255, 0, 255), (0, 255, 255), (255, 255, 255)][n]
    if n < 232:
        r, g, b = (n - 16) // 36, (n - 16) // 6 % 6, (n - 16) % 6
        return tuple(0 if v == 0 else 55 + 40 * v for v in (r, g, b))
    return (8 + 10 * (n - 232),) * 3


def colors(style):
    """(foreground, background, bold) for one of nvmon's styles."""
    fg, bg, faint = FG, BG, style.startswith(nvmon.FADED)
    for layer, r, g, b, code in re.findall(r"\x1b\[([34])8;(?:2;(\d+);(\d+);(\d+)|5;(\d+))m", style):
        color = palette256(int(code)) if code else (int(r), int(g), int(b))
        fg, bg = (color, bg) if layer == "3" else (fg, color)
    if faint:
        fg = tuple((a + b) // 2 for a, b in zip(fg, BG))
    if "\x1b[7m" in style:
        fg, bg = bg, fg
    return fg, bg, "\x1b[1m" in style


BLOCKS = {c: i + 1 for i, c in enumerate("▁▂▃▄▅▆▇█")}


def box(draw, ch, x, y, fg):
    """Box-drawing characters as lines, so that they join from cell to cell."""
    cx, cy, r = x + CELL_W // 2, y + CELL_H // 2, 4
    right, left, down, up = (x + CELL_W, cy), (x, cy), (cx, y + CELL_H), (cx, y)
    lines = {"─": [(left, right)], "│": [(up, down)], "├": [(up, down), ((cx, cy), right)],
             "┤": [(up, down), (left, (cx, cy))]}.get(ch)
    if lines:
        for a, b in lines:
            draw.line([a, b], fill=fg)
        return True
    arcs = {"╭": ((cx, cy + r), down, (cx + r, cy), right, (cx, cy, cx + 2 * r, cy + 2 * r), 180, 270),
            "╮": ((cx, cy + r), down, (cx - r, cy), left, (cx - 2 * r, cy, cx, cy + 2 * r), 270, 360),
            "╰": ((cx, cy - r), up, (cx + r, cy), right, (cx, cy - 2 * r, cx + 2 * r, cy), 90, 180),
            "╯": ((cx, cy - r), up, (cx - r, cy), left, (cx - 2 * r, cy - 2 * r, cx, cy), 0, 90)}.get(ch)
    if arcs:
        a1, a2, b1, b2, bounds, start, end = arcs
        draw.line([a1, a2], fill=fg)
        draw.line([b1, b2], fill=fg)
        draw.arc(bounds, start, end, fill=fg)
        return True
    return False


def picture(lines, cols, rows, fonts):
    image = Image.new("RGB", (cols * CELL_W + 2 * PAD, rows * CELL_H + 2 * PAD), BG)
    draw = ImageDraw.Draw(image)
    for r, line in enumerate(lines):
        x0, y = PAD, PAD + r * CELL_H
        col = 0
        for seg in line:
            text, style = seg[0], seg[1]
            fg, bg, bold = colors(style)
            if bg != BG and text:
                draw.rectangle([x0 + col * CELL_W, y, x0 + (col + len(text)) * CELL_W - 1, y + CELL_H - 1], fill=bg)
            for i, ch in enumerate(text):
                x = x0 + (col + i) * CELL_W
                if ch == " ":
                    continue
                if ch in BLOCKS:
                    top = y + CELL_H - CELL_H * BLOCKS[ch] // 8
                    draw.rectangle([x, top, x + CELL_W - 1, y + CELL_H - 1], fill=fg)
                elif ch == "▀":  # the graph's full cells: the upper half; the lower half is the background
                    draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H // 2 - 1], fill=fg)
                elif not box(draw, ch, x, y, fg):
                    draw.text((x, y + 1), ch, font=fonts[bold], fill=fg)
            col += len(text)
    return image


def arrow(image, at):
    """`image` with the mouse pointer on it, its tip at `at` = (row, column) in cells, as the terminal would
    draw it (a picture of the screen does not show it)."""
    x = PAD + at[1] * CELL_W + CELL_W // 2
    y = PAD + at[0] * CELL_H + CELL_H // 2
    shape = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    ImageDraw.Draw(image).polygon([(x + a, y + b) for a, b in shape], fill=(255, 255, 255), outline=(0, 0, 0))
    return image


def gallery(gpu, fonts, width=80):
    """`gpu`'s box once in each theme, its name above, two to a row."""
    boxes = []
    for theme in nvmon.THEMES:
        nvmon.THEME = theme
        lines = nvmon.panel(gpu, width, 5)
        if theme.gray:
            lines = [[(seg[0], nvmon.restyle(seg[1], "truecolor", True)) + tuple(seg[2:]) for seg in line]
                     for line in lines]
        boxes.append(picture([[(theme.name, nvmon.BOLD)]] + lines, width, 8, fonts))
    nvmon.THEME = nvmon.THEMES[0]
    w, h = boxes[0].size
    image = Image.new("RGB", (2 * w, math.ceil(len(boxes) / 2) * h), BG)
    for k, box in enumerate(boxes):
        image.paste(box, (k % 2 * w, k // 2 * h))
    return image


def save(image, here, name):
    image.save(os.path.join(here, name), optimize=True)
    print(name, os.path.getsize(os.path.join(here, name)) // 1024, "KB")


def main():
    nvmon.time = Clock()
    nvmon.socket.gethostname = lambda: "gpu-node"
    nvmon.__version__ = nvmon.__version__.split("+")[0]  # as released, not "+dev"
    driver = "driver 580.173.02  CUDA 13.0"
    fonts = {False: ImageFont.truetype(FONT.format(""), SIZE), True: ImageFont.truetype(FONT.format("-Bold"), SIZE)}
    here = os.path.dirname(os.path.abspath(__file__))
    gpus, shown = server()
    frames = record(gpus, shown, driver)
    name, cols, rows = GIF
    pictures = [picture(lines, cols, rows, fonts) for lines, _, _ in frames]
    pictures = [arrow(p, at) if at else p for p, (_, at, _) in zip(pictures, frames)]
    starts = itertools.accumulate([0] + [ms for _, _, ms in frames])
    save(next((p for p, start in zip(pictures, starts) if start >= STILL_AT * 1000), pictures[-1]), here,
         name.replace(".gif", ".png"))
    pictures = [p.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for p in pictures]
    pictures[0].save(os.path.join(here, name), save_all=True, append_images=pictures[1:],
                     duration=[ms for _, _, ms in frames], loop=0, optimize=True)
    print(name, os.path.getsize(os.path.join(here, name)) // 1024, "KB")
    for name, cols, rows, setup in STILLS:
        view = nvmon.View()
        view.screen(gpus, cols, rows, [], None)  # the jobs, for setup to pick from
        setup(view)
        save(picture(view.screen(gpus, cols, rows, nvmon.header(cols, 0.5, driver), None), cols, rows, fonts), here,
             name)
    # The themes on a GPU busy all along: its graph a wall of the theme's colors, bottom to top, ragged at the top
    busy = Scripted(0, phases, gpus[0].procs, segments=[(TICKS, 88, 8, 0)])
    for _ in range(TICKS):
        busy.poll()
        busy.frame()
    save(gallery(busy, fonts), here, "themes.png")


if __name__ == "__main__":
    main()
