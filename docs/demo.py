"""Record nvmon on this machine's GPUs and write the README's GIFs, one per window size, and a picture of
its themes:

    uv run --with pillow docs/demo.py [SECONDS]

Process names, conda environments, accounts, commands and the host name are swapped for made-up ones,
so nothing of the machine or its users shows. Needs an NVIDIA GPU, and DejaVu Sans Mono for the text.
"""
import math
import os
import re
import sys
import time

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nvmon  # noqa: E402

# (file, columns, rows, what is on: "list" = the process list with a job picked, "cards" = the processes in place of the graphs)
SIZES = [("demo-wide.gif", 160, 31, None), ("demo-small.gif", 100, 30, None), ("demo-list.gif", 160, 56, "list"),
         ("demo-cards.gif", 160, 23, "cards")]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono{}.ttf"
CELL_W, CELL_H, PAD, SIZE = 9, 18, 14, 15
BG, FG = (24, 24, 27), (215, 215, 215)


def anonymize():
    """Swap every process's name, account, environment and command for made-up ones, consistently."""
    jobs, alone = iter(["train.py", "finetune.py", "pretrain.py"]), iter(["sweep.py", "eval.py", "bench.py"])
    owners, envs = iter(["kim", "lee", "park", "choi"]), iter(["torch", "jax", "vllm", "hf"])
    seen = {}

    def pick(pool, value, fallback):
        if value and (id(pool), value) not in seen:
            seen[id(pool), value] = next(pool, fallback)
        return seen.get((id(pool), value))

    real = nvmon.Gpu._facts_of

    def facts(self, pid):
        name, _, owner, env, started, job, launcher = real(self, pid)
        name = pick(jobs, name, "job.py") if launcher else pick(alone, name, "run.py")  # launched: training
        command = "{} --config configs/{}.yaml --seed {}".format(name, name[:-3], pid % 7)
        return (name, command, pick(owners, owner, "user"), pick(envs, env, "env"), started, job,
                "torchrun" if launcher else None)

    nvmon.Gpu._facts_of = facts
    nvmon.socket.gethostname = lambda: "gpu-node"


def record(seconds, warm_up=70):
    """Screens, as nvmon's segment lines, every half second: {file: [frame, ...]}, and the GPUs as they are
    at the end. The first `warm_up` seconds are not kept: they fill the graphs."""
    nv = nvmon.load_nvml()
    gpus, driver = nvmon.open_gpus(nv), nvmon.versions(nv)
    views = {name: nvmon.View() for name, *_ in SIZES}
    frames = {name: [] for name, *_ in SIZES}
    for tick in range(-int(warm_up * 2), int(seconds * 2)):
        start = time.monotonic()
        for gpu in gpus:
            gpu.poll(detail=True)
            gpu.frame()
        if tick < 0:
            time.sleep(max(0, 0.5 - (time.monotonic() - start)))
            continue
        for name, cols, rows, mode in SIZES:
            view = views[name]
            if mode == "list" and tick == 0:  # the list open, the biggest job with a launcher picked
                view.screen(gpus, cols, rows, [], None)
                view.listing = True
                view.job = max(view.jobs, key=lambda job: (job.launcher is not None, job.mem)).pid
            view.graphs = mode != "cards"
            frames[name].append(view.screen(gpus, cols, rows, nvmon.header(cols, 0.5, driver), None))
        time.sleep(max(0, 0.5 - (time.monotonic() - start)))
    return frames, gpus


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
        fg, bg = BG, fg
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
            if bg != BG:
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


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    anonymize()
    frames, gpus = record(seconds)
    fonts = {False: ImageFont.truetype(FONT.format(""), SIZE), True: ImageFont.truetype(FONT.format("-Bold"), SIZE)}
    here = os.path.dirname(os.path.abspath(__file__))
    for name, cols, rows, _ in SIZES:
        pictures = [picture(lines, cols, rows, fonts) for lines in frames[name]]
        pictures = [p.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for p in pictures]
        pictures[0].save(os.path.join(here, name), save_all=True, append_images=pictures[1:], duration=500,
                         loop=0, optimize=True)
        print(name, os.path.getsize(os.path.join(here, name)) // 1024, "KB")
    busiest = max(gpus, key=lambda gpu: sum(gpu.history))
    gallery(busiest, fonts).save(os.path.join(here, "themes.png"), optimize=True)
    print("themes.png", os.path.getsize(os.path.join(here, "themes.png")) // 1024, "KB")


if __name__ == "__main__":
    main()
