# nvmon

A fancy NVIDIA GPU monitor for the terminal. Single Python file, no dependencies.

![nvmon showing eight H100 GPUs in two columns](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-wide.gif)

It fits the window. In a small one the boxes get shorter, and scroll once they no longer fit:

![nvmon in a small window, scrolling through the GPUs](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-small.gif)

`p` lists every job; a picked one stands out on the GPUs it uses:

![nvmon with the process list open and a torchrun job picked](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-list.gif)

`g` turns the graphs off, for more GPUs in less room:

![nvmon without graphs, the GPUs as cards three to a row](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-cards.gif)

## Install

```bash
uv tool install nvmon    # or: pipx install nvmon
```

No uv? [Install it](https://docs.astral.sh/uv/getting-started/installation/), or use `pip install nvmon` inside whichever environment you use.

On a server without internet, copy `nvmon.py` over and run it with any Python 3.6+.

## Usage

```bash
nvmon              # Esc, q or Ctrl+C to quit
nvmon -i 1         # refresh every second (default: 0.5)
nvmon -g 0,2,4-7   # only these GPUs
ssh -t HOST nvmon  # over ssh: -t gives it a terminal
```

Click a process, or press Tab, to pick its job: the GPUs it uses stand out, and `f` shows only those.
`t` asks the job to stop (SIGTERM: it may save and exit), `k` ends it at once (SIGKILL); both ask for a
`y` first. `p` lists every job with its utilization and command line; click a heading to sort by it.
Click an empty spot to let the job go and close the list.

The GPU numbers at the bottom left hide a GPU when clicked, and show it again; so does the number's
key. `g` turns the graphs on and off. `c` goes through the themes (`C` back): nvmon's own, then palettes
from works by Hiroshige, Hokusai, O'Keeffe, Van Gogh, Bénédictus and Cassatt (as [MetBrewer](https://github.com/BlakeRMills/MetBrewer)
has them) and the [Rosé Pine](https://rosepinetheme.com) theme's, then black and white. The theme,
graphs on or off, the list's order and the GPUs hidden stay for the next run, for each computer you
connect from (in `~/.config/nvmon/settings.json`). Hold Shift to select text with the mouse.

![nvmon's themes, one box in each](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/themes.png)

## What it shows

- **GPU**: how much of the time something ran on it. **cores / tensor** (H100 and newer): how much
  of the chip that work kept busy, and how much of it was matrix math on the Tensor Cores. A GPU
  reads 100 % as soon as one small kernel runs all the time, while its cores may sit at 30 %.
- **MEM**, data sent **CPU -> GPU** and **GPU -> CPU** against the link's top speed, power,
  temperature, fan and clock.
- Processes on each GPU with run time and memory, grouped by account and conda environment
  (your own account unlabelled), script names for Python.
- Warnings only when something is off: `SLOWED: power cap | too hot | hw brake` when the clock is held back,
  `PCIe DEGRADED: x16 -> x8` when the card runs on fewer lanes than it supports.

## Notes

- Needs an NVIDIA driver.
- On Windows, run it in Windows Terminal; Git Bash's default window is not a terminal to Python.
- nvmon asks the terminal how it draws. One that puts line and block characters two cells wide (Xshell,
  PuTTY and the like, set up for Korean, Chinese or Japanese) gets boxes and graphs drawn in ASCII. One
  that does not say it shows 24-bit color gets 256 colors, which look nearly the same.
- Boxes still broken: the terminal moves on one cell, but its font draws the characters two wide. Pick a
  font whose line characters are one cell wide, and make sure the encoding is UTF-8.
- When a newer release is out, the bottom line says so, with the command that updates your install.
  nvmon asks PyPI once at start; `NVMON_NO_UPDATE_CHECK=1` turns that off.

## License

MIT
