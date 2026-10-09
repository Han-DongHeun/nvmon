# nvmon

A fancy NVIDIA GPU monitor for the terminal. One Python file, no dependencies.

![nvmon showing four H100 GPUs one under the other, two people's jobs on them, the other four hidden, the pointer moving along a graph with the utilization and memory then beside it](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-wide.gif)

## Install

```bash
uv tool install nvmon    # or: pipx install nvmon
```

No uv? [Install it](https://docs.astral.sh/uv/getting-started/installation/), or use `pip install nvmon`
inside whichever environment you use.

Server without internet: copy `nvmon.py` over and run it with Python 3.6+.

## Start

```bash
nvmon              # Esc, q or Ctrl+C to quit
nvmon -i 1         # refresh every second (default: 0.5)
nvmon -g 0,2,4-7   # only these GPUs
ssh -t HOST nvmon  # over ssh: -t gives it a terminal
```

## Reading a box

- **Top edge**: number and name, power against its limit, temperature, fan (if any), clock. Warnings
  only when something is off: `SLOWED: power cap | too hot | hw brake`, `PCIe DEGRADED: x16 -> x8`.
- **Graph**: utilization over time, newest on the right. Point at it for the utilization and memory at that
  moment.
- **GPU**: how much of the time a kernel ran. **cores / tensor** (H100 and newer): how busy that kept
  the chip, and how much was Tensor Core math. One small kernel running nonstop reads GPU 100 % with
  cores at 30 %.
- **MEM**: memory in use. **CPU -> GPU**, **GPU -> CPU**: PCIe traffic against the link's top speed.
- **Bottom edge**: the processes, with run time and memory, by account and conda environment (yours
  underlined); `+3` for those that did not fit.

The bottom line has the GPU numbers (click one, or press its key, to hide or show it), the keys, and a
newer release when there is one.

## Views: `v`

`v` goes through four views, `V` back. What does not fit scrolls (wheel, PgUp/PgDn).

**Graphs**: one under the other; two columns only when they would not fit, as here with eight GPUs in
31 lines.

![nvmon with eight graphs in two columns](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/view-graphs.png)

**Processes** in the graph's place: account, run time, memory, command. The wheel scrolls them;
Shift+wheel or ← → sideways.

![nvmon with each GPU's processes in place of its graph](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/view-processes.png)

**Numbers only** and **GPU and MEM only**: narrow, always one column, for a window beside another.

<div align="center">

| Numbers only | GPU and MEM only |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/view-numbers.png" alt="nvmon with the numbers alone" width="200"> | <img src="https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/view-brief.png" alt="nvmon with GPU and MEM alone" width="200"> |

</div>

## Processes and jobs: `p`

A **job** is a process, or a launcher (torchrun, deepspeed…) with its workers.

- **Pick** one: click a process, or press Tab (Tab and the arrows move on). Its GPUs stand out, the
  list opens with it picked, and the bottom line sums it up.
- **Pick a GPU's jobs**: click anywhere else in its box. The same again lets go, and closes the list it
  opened.
- `f`, or a double click: only the picked jobs' GPUs; again for all.
- `p`: every job in a list. Click a heading to sort, again to reverse (or `s`, `r`). ← → scroll the
  picked job's command. Esc, or a click on an empty spot, lets go.
- `t` stops the picked jobs of yours (SIGTERM), `k` kills them (SIGKILL): after a `y`, and only those
  still the jobs you picked.

The list, busiest first, one job picked; and that job with `f`:

![nvmon with the process list open and a torchrun job picked](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/list.png)

![nvmon showing only the two GPUs of the picked job](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/focus.png)

## Themes: `c`

`c` goes through the themes, `C` back:

- **nvmon**, the default
- **Hiroshige, Hokusai, O'Keeffe, Van Gogh, Bénédictus, Cassatt, Greek, Tam, Homer, Demuth**: palettes
  of works at the Met, from [MetBrewer](https://github.com/BlakeRMills/MetBrewer)
- **Rosé Pine**, from the [editor theme](https://rosepinetheme.com)
- **black and white**, and **black on white**

![nvmon's themes, one box in each](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/themes.png)

## All keys

| Key | Does |
|---|---|
| `v` / `V` | next / previous view |
| `c` / `C` | next / previous theme |
| `p` | process list, open or close |
| point at a graph | utilization and memory then |
| click, Tab, ↑ ↓ | pick a job |
| click a GPU | pick its jobs |
| `f`, double click | only the picked jobs' GPUs, or all |
| `t`, `k`, then `y` | stop (SIGTERM) or kill (SIGKILL) the picked jobs |
| click a heading, `s`, `r` | sort the list, by the next heading, the other way |
| wheel, ← → | scroll; sideways the commands |
| PgUp, PgDn | scroll the GPUs |
| `0`-`9`, click a number | hide or show a GPU |
| Esc | a step back, then quit |
| `q`, Ctrl+C | quit |

Hold Shift to select text.

## It remembers

The view, theme, list order and hidden GPUs, for each computer you connect from, in
`~/.config/nvmon/settings.json` (`%APPDATA%\nvmon` on Windows). Delete it to start afresh.

## Terminals

- Line characters drawn two cells wide (Xshell, PuTTY and the like, set up for Korean, Chinese or
  Japanese): nvmon notices and draws in ASCII. No 24-bit color reported: 256 colors.
- Boxes still broken: use a font with one-cell line characters (D2Coding, Cascadia Mono, JetBrains Mono,
  DejaVu Sans Mono) and UTF-8.
- Windows: use Windows Terminal. There is no mouse, and no `t` / `k`.
- nvmon asks PyPI once at start for a newer release; `NVMON_NO_UPDATE_CHECK=1` turns that off.

## License

MIT. I designed nvmon's features and usability and tested it on shared H100 servers; most of the code was
written with an AI coding assistant.
