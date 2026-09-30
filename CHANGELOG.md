# Changelog

What changes for you in each release. The same notes appear on the
[releases page](https://github.com/Han-DongHeun/nvmon/releases).

## 0.3.2

- Four more themes from MetBrewer: Greek, Tam, Homer and Demuth.
- Black and white is back as it was in 0.3.0, grays as light as the colors; black on white is a theme of
  its own, after it.
- A shorter README, its GIF the four busiest GPUs one under the other.

## 0.3.1

- Numbers only, and GPU and MEM only, keep their boxes one under the other, never side by side; what
  does not fit scrolls. In a narrow window, the line on the GPUs left out is shorter, not cut off.
- The black and white theme draws the GPU boxes black on white, every color the gray as dark as it was
  light, the busiest bars the darkest; the rest of the screen stays on the terminal's own background.
- The README shows each view and the process list, and lists every key.

## 0.3.0

- Processes: click one, or press Tab, to pick its job (a torchrun job's workers count as one). The GPUs
  it uses stand out, and `f` shows only those. `t` asks it to stop (SIGTERM), `k` ends it at once
  (SIGKILL), each after a `y`; only your own jobs, and only once nvmon has checked they are still the
  same. `p` lists every job with its utilization and command line, sorted by whichever heading you
  click; a click on an empty spot closes it. A picked job's utilization shows at once.
- `v` (`V` back) goes from the graphs to each GPU's processes in their place (account, run time,
  memory, the command line; the wheel scrolls them, up and down or sideways, as do ← →), to the numbers
  alone, to GPU and MEM alone. Boxes go one under the other, side by side only when they would not fit.
- Hide GPUs you don't need: click their numbers at the bottom left, or press the number's key; again to
  show them. When the GPUs do not fit the window, their boxes scroll (wheel, PgUp/PgDn).
- Themes: `c` goes through them, `C` back. nvmon's own colors stay the default; the others take the
  palettes of works by Hiroshige, Hokusai, O'Keeffe, Van Gogh, Bénédictus and Cassatt, and of the Rosé
  Pine theme. Black and white keeps each color's lightness.
- The graph shades more smoothly: each cell shows two colors, twice the steps from green to red, and a
  height has one color from bar to bar. In 256 colors the bands stay even, with no dull olive among the
  greens.
- The view, the theme, the process list's order and the GPUs hidden stay for the next run, for each
  computer you connect from (`~/.config/nvmon/settings.json`).
- Works in Xshell, PuTTY and the like without setting them up: nvmon asks the terminal how it draws.
  Where lines come out two cells wide (Korean, Chinese, Japanese settings), boxes and graphs are drawn
  in ASCII; where 24-bit color is not known to work, 256 colors are used. Windows Terminal (over
  Windows's own ssh too), Tabby, VS Code's terminal, kitty, iTerm2, WezTerm, Konsole and GNOME Terminal
  get their 24-bit color.
- The keys work with a Korean keyboard in Hangul mode: ㅂ is q, ㅊ is c, and so on. A capital A, B, C,
  D, F, H or Z is that letter, no longer an arrow key, Home or End.
- The boxes are a line shorter: the "PCIe transfer" title is gone, as CPU -> GPU and GPU -> CPU say it.
- No more stutter when another program uses the GPU driver at the same time, such as someone running
  `watch nvidia-smi`. The screen refreshes on time and keeps the last numbers until the driver answers;
  if it has not answered for a second, the top line says so.
- Uses about a third of the CPU: 1.4 % of one core instead of 4.1 %, with eight H100s.
- A newer release is announced at the bottom left, with the command that updates your install and a
  link to what's new. nvmon asks PyPI once at start; `NVMON_NO_UPDATE_CHECK=1` turns that off.
- `nvmon -V` and `nvmon -h` show the project's address.
- `-i` below 0.1 s counts as 0.1 s, since the GPU's own readings don't change faster than that; anything
  that is not a number of seconds means the default, 0.5 s.
