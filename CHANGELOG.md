# Changelog

What changes for you in each release. The same notes appear on the
[releases page](https://github.com/Han-DongHeun/nvmon/releases).

## Unreleased

- Processes: click one, or press Tab, to pick its job (a torchrun job's workers count as one). The GPUs
  it uses stand out, and `f` shows only those. `t` asks it to stop (SIGTERM), `k` ends it at once
  (SIGKILL), each after a `y`; only your own jobs, and only once nvmon has checked they are still the
  same. `p` lists every job with its utilization and command line, sorted by whichever heading you click;
  a click on an empty spot closes it. A picked job's utilization shows at once.
- Hide GPUs you don't need: click their numbers at the bottom left, or press the number's key; again
  to show them. `g` turns the graphs off, leaving compact cards, several to a row.
- Works in Xshell, PuTTY and the like without setting them up: nvmon asks the terminal how it draws.
  Where lines come out two cells wide (Korean, Chinese, Japanese settings), boxes and graphs are drawn
  in ASCII; where 24-bit color is not known to work, 256 colors are used.
- The graph shades smoothly: each cell shows two colors, so there are twice as many steps from green
  to red, and in 256 colors no more dull olive among the greens.
- Themes: `c` goes through them, `C` back. nvmon's own colors stay the default; the others take the
  palettes of works by Hiroshige, Hokusai, O'Keeffe, Van Gogh, Bénédictus and Cassatt, and of the
  Rosé Pine theme; black
  and white keeps each color's lightness.
- The theme, graphs on or off, the process list's order and the GPUs hidden stay for the next run, for
  each computer you connect from.
- A capital A, B, C, D, F, H or Z typed is that letter, no longer an arrow key, Home or End.
- The keys work with a Korean keyboard in Hangul mode: ㅂ is q, ㅊ is c, and so on.
- When the GPUs do not fit the window, their boxes scroll (wheel, PgUp/PgDn). The boxes are a line
  shorter: the "PCIe transfer" title is gone, as CPU -> GPU and GPU -> CPU say it already.
- No more stutter when another program uses the GPU driver at the same time, such as someone
  running `watch nvidia-smi`. The screen now refreshes on time and keeps the last numbers until the
  driver answers. If it has not answered for a second, the top line says so.
- Uses about a third of the CPU: 1.4 % of one core instead of 4.1 %, with eight H100s.
- A newer release is announced at the bottom left, with the command that updates your install and
  a link to what's new. nvmon asks PyPI once at start; `NVMON_NO_UPDATE_CHECK=1` turns that off.
- `nvmon -V` and `nvmon -h` show the project's address.
- `-i` below 0.1 s counts as 0.1 s, since the GPU's own readings don't change faster than that; anything
  that is not a number of seconds means the default, 0.5 s.
