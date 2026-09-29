# Changelog

What changes for you in each release. The same notes appear on the
[releases page](https://github.com/Han-DongHeun/nvmon/releases).

## Unreleased

- No more stutter when another program uses the GPU driver at the same time, such as someone
  running `watch nvidia-smi`. The screen now refreshes on time and keeps the last numbers until the
  driver answers. If it has not answered for a second, the top line says so.
- Uses about a third of the CPU: 1.4 % of one core instead of 4.1 %, with eight H100s.
- A newer release is announced at the bottom left, with the command that updates your install and
  a link to what's new. nvmon asks PyPI once at start; `NVMON_NO_UPDATE_CHECK=1` turns that off.
- `nvmon -V` and `nvmon -h` show the project's address.
- `-i` below 0.1 s counts as 0.1 s, since the GPU's own readings don't change faster than that; anything
  that is not a number of seconds means the default, 0.5 s.
