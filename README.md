# nvmon

A compact, btop-style NVIDIA GPU monitor for the terminal.

```
nvmon 0.1.0  gpu-node  2026-09-28 Mon 21:39:29.75    1 GPU  299 W  16 / 80 GiB   refresh 0.5s
╭─ GPU 6  H100 80GB HBM3 ───────────────────────────────────────────────── 49°C  1980 MHz ─╮
│ ▄ ▃  ▁▁▄      ▄    ▇   ▆ ▄   ▂   ▇ ▆▃   ▆      █    ▁   ▃ │ GPU                      67% │
│ █ ██▅███▆▄▄▃▇▃█ ▂▃ █ ▆▄█ █▄▆▄█▂▆██ ██▅█▅██▆▆▆▆▃█  ▆▆█▆▆██ │ MEM  15.6 / 79.6 GiB     20% │
│ █▅█████████████▆██▅█▄███▆█████████▄█████████████▆ ███████ │ PWR  299 / 700 W         43% │
│ █████████████████████████████████████████████████▇███████ │ TX 6.39 MiB/s RX  163 MiB/s  │
╰─ train.py 15.1G  eval.py 0.5G ───────────────────────────────────────────────────────────╯
```

- One box per GPU: utilization history on the left (0 % at the bottom, 100 % at the top),
  utilization, memory, power and PCIe traffic on the right.
- GPU name, fan, temperature and clock on the top edge; running processes (script names
  for Python) on the bottom edge.
- Top line: host, local time, driver / CUDA version and totals over all GPUs.
- Refreshes every 0.5 s. Up to 8 GPUs fit on one screen; on wide terminals the boxes
  go into two columns.
- **Zero dependencies.** One Python file that talks to the NVML library shipped with the
  NVIDIA driver. Runs on any Python >= 3.6, Linux or Windows, no root needed.

## Install

```bash
uv tool install nvmon      # or: pipx install nvmon
```

Or run it once without installing: `uvx nvmon`.

**Servers without internet access:** `nvmon.py` is the whole program. Copy it and run it:

```bash
ssh HOST 'mkdir -p ~/.local/bin && cat > ~/.local/bin/nvmon && chmod +x ~/.local/bin/nvmon' < nvmon.py
```

## Usage

```bash
nvmon              # Ctrl+C to quit
nvmon -i 1         # update every second
ssh -t HOST nvmon  # over ssh, -t gives the remote program a terminal
```

Requirements: an NVIDIA driver (for NVML) and a truecolor terminal (Windows Terminal, VS Code,
iTerm2, GNOME Terminal, kitty, WezTerm, ...).
GPUs that the current user may not access (e.g. outside a Slurm allocation) are skipped.

## Releasing

1. Bump `__version__` in `nvmon.py` and commit.
2. `git tag -a v0.1.1 -m v0.1.1 && git push --follow-tags`
3. The `publish` GitHub Actions workflow builds and uploads to PyPI (Trusted Publishing).

## License

MIT
