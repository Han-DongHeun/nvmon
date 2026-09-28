# nvmon

A compact, btop-style NVIDIA GPU monitor for the terminal.

```
nvmon 0.1.0  gpu-node  2026-09-28 Mon 22:05:00.07                                refresh 0.5s
╭─ GPU 6  H100 80GB HBM3 ────────────────────────────────────────────────  49°C  1980 MHz ─╮
│ ▃  ▁▁▄      ▄    ▇   ▆ ▄   ▂   ▇ ▆▃   ▆      █    ▁   ▃ │ GPU  67%                       │
│ ██▅███▆▄▄▃▇▃█ ▂▃ █ ▆▄█ █▄▆▄█▂▆██ ██▅█▅██▆▆▆▆▃█  ▆▆█▆▆██ │ MEM  20%         15.6/79.6 GiB │
│ █████████████▆██▅█▄███▆█████████▄█████████████▆ ███████ │ PWR  43%             299/700 W │
│ ███████████████████████████████████████████████▇███████ │  TX 6.39 MiB/s   RX  163 MiB/s │
╰─────────────────────────── train.py 15.1G  eval.py 0.5G ─────────────────────────────────╯
                                                                                Esc / q quit
```

- One box per GPU: utilization history on the left (0 % at the bottom, 100 % at the top),
  utilization, memory, power and PCIe traffic on the right.
- Top edge: GPU name, fan, temperature (blue when cool, red from about 85 °C) and clock.
  Bottom edge: running processes (script names for Python) with their memory.
- Top line: host, local time, driver / CUDA version and refresh interval.
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
nvmon              # Esc, q or Ctrl+C to quit
nvmon -i 1         # update every second
nvmon -g 0,2,4-7   # only these GPUs
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
