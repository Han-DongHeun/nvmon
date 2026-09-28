# nvmon

A compact, btop-style NVIDIA GPU monitor for the terminal.

```
╭─ GPU 4 ──────────────────────────────────────────────────────────────────────╮
│                ▃▄▆██▇▆▅▃▂           │ H100 80GB HBM3              1980 MHz │
│           ▂▄▆█████████████▆▄        │ GPU ■■■■■■■■■■■■■■■■■■■■  100%  51°C │
│      ▁▃▅▇███████████████████▇▅▃     │ PWR ■■■■■■■■■■■■■■■■■■■■  378W    P0 │
│  ▁▃▆███████████████████████████▆▃▁  │ MEM ■■■■■■■■■■■■■■■■■■■■  60.5/80G │
│ ████████████████████████████████████│ TX 8.20 MiB/s RX 50.4 MiB/s FAN 72% │
╰──────────────────────────────────────────────────────────────────────────────╯
```

- One box per GPU: utilization history on the left (0 % at the bottom, 100 % at the top),
  clock, temperature, power, P-state, memory, PCIe traffic and fan on the right.
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

Requirements: an NVIDIA driver (for NVML) and a terminal with 256 colours.
GPUs that the current user may not access (e.g. outside a Slurm allocation) are skipped.

## Releasing

1. Bump `__version__` in `nvmon.py` and commit.
2. `git tag -a v0.1.1 -m v0.1.1 && git push --follow-tags`
3. The `publish` GitHub Actions workflow builds and uploads to PyPI (Trusted Publishing).

## License

MIT
