# nvmon

A fancy NVIDIA GPU monitor for the terminal. Single Python file, no dependencies.

![nvmon showing four H100 GPUs](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo.gif)

On a wide terminal, eight GPUs go into two columns:

![nvmon showing eight H100 GPUs in two columns](https://raw.githubusercontent.com/Han-DongHeun/nvmon/main/docs/demo-grid.gif)

## Install

```bash
uv tool install nvmon    # or: pipx install nvmon
```

No uv? `pip install nvmon` works too, inside whichever environment you use.

On a server without internet, copy `nvmon.py` over and run it with any Python 3.6+.

## Usage

```bash
nvmon              # Esc, q or Ctrl+C to quit
nvmon -i 1         # refresh every second (default: 0.5)
nvmon -g 0,2,4-7   # only these GPUs
ssh -t HOST nvmon  # over ssh: -t gives it a terminal
```

## What it shows

- **GPU**: share of time any kernel ran. **cores / tensor** (H100 and newer): share of SMs busy
  and Tensor Core activity. A GPU can read 100 % while its cores sit at 30 %.
- **MEM**, **PCIe** traffic in each direction against the link's top speed, power, temperature,
  fan and clock.
- Processes on each GPU with run time and memory, grouped by account and conda environment
  (your own account unlabelled), script names for Python.
- Warnings only when something is off: `SLOWED: power cap | too hot | hw brake` when the clock is held back,
  `PCIe DEGRADED: x16 -> x8` when the card runs on fewer lanes than it supports.

## Notes

- Needs an NVIDIA driver and a truecolor terminal. In tmux, add
  `set -ag terminal-overrides ",*:RGB"` to `~/.tmux.conf`.
- On Windows, run it in Windows Terminal; Git Bash's default window is not a terminal to Python.

## License

MIT
