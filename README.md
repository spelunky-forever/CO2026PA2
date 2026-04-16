# Mel Spectrogram on RISC-V Vector Extension

NCKU CSIE 2026 Computer Organization Programming Assessment II repository.

## Overview

This project asks students to implement three computational kernels for mel spectrogram generation on RISC-V (`RV64GCV`) with RVV intrinsics.

Only `src/main.c` is edited and submitted.

## Assessment Scope

Implement exactly three functions in `src/main.c`:

1. `fft(real, imag, n)` — in-place complex FFT on separate real/imag arrays (`n` is a power of 2).
2. `power_spectrum(stft_data, num_frames, output)` — compute `re² + im²` per bin from interleaved STFT input.
3. `mel_filter_bank(power, mel_bank, num_frames, n_mels, n_freq_bins, output)` — frame-wise dot-product accumulation into mel bins.

Key constants are defined in `include/mel_spectrogram.h`.

## Project Structure

| File                         | Edit? | Description                                                 |
| ---------------------------- | :---: | ----------------------------------------------------------- |
| `src/main.c`                 |   ✓   | Student implementation (only submitted file)                |
| `data/`                      |   X   | Judge input/expected files                                  |
| `include/mel_spectrogram.h`  |   X   | Constants and function signatures                           |
| `scripts/judge.py`           |   X   | Score calculation used by `make judge`                      |
| `scripts/mel_spectrogram.py` |   X   | Simplified exam ground-truth reference                      |
| `scripts/reference.py`       |   X   | Full librosa reference pipeline for cross-checking          |
| `src/bench.c`                |   X   | Benchmark/judge harness                                     |
| `src/utils.c`                |   X   | Provided pipeline (`hann_window`, `stft`, `melspectrogram`) |
| `Makefile`                   |   X   | Build and judge commands                                    |

## Prerequisites

- This repository assumes the RISC-V toolchain, Spike, and `pk` are already installed and available in `PATH`.
- Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) before running `make judge`; the judge scripts depend on it.
- If you are using the provided Docker image `docker.io/asrlab/comp-org:pa2`, these tools are already included.

## Getting Started

`make judge` is the primary command during the exam.

Makefile commands:

```bash
make compile     # Compile student implementation
make run         # Compile + run on Spike, write output/results.csv
make judge       # compile + run + score results
make clean       # Remove build/ and output/
```

After `make judge`, scores are printed to the terminal.

Compile command (equivalent to `make compile`):

```bash
riscv64-unknown-linux-gnu-gcc -O3 -fno-tree-vectorize -static -march=rv64gcv -Wall -Iinclude -o build/bench src/bench.c src/main.c src/utils.c -lm
```

Run command on Spike:

```bash
spike --isa=RV64GCV_Zicntr /opt/riscv/riscv64-unknown-linux-gnu/bin/pk build/bench
```
