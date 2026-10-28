# Benchmark baselines for MP-STGODE

Reference implementations of the benchmark models compared in the paper:
Fixed SIR, Rolling SIR, Meta-SIR, AR, Ridge-VAR, LSTM, GRU, STGCN, DCRNN,
STAN, and MepoGNN.

## Environment

Python 3.9 and the packages listed in `requirements.txt`. Install the PyTorch
build that matches your CUDA runtime first, then install the remaining
requirements.

## Data

The benchmark expects the prepared dataset at `../data/Data_meta_stgode_causal7.pkl`
(the same file used by the main model). Alternatively, pass an explicit path
with `--data-pkl path/to/Data_meta_stgode_causal7.pkl`.

## Usage

```bash
# Preview the commands without training
python main_benchmark_walk_forward.py --action plan --folds F2 F3

# Run all benchmark models on F2 and F3
python main_benchmark_walk_forward.py --action run --folds F2 F3

# Run the fixed 8:1:1 split (F4)
python main_benchmark_walk_forward.py --action run --folds F4 --f4-protocol fixed_8_1_1
```

Model settings default to each runner's own values. Override them with
`--set FOLD.MODEL.HORIZON.ARG=VALUE`, for example
`--set F2.stgcn.3.hidden_dim=128`.

## License

MIT; see `LICENSE`.
