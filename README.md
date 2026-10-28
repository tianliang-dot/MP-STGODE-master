# MP-STGODE

This repository contains the source code for MP-STGODE, a metapopulation-based spatiotemporal graph neural ODE model for epidemic forecasting with mobility-aware mechanistic dynamics. It also provides the benchmark baseline implementations used in the paper (see `benchmark/`).

## Repository layout

```text
.
|-- main.py                  # Training, validation, testing, bootstrap aggregation
|-- model.py                 # STGODE-SIR model and mechanistic rollout modules
|-- odegcn.py                # Neural ODE graph block
|-- Utils.py                 # Data loading, preprocessing, split, and dataloader helpers
|-- requirements.txt         # Python package requirements
|-- data/                    # Shared prepared dataset
|-- benchmark/               # Benchmark baseline implementations
`-- README.md
```

## Environment

The requirements were prepared from the local `pinnstorch_master` conda environment with Python 3.9.19.

```bash
conda create -n mp-stgode python=3.9.19
conda activate mp-stgode
pip install -r requirements.txt
```

## Data

The prepared dataset is provided at:

```text
data/Data_meta_stgode_causal7.pkl
```

If the file is stored elsewhere, pass an explicit path with `--data_pkl`.

## Run

```bash
python main.py
```

Common options can be overridden from the command line:

```bash
python main.py --data_pkl data/Data_meta_stgode_causal7.pkl --device cuda:0
```

## Benchmark baselines

The benchmark implementations are under `benchmark/` (Fixed SIR, Rolling SIR, Meta-SIR, AR, Ridge-VAR, LSTM, GRU, STGCN, DCRNN, STAN, and MepoGNN). See `benchmark/README.md` for details.

## Notes

- `main.py` sets OpenMP/MKL thread counts before importing numerical libraries.
- The model supports several forecast modes, gamma modes, encoder fusion strategies, and future mobility construction modes through command-line arguments.
- Generated artifacts such as `artifacts/`, `output_stgode_mob/`, checkpoints, and result arrays are ignored by git.

## License

MIT; see `LICENSE`.
