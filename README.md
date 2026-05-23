# MP-STGODE

This repository contains the main source code for MP-STGODE, a metapopulation-based spatiotemporal graph neural ODE model for epidemic forecasting with mobility-aware mechanistic dynamics.

Only the core code is included in this repository. Datasets, generated outputs, model checkpoints, figures, and IDE files are intentionally excluded.

## Files

```text
.
|-- main.py              # Training, validation, testing, bootstrap aggregation
|-- model.py             # STGODE-SIR model and mechanistic rollout modules
|-- odegcn.py            # Neural ODE graph block
|-- Utils.py             # Data loading, preprocessing, split, and dataloader helpers
|-- requirements.txt     # Python package requirements
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

The default entry point expects a prepared pickle dataset:

```text
../MP_STGODE-master/datasets/Epi_data/Data_meta_stgode_causal7.pkl
```

Data files are not included in this upload. Update `--data_pkl`, `--resroot`, and `--output_dir` when running in a different directory layout.

## Run

```bash
python main.py
```

Common options can be overridden from the command line, for example:

```bash
python main.py --data_pkl path/to/Data_meta_stgode_causal7.pkl --device cuda:0
```

## Notes

- `main.py` sets OpenMP/MKL thread counts before importing numerical libraries.
- The model supports several forecast modes, gamma modes, encoder fusion strategies, and future mobility construction modes through command-line arguments.
- Generated artifacts such as `artifacts/`, `output_stgode_mob/`, `datasets/`, checkpoints, and result arrays are ignored by git.
