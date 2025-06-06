# TODO
## MAIN
- [x] Dataset
- [x] Modelo de estimación de incertidumbre
- [x] Pipeline

Idea: Predecir offset/media además de la varianza?

## Dataset
### Synthetic data
https://github.com/apple/ml-hypersim

# Running the code
## Training
`uv run src/uncertainty_estimation/train.py --config trainer_config.yaml`

Or a sweep:

`uv run src/uncertainty_estimation/train.py --sweep True --config trainer_config.yaml`