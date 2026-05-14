# SuperADD

This repository contains the code for the SuperADD submission to the VAND 4.0 Challenge.

SuperADD is a class-agnostic anomaly detection model and trained using only anomaly-free images.
It is based on [SuperAD](https://github.com/Summerdayhurricane/SuperAD/tree/main) by Zhang et al.

## Reproduce our results

### 1. Preparation
Download the DINOv3 weights from [the official meta ai site](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).
Place the weights into the `./weights/` directory.
If weights are present elsewhere, you can adjust the `dino_weights_dir` parameter in the `config.json`.

Place the `mvtec_ad_2` dataset into the `./data/` directory.
If the dataset is stored elsewhere, you can adjust the `datasets_dir` parameter in the `config.json`.

### 2. Install
Run the following to install all dependencies:
```bash
uv sync --all-packages
```

### 3. Run
Train all models by running:
```bash
uv run train-industrial
```
To test models and generate anomaly maps and thresholded anomaly maps, run:
```bash
uv run test-industrial
```
To create the final submission, run:
```bash
uv run submit-industrial
```

## Configuration
The `config.json` file contains all relevant configuration parameters. Make sure to run train, test and submission after
any change.

## Dependencies
For a full list of all dependencies, please refer to the `./pyproject.toml`, `./utils/pyproject.toml`,
and `./tracks/industrial/pyproject.toml` files. Additionally, the execution depends on the provided DINOv3 model weights.

## Licensing
For licensing information see the `LICENSE` and `NOTICE` files.
