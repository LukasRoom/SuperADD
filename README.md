# SuperADD

[![arXiv](https://img.shields.io/badge/arXiv-2605.14808-b31b1b.svg)](https://arxiv.org/abs/2605.14808)

This repository contains the code for the SuperADD submission to the VAND 4.0 Challenge.

SuperADD is a class-agnostic anomaly detection model and trained using only anomaly-free images.
It is based on [SuperAD](https://github.com/Summerdayhurricane/SuperAD/tree/main) by Zhang et al.

Built with DINOv3.

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
When loading it to the official benachmark server, results of test_public are not expected, in this case you can remove test_public from the settings.json file before running test-industrial or you can manually remove the test_public predictions.

## Configuration
The `config.json` file contains all relevant configuration parameters. Make sure to run train, test and submission after
any change.

## Pretrained Models
We provide “pretrained” models for each category for easier reproduction and to skip the training script. In our case, a pretrained model does not contain weights, but a memory bank with reference embeddings stored in an .npz file, along with the corresponding model settings in a .json file for that category.

Models are available at: https://owncloud.fraunhofer.de/index.php/s/THkX7W8AhRd2RCs

Place all models in the `./models/` directory and make sure you installed DINOv3 weights. After that, you can run inference directly with:
```bash
uv run test-industrial
```

## Dependencies
For a full list of all dependencies, please refer to the `./pyproject.toml`, `./utils/pyproject.toml`,
and `./tracks/industrial/pyproject.toml` files. Additionally, the execution depends on the installed DINOv3 model weights.

## Citation
To cite this work, please use:
```bibtex
@article{Roming_SuperADD_Training-free_Class-agnostic_2026,
  author = {Roming, Lukas and Lehnerer, Felix and Funk, Jonas V. and Michel, Andreas and Maier, Georg and Längle, Thomas and Beyerer, Jürgen},
  journal = {arXiv preprint arXiv:2605.14808},
  title = {{SuperADD: Training-free Class-agnostic Anomaly Segmentation -- CVPR 2026 VAND 4.0 Workshop Challenge Industrial Track}},
  url = {https://arxiv.org/abs/2605.14808},
  month = 5,
  year = {2026}
}
```

## Licensing
For licensing information see the `LICENSE` and `NOTICE` files.
