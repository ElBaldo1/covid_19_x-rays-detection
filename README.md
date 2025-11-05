# COVID-19 Chest X-Ray Detection with CNN

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)

## Overview
This project provides an end-to-end pipeline for classifying chest X-ray images as
COVID-positive or COVID-negative using a convolutional neural network (CNN) built in
PyTorch. The repository includes reproducible training scripts, a command-line
inference tool, a Streamlit dashboard for interactive exploration, and automated
unit tests. The code base has been modernized with rich documentation, type hints,
and packaging metadata so it is ready for public release.

Key dependencies include `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`,
`matplotlib`, `seaborn`, and `streamlit`.

## Dataset
The model is trained on the [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
After downloading and extracting the archive, ensure the directory structure matches
the expected layout:

```
COVID-Data-Radiography/
├── no/
│   ├── *.png
├── yes/
│   ├── *.png
```

An optional `manual-test/` folder can contain individual images for ad-hoc
experiments. The repository validates the presence of the `no/` and `yes/`
sub-directories before running.

## Features
- Reproducible CNN training pipeline with early stopping, learning rate scheduling,
  confusion matrix plots, and classification reports.
- Modern Python packaging with `setup.py`, `pyproject.toml`, and preconfigured linting.
- Command-line inference utility that batches predictions across folders or files.
- Streamlit app for drag-and-drop exploration of predictions with probability bars.
- Automated test suite powered by `pytest` to smoke test dataset loading, inference,
  and training flows.

## Installation & Environment Setup
Create an isolated environment with Conda (recommended) and install the package in
editable mode:

```bash
conda env create -f environment.yml
conda activate covid_proj
pip install -e .
```

Alternatively, install directly via `pip`:

```bash
pip install .
```

## Usage

### Training
Run the training script to train (or fine-tune) the CNN. Training artifacts are
written to `outputs/` and the best model weights are saved to `best_model.pth`.

```bash
python covid_classification.py \
  --dataset-dir /path/to/COVID-Data-Radiography \
  --output-dir outputs \
  --epochs 20
```

### Prediction
Use the CLI to classify individual images or entire directories:

```bash
python predict.py manual-test/image1.png manual-test/image2.png
```

Sample output:

```
2024-03-16 12:00:00 - covid_xray.inference - INFO - Loaded model weights from /repo/best_model.pth
2024-03-16 12:00:00 - predict - INFO - Running inference on 2 image(s).
2024-03-16 12:00:00 - predict - INFO - image1.png -> COVID-negative (confidence: 96.42% | probs: 0.964, 0.036)
2024-03-16 12:00:00 - predict - INFO - image2.png -> COVID-positive (confidence: 88.17% | probs: 0.118, 0.882)
```

### Streamlit Demo
Launch the Streamlit dashboard to upload images and view probability charts:

```bash
streamlit run streamlit_app.py
```

Expected interface:
- Upload a PNG or JPG chest X-ray image using the file uploader.
- View the predicted label and confidence score.
- Inspect the probability bar chart contrasting the two classes.

## Model Architecture & Evaluation
The `RadiographyCNN` architecture consists of three convolutional blocks with batch
normalization and max pooling, followed by two fully connected layers with dropout.
Data augmentation includes random crops, flips, rotations, and color jittering to
improve generalization. Training uses the Adam optimizer with a `ReduceLROnPlateau`
scheduler and early stopping after three stagnant epochs.

Evaluation exports both a classification report and confusion matrix heatmap. The
model achieves the following metrics on the held-out test split:

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.93  |
| Precision  | 0.92  |
| Recall     | 0.94  |
| F1-score   | 0.93  |

## Results
Training generates the following artifacts under `outputs/`:

- `training_log.csv`: per-epoch loss, accuracy, and learning rate.
- `training_curves.png`: dual-panel plot of loss and accuracy across epochs.
- `classification_report.txt`: detailed precision/recall metrics by class.
- `cm.png`: confusion matrix visualization.

## License
This project is released under the [MIT License](LICENSE).
