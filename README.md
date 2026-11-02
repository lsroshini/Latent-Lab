# Brain MRI reconstruction and anomaly analysis

Current application comparing healthy-trained convolutional AE and VAE
reconstruction, with a separate supervised classifier based on AE outputs.
This snapshot contains the current application, its required implementation,
two command-line files, and relevant tests. Old MNIST code, experiment runners,
parameter searches, recovery scripts, and raw experiment logs are omitted.

Read [the project document](docs/PROJECT_REPORT.md) for the objective, methods
tried, selection decisions, hyperparameters, results, limitations, and a code
walkthrough. The full research workspace remains preserved locally.

## What each page does

- **Reconstruction explorer:** choose an available trained AE or VAE
  checkpoint, including spatial AE/VAE and the earlier baseline, denoising,
  larger-latent and MSE models. Inspect only the original MRI and reconstruction.
  This page has no anomaly scores, thresholds, heatmaps or classification.
- **Supervised AE comparison:** compare two fixed AE reconstructions and
  supervised normal/abnormal decisions. This page does not use a VAE.

The latest combined classifier detected 952/1,000 abnormal scans and falsely
flagged 3/45 healthy scans at its frozen balanced setting. These exploratory
results use previously examined patients, with healthy IXI and glioma BraTS
scans from different datasets. They do not establish clinical reliability or
prove that lesions disappear during reconstruction.

## Install

Python 3.10 or newer is required. The tested local application uses CPU PyTorch
and two inference threads; operating-system protections remain enabled.

```powershell
py -m venv .venv-cpu
.venv-cpu/Scripts/python.exe -m pip install --upgrade pip
.venv-cpu/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-cpu/Scripts/python.exe -m pip install -e .
.venv-cpu/Scripts/python.exe -m unittest discover -s tests -v
```

## Run with existing trained artifacts

```powershell
./launch_app.ps1
```

Alternatively:

```powershell
.venv-cpu/Scripts/python.exe -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501. To see a VAE, choose **Reconstruction explorer**
and then **Spatial VAE** or **VAE · denoising, latent 64** in the Model selector, when that checkpoint is
installed. The supervised page is at `/supervised-comparison`.

MRI images, model weights, fitted classifiers, calibration metadata, and
generated `runs/` outputs are not included in Git. A fresh clone therefore
does not contain a pretrained demonstration. This is the current inference
snapshot, not a distribution of the historical training workspace.
Required artifact locations are listed in the project document. Obtain data
through the official [IXI](https://brain-development.org/ixi-dataset/) and
[BraTS](https://www.synapse.org/Synapse:syn51514132) access routes and follow
their terms. Do not commit patient data, credentials, or model bundles here.

Prepared examples and previously cached NIfTI inputs work on the current
laptop. Fresh raw NIfTI preprocessing is still blocked by an HD-BET dependency;
see [runtime limitations](docs/runtime_recovery.md).

## Command-line prediction

```powershell
.venv-cpu/Scripts/python.exe scripts/predict_supervised_head.py --prepared PATH_TO_PREPARED_VOLUME.npz --output-dir runs/supervised-prediction
```

Predictions are research outputs. Classifier scores are not calibrated disease
probabilities, and reconstruction heatmaps are not explanations of the classifier.

Verification: 17 focused unit tests passed. Real prepared-MRI checks passed for
all ten reconstruction models, matching spatial AE outputs across pages, and
normal/abnormal success and failure cases with unchanged classifier scores.
