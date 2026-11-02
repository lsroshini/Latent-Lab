# Brain MRI reconstruction and anomaly analysis

Current application and development record, 2 November 2026.

## 1. Objective and current outcome

The original question was whether a convolutional variational autoencoder,
trained only on healthy brain MRI, could reconstruct familiar normal anatomy
and produce larger errors over abnormal tissue. A conventional autoencoder was
the necessary control. Reconstruction-based anomaly analysis remains the basis
of this project.

Healthy-only training does not guarantee healthy replacement of a lesion. An
AE or VAE can reconstruct abnormal tissue, and normal anatomical variation can
produce errors. Sharper reconstruction alone cannot establish anomaly removal.

The delivered app has two explicit functions:

1. **Reconstruction explorer:** original MRI and selected AE/VAE reconstruction
   only. It includes spatial AE/VAE plus the earlier baseline, denoising,
   larger-latent and MSE models. It displays no anomaly scores, thresholds,
   heatmaps, overlays or classification. The spatial AE is the default.
   Comparison mode displays up to three selected models beside the original,
   sharing the same MRI slice and intensity scale. Completed outputs are reused
   when changing slices or adding another model.
2. **Supervised AE comparison:** a separate normal/abnormal decision stage using
   fixed AE outputs, with image-only and reconstruction-only controls.

The second function was added after explicit authorization to use labeled
development data. Reconstruction networks stayed frozen and healthy-trained.
The classifier does not change an MRI reconstruction. Its decision is supervised;
the whole combined system must not be described as purely unsupervised.

The best demonstrated classification setting uses two AEs, not a VAE. The VAE
remains implemented and inspectable in the reconstruction explorer. We did not
assume or force VAE superiority.

## 2. Data and patient roles

Healthy images: **IXI T2**. Of 578 acquired scans, one coarse-spacing scan was
quarantined, leaving 577 eligible scans. Abnormal images: **BraTS 2023 Adult
Glioma**, 1,251 scans corresponding to 1,133 conservative filename-derived
patient groups. These group identities are not independently verified aliases.

Original healthy roles were 406 training, 57 validation, 57 calibration, and
57 normal-test scans. All abnormal scans were kept out of reconstruction-network
training. Healthy validation selected network checkpoints. Patient groups were
kept together when later decision-stage partitions were assigned.

Later exploratory comparisons split previously examined test cases into
263 development cases (12 healthy, 251 abnormal) and 1,045 benchmark cases
(45 healthy, 1,000 abnormal). Reassigning a previously examined patient to a new
role does not make that patient an independent test.

The supervised readout pool comprised those 263 development scans plus the
57 original healthy AE-validation scans. Its component-specific roles were
238 fitting, 82 head-validation, 57 healthy-calibration, and 1,045 benchmark
scans. Original reconstruction-network training patients did not fit these
readouts. Some AE-validation patients had already influenced AE early stopping;
this is disclosed rather than treated as an untouched representation-learning
validation cohort.

**Central limitation:** every included IXI scan has a dataset-normal label and
every included BraTS scan has a glioma label. The experiment cannot fully
separate pathological differences from dataset/acquisition differences. Only
adult glioma abnormalities were evaluated. Stronger claims require previously
unseen, acquisition-matched data containing both labels.

## 3. MRI preparation

The current preparation contract is
`rigid-t2-v3-bounded-singlethread-192-1p5mm`.

- Structural T2 grayscale MRI; no arbitrary screenshot inputs.
- Healthy MRI uses brain extraction; BraTS inputs are already skull-stripped.
- Brain intensities are clipped and normalized using the volume's first and
  99th brain percentiles. Background is zeroed for prepared input.
- Bounded rigid registration aligns MRI to a healthy-training reference.
  The reference grid is 192 x 192 x 160 at 1.5 mm spacing in RAS orientation.
- Image resampling uses linear interpolation; masks use nearest-neighbor
  interpolation. Real lesion masks never fit a registration transform.
- Retained axial slices are saved as float32 images with brain masks, original
  prepared-grid slice indices, and evaluation masks where available.

One explicitly reviewed small-brain alignment exception was retained. No
abnormal case was silently removed to improve metrics. Full-volume inference
means every retained prepared slice, not every slice of the native MRI grid.
The current laptop's fresh HD-BET preparation restriction is separate from
prepared/cached inference, which works.

## 4. Models, loss, and training decisions

### Vector AE/VAE

`models.py` defines four stride-two convolutional encoder stages with GroupNorm
and SiLU, a configurable vector latent bridge, and a bilinear-upsampling
convolutional decoder. The principal denoising comparison uses latent dimension
64. A VAE produces mean and log-variance and samples with the reparameterization
trick during training. Evaluation uses the latent mean for deterministic output.
The conventional AE uses a deterministic latent bridge.

Training minimized brain-masked L1 reconstruction error plus a smaller
background term (coefficient 0.05). The VAE additionally used KL divergence
averaged over batch and latent elements, with beta 0.001 and five-epoch warmup.
Beta depends on this reduction convention; it is not interchangeable with a
sum-reduced implementation's beta.

The original training settings were AdamW, learning rate 0.0003, weight decay
0.00001, batch size 32, maximum 40 epochs, early-stopping patience eight,
gradient clipping at five, and seed 42. Learning rate reduced on validation
plateaus. Best and resumable last checkpoints, optimizer/RNG state, validation
metrics, and source/data fingerprints were saved. Denoising supplied coarse
16 x 16 Gaussian noise, upsampled into brain tissue, with standard deviation
0.2; the target was the unchanged clean healthy image.

### Spatial AE/VAE

`restoration.py` defines the later spatial bridge model. It uses channels
8, 16, 32, 64, a 24 x 24 x 8 latent grid for 192 x 192 input, and feature skip
connections during upsampling. It does not directly add raw input to output.
Skip features can nevertheless transmit lesions. The VAE version has spatial
mean/log-variance maps and reparameterization.

The paired spatial pilot used healthy-only denoising, 406 training patients
with eight sampled slices each, 57 validation patients with 16 slices each,
batch size 16, AdamW at 0.0003, weight decay 0.00001, maximum 12 epochs,
patience four, and the same beta/warmup convention. These are historical
training procedures; the serving app does not train or search parameters.

## 5. Reconstruction scoring and decision stage

For normalized input x and reconstruction r, the raw residual is |x-r|.
Brain masking removes background from anomaly scoring in the supervised path.
The reconstruction explorer displays none of these scores or maps. A slice score is the
mean of its highest 1% of brain residuals; the volume score is the mean of
its highest 5% of slice scores. Calibration thresholds come from held-out
healthy patients, and are tied to checkpoint/preprocessing fingerprints.

Structural scoring uses an SSIM-derived difference map. A geometric mean
combines intensity and structural errors; brain-normalized spatial smoothing
was evaluated during research. Connected-component filtering was a historical
binary-mask experiment; it is not included in the current reconstruction-only
explorer or in the unchanged supervised page.

The supervised page computes two unchanged reconstructions: spatial AE and
the earlier denoising AE. Its reconstruction features are:

1. Spatial AE full-volume raw residual score.
2. Earlier AE full-volume raw residual score.
3. Earlier AE full-volume structural residual score.
4. Spatial AE mean absolute error across all brain voxels.

Eleven image features summarize brain intensity distribution and in-plane
gradient magnitude. The combined classifier uses these 11 plus the four
reconstruction features. Patient IDs, labels, filenames, and site names are
not classifier features. Image statistics can still encode acquisition source.

StandardScaler and regularized logistic regression form each readout. Equal
patient weight within each class and equal total class weight limit repeat-visit
and class-count effects. Three-fold grouped cross-validation on fitting patients
compared C = 0.1, 1, 10. The combined, image-only, and reconstruction-only heads
selected C = 10; the geometry-only nuisance control selected C = 0.1. Final
heads, feature sets, and thresholds froze before benchmark prediction.

Four fixed operating settings are available: validation-balanced, healthy95,
validation false-alarm cap, and validation sensitivity target. The default
combined/balanced setting was predeclared, not selected for winning the benchmark.
The flag uses the full-precision strict comparison score > threshold.
Classifier scores are not calibrated disease probabilities. Moving the slice
slider or changing reconstruction view does not change the volume decision.

## 6. What was tried and why the current method was retained

These are historical evaluations with different case selections; rows are not
all directly comparable. No additional experiments are running in this snapshot.

| Stage | Observation | Consequence |
|---|---|---|
| Healthy-only baseline AE and VAE | Original full-test image AUROC about 0.5506 and 0.5449 | Plain residual ranking was weak; VAE did not establish an advantage. |
| Larger vector latent, dimension 256 | Healthy reconstruction improved, but image AUROC fell to about 0.5146 and 0.5019 | Better reconstruction alone was not a detection improvement. |
| MSE reconstruction alternative | Healthy MAE worsened in the trial | Not advanced to abnormal evaluation. |
| Coarse-noise denoising, latent 64 | Original full-test image AUROC improved to 0.6581 AE and 0.6453 VAE | Retained denoising as a stronger reconstruction basis. |
| Structural/fused residual maps and healthy-fitted latent one-class control | On the reused 1,045-scan benchmark, selected AE/VAE image AUROC was 0.8403/0.8248; latent scoring was not the retained method | Kept structural scoring as an optional reconstruction-based branch. |
| Minimum 3D component-size filtering | Mean Dice rose to about 0.4065 AE and 0.4035 VAE; small-lesion cases could worsen | It was optional and off by default in the earlier explorer; it is omitted from the current UI. No AUROC gain was claimed. |
| Paired spatial denoising AE/VAE | Healthy validation MAE/SSIM improved substantially; AE did better in a matched 16-slice classification comparison | Kept spatial reconstruction available; sharper images did not prove lesion removal. |
| Full-volume threshold tradeoff study | Spatial AE AUROC 0.7818 versus structural AE 0.8403; higher detection led to many healthy false alarms | Sampled-slice advantage did not establish full-volume superiority. |
| Separately supervised readouts | Combined features outperformed image-only controls on the reused cohort | Added optional supervised page; retained source-confounding warnings and unchanged reconstructions. |

All experiments, including failures and source/data fingerprints, remain in the
preserved local research workspace. Their runners and raw journals are omitted
from this current-code snapshot to keep its implementation explainable.

## 7. Current measured results

At the frozen validation-balanced setting, on 1,000 glioma and 45 dataset-normal
benchmark scans:

| Readout | Scan AUROC | Abnormal detected | Abnormal missed | Healthy falsely flagged | Healthy correctly accepted |
|---|---:|---:|---:|---:|---:|
| Geometry-only nuisance control | 0.6557 | 718 | 282 | 21 | 24 |
| Image-only control | 0.9529 | 904 | 96 | 11 | 34 |
| Reconstruction-only classifier | 0.9807 | 945 | 55 | 3 | 42 |
| Image + reconstruction classifier | **0.9902** | **952** | **48** | **3** | **42** |

The combined setting has 95.2% scan sensitivity, 93.3% specificity, and about
94.27% balanced accuracy. AUROC 0.9902 is not 99.02% classification accuracy.
Combined minus image-only patient-group AUROC was +0.0330, with an exploratory
paired 95% bootstrap interval of approximately +0.0160 to +0.0529. This interval
is conditional on the fitted models and this cohort; it does not remove source
confounding or establish performance on new populations.

Healthy spatial-AE validation MAE improved from about 0.09012 to 0.04836 and SSIM
from 0.34697 to 0.89831 relative to the matched older AE. The spatial VAE improved
MAE from 0.09038 to 0.05755 and SSIM from 0.33171 to 0.84588. On abnormal MRI,
nonlesion-brain MAE also improved in the sampled-slice comparison. These measure
anatomical reconstruction quality, not a known healthy counterfactual inside
lesions. Partial lesion reproduction, anatomical false positives, background
artifacts, and missed abnormal scans remain observed limitations.

## 8. Code walkthrough

Start with the page the audience is viewing, then follow its imported modules.

| File | Role in the current project |
|---|---|
| `app/streamlit_app.py` | Routes between the two existing pages. |
| `app/reconstruction_explorer.py` | All ten trained-model choices, prepared/upload inputs, original/reconstruction comparison and slice download. |
| `app/supervised_comparison.py` | Two fixed AE reconstructions, four readouts, frozen settings, benchmark context and exports. |
| `models.py` | Vector AE/VAE encoder, latent bridge, decoder and reconstruction/KL loss definition. |
| `restoration.py` | Spatial AE/VAE architecture and loading the trained spatial AE. |
| `inference.py` | Checkpoint loading, deterministic batched reconstruction, residual aggregation, calibration binding and upload/cache handling. |
| `supervised_head.py` | Feature definitions, grouped fitting weights, small classifier definition and prediction. |
| `supervised_inference.py` | Loads fixed readouts, validates runtime/artifact hashes, obtains both reconstructions and applies frozen decisions. |
| `hybrid.py` | Structural/fused score maps and shared scoring/statistical utilities in the evaluated module. |
| `preprocessing.py` | Normalization, rigid registration and prepared coordinate contract. |
| `brain_masks.py` | HD-BET integration for brain extraction where its runtime is available. |
| `visualization.py` | Five image panels and saved comparison figures. |
| `common.py` | Project paths, JSON, hashing and seed utilities. |
| `scripts/predict_supervised_head.py` | Command-line full-volume supervised readout and both reconstructions. |
| `scripts/bootstrap.py` | Adds the local package to command-line imports. |
| `tests/` | Tests for these architectures, runtime computations, artifact checks and upload/cache behavior. |

Package filenames in the table are under `src/brain_mri_anomaly/` unless a
different directory is shown. A few fitting, loss, and measurement helpers share
the evaluated model/scoring modules; these are not alternative training runners
or extra app workflows. Frozen numerical modules were retained byte-for-byte.

Explain the VAE path as encoder -> mean/log-variance -> deterministic mean at
evaluation -> decoder -> residual map. Explain the supervised path separately as
input -> two frozen AEs -> 15 feature values -> scaler/logistic regression ->
frozen threshold. The supervised flag does not identify which pixels caused it.

## 9. Local trained artifacts and verification scope

Git contains code and this account of results, not MRI or trained models.
The explorer needs `runs/models/<model>/best.pt` for vector models and
`runs/spatial-restoration-v1/models/<spatial-model>/best.pt` for spatial models.
It does not require detection calibration because it only shows reconstructions.
Prepared examples use `data/processed/prepared_v3/index.json` and its referenced
prepared volumes. Registration also needs the matching prepared reference.

The supervised page needs `runs/spatial-restoration-v1/models/spatial-ae/best.pt`,
`runs/models/ae-denoising64/best.pt` with its calibration, and the completed
`runs/supervised-head-v1/` status, protocol, model manifest, selection, benchmark
results, and four `models/*.joblib` files. The explorer no longer exposes structural/mask options. These directories are not
invented or regenerated when the app starts.

Prediction still validates the fixed checkpoint/classifier hashes, selection
bindings, preprocessing version and the five evaluated numerical source files
used by the supervised prediction path. Historical training/evaluation runner
hashes remain in original protocol metadata as historical information. Those
omitted programs are not required to serve the current model, and this snapshot
does not claim to rerun the complete historical research workflow.

## Why the two pages previously looked different

The old explorer served only vector-bottleneck checkpoints. The supervised page
defaulted to a newer spatial AE with skip features, so the model architecture,
not classifier fitting, caused the reconstruction-quality difference. Both pages
now offer that same spatial AE output for the same prepared input. The explorer
also offers spatial VAE and the eight earlier vector models for comparison.
Switching the supervised classifier or threshold never improves a reconstruction.

The clean distribution was checked separately from the preserved research
workspace: all 17 focused unit tests passed. All ten reconstruction-model
choices rendered without errors on real prepared MRI. Spatial AE reconstruction
matched across the two pages. Saved features and all six classifier/control
scores matched on healthy acceptance, detected glioma, healthy false-alarm and
missed-glioma cases; the new explorer displayed no decision metrics.
These checks verify packaging and behavior, not new MRI performance claims.
No numerical method, model weight or operating threshold is
retuned during packaging. Independent, acquisition-matched validation remains
the next scientific requirement.

## 10. References and claim boundaries

Sources: [IXI](https://brain-development.org/ixi-dataset/),
[BraTS Adult Glioma access](https://www.synapse.org/Synapse:syn51514132).
Related reconstruction research includes
[Kascenas et al., 2022](https://proceedings.mlr.press/v172/kascenas22a.html) and
[a 2023 reconstruction/classifier study](https://arxiv.org/abs/2307.07534).
This implementation is not claimed to reproduce those papers or introduce
the general AE/VAE anomaly-detection concept. The contribution is the implemented
system, evaluated comparisons, explicit controls, and honest development record.

## Automatic raster-image reconstruction previews

The reconstruction explorer now accepts PNG/JPEG without manual formatting.
`image_input.py` reads image orientation, composites transparency over black,
converts RGB to grayscale (retaining grayscale 16-bit precision), normalizes
using image percentiles, and fits the image within 192 x 192 with symmetric
padding. The nonzero-content mask is not a validated brain segmentation.
Screenshots are not registered to the training reference, so this preview does
not establish cohort-comparable reconstruction performance. The advanced
prepared-PNG path retains its original normalization, and the NIfTI path and
supervised page are unchanged. Downloads mark screenshot previews explicitly.
All 20 focused tests passed, with a synthetic color-JPEG upload checked through
both single-model and comparison views. These are functional checks, not new
MRI accuracy results.
