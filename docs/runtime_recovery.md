# Current input and runtime limits

The local app uses the official CPU PyTorch environment in `.venv-cpu`.
Windows Smart App Control remains enabled. The original CUDA runtime is not
required by this snapshot.

Prepared MRI examples, matching prepared grayscale PNG inputs in the
reconstruction explorer, and previously cached NIfTI preparations are supported.
Fresh raw NIfTI brain extraction is blocked on the current laptop because
Windows rejects a compiled HD-BET dependency. A cached preparation is checked
before extraction; a known-unavailable runtime fails with an explanation.
This restriction is not a reported reconstruction or classification failure.

Do not disable protections or patch rejected vendor binaries. Repairing fresh
uploads requires an independently verified, compatible official dependency
installation followed by end-to-end preprocessing checks. No such repair is
claimed here. Use prepared inputs for the current demonstration.

Data and artifacts remain on the local computer. Prepared coordinates differ
from native MRI coordinates; this application does not export a native-space
clinical segmentation. See `PROJECT_REPORT.md` for the preparation contract
and expected local artifact directories.
