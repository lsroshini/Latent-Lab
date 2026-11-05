"""Rigid, physical-space MRI preparation; never use lesion labels to fit transforms."""
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

SHAPE = (192, 192, 160)
SPACING = (1.5, 1.5, 1.5)
VERSION = "rigid-t2-v3-bounded-singlethread-192-1p5mm"


def normalized_image(image_path, brain_mask_path=None):
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise ValueError("Expected a single 3D MRI volume")
    data = sitk.GetArrayFromImage(image)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite image intensity")
    if brain_mask_path:
        mask_image = sitk.ReadImage(str(brain_mask_path), sitk.sitkUInt8)
        if (image.GetSize() != mask_image.GetSize() or
                not np.allclose(image.GetOrigin(), mask_image.GetOrigin(), atol=1e-4) or
                not np.allclose(image.GetDirection(), mask_image.GetDirection(), atol=1e-4) or
                not np.allclose(image.GetSpacing(), mask_image.GetSpacing(), atol=1e-4)):
            raise ValueError("Brain mask geometry mismatch")
        mask = sitk.GetArrayFromImage(mask_image) > 0
    else:
        # Only for already skull-stripped BraTS. Never use on raw full-head MRI.
        mask = binary_fill_holes(data != 0)
    if not mask.any():
        raise ValueError("Empty brain mask")
    lo, hi = map(float, np.percentile(data[mask], [1, 99]))
    if hi <= lo:
        raise ValueError("Degenerate intensity range")
    values = np.where(mask, np.clip((data - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    normal = sitk.GetImageFromArray(values); normal.CopyInformation(image)
    mask_image = sitk.GetImageFromArray(mask.astype(np.uint8)); mask_image.CopyInformation(image)
    return normal, mask_image, {"p1": lo, "p99": hi, "brain_ml": float(mask.sum()*np.prod(image.GetSpacing())/1000)}


def make_reference(image_path, brain_mask_path, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    fixed, mask, intensity = normalized_image(image_path, brain_mask_path)
    positions = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
    center_index = ((positions.min(0) + positions.max(0))/2)[::-1]
    center = np.array(fixed.TransformContinuousIndexToPhysicalPoint(center_index.tolist()))
    grid = sitk.Image(list(SHAPE), sitk.sitkFloat32)
    grid.SetSpacing(SPACING)
    direction = np.diag([-1., -1., 1.])  # NIfTI RAS axes in ITK's LPS coordinates.
    grid.SetDirection(direction.ravel().tolist())
    grid.SetOrigin((center-direction@((np.array(SHAPE)-1)*np.array(SPACING)/2)).tolist())
    identity = sitk.Transform(3, sitk.sitkIdentity)
    reference = sitk.Resample(fixed, grid, identity, sitk.sitkLinear, 0, sitk.sitkFloat32)
    reference_mask = sitk.Resample(mask, grid, identity, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    sitk.WriteImage(reference, str(output_dir/"reference.nii.gz"))
    sitk.WriteImage(reference_mask, str(output_dir/"reference_mask.nii.gz"))
    return intensity


def prepare_volume(row, brain_mask_path, reference_dir, output_path):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    moving, moving_mask, normalization = normalized_image(row["image_path"], brain_mask_path)
    fixed = sitk.ReadImage(str(Path(reference_dir)/"reference.nii.gz"), sitk.sitkFloat32)
    fixed_mask = sitk.ReadImage(str(Path(reference_dir)/"reference_mask.nii.gz"), sitk.sitkUInt8)
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(1)
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(.15, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel([4,2,1])
    registration.SetSmoothingSigmasPerLevel([2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=.5,minStep=.005,
        numberOfIterations=150,relaxationFactor=.5)
    registration.SetOptimizerScalesFromPhysicalShift()
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                               sitk.CenteredTransformInitializerFilter.MOMENTS)
    registration.SetInitialTransform(initial, inPlace=False)
    from .common import read_json
    reference_id=read_json(Path(reference_dir)/"provenance.json")["reference"]["scan_id"]
    is_reference=row["dataset"]=="ixi" and row["scan_id"]==reference_id
    registration_error=None
    try:
        transform = sitk.Transform(3,sitk.sitkIdentity) if is_reference else registration.Execute(fixed, moving)
    except RuntimeError as exc:
        registration_error=str(exc)
        transform=initial
    prepared = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0, sitk.sitkFloat32)
    brain = sitk.Resample(moving_mask, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    values = sitk.GetArrayFromImage(prepared)
    brain_values = sitk.GetArrayFromImage(brain).astype(bool)
    fixed_values = sitk.GetArrayFromImage(fixed_mask).astype(bool)
    overlap = float(2*(brain_values & fixed_values).sum() / (brain_values.sum()+fixed_values.sum()))
    retained = float(brain_values.sum()*np.prod(SPACING)/1000/normalization["brain_ml"])
    strategy="identity" if is_reference else "bounded_step_lr0p5_singlethread"
    attempts=[{"strategy":strategy,"brain_mask_dice":overlap,"brain_volume_ratio":retained,"optimizer_error":registration_error}]
    if not is_reference and (registration_error or overlap < .75 or not .9 <= retained <= 1.1):
        # Retry the same rigid problem from its original initialization using
        # bounded, decreasing optimizer steps and explicit brain metric masks.
        registration.SetMetricFixedMask(fixed_mask)
        registration.SetMetricMovingMask(moving_mask)
        registration.SetOptimizerAsRegularStepGradientDescent(learningRate=.1,minStep=.001,
            numberOfIterations=150,relaxationFactor=.5)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(initial,inPlace=False)
        transform=registration.Execute(fixed,moving)
        prepared=sitk.Resample(moving,fixed,transform,sitk.sitkLinear,0,sitk.sitkFloat32)
        brain=sitk.Resample(moving_mask,fixed,transform,sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
        values=sitk.GetArrayFromImage(prepared);brain_values=sitk.GetArrayFromImage(brain).astype(bool)
        overlap=float(2*(brain_values&fixed_values).sum()/(brain_values.sum()+fixed_values.sum()))
        retained=float(brain_values.sum()*np.prod(SPACING)/1000/normalization["brain_ml"])
        strategy="bounded_step_lr0p1_singlethread_fallback"
        attempts.append({"strategy":strategy,"brain_mask_dice":overlap,"brain_volume_ratio":retained})
    alignment_review = None
    if overlap < .75 and .9 <= retained <= 1.1:
        # Explicit diagnostic exceptions are bound to the raw image and exact
        # rejected transform. They never use lesion labels or alter registration.
        from .common import ROOT, sha256
        reviews_path = ROOT / "data/manifests/alignment_reviews.json"
        reviews = read_json(reviews_path) if reviews_path.exists() else {}
        review = reviews.get(row['dataset']+'_'+row['scan_id'])
        if review and review['image_sha256']==sha256(row['image_path']):
            current_transform = list(transform.GetParameters())
            if (np.allclose(current_transform, review['transform_parameters'],rtol=0,atol=1e-10)
                    and abs(overlap-review['measured_dice'])<1e-10
                    and review['theoretical_maximum_dice']<.75
                    and overlap/review['theoretical_maximum_dice']>=.97):
                alignment_review = review
    if (overlap < .75 and alignment_review is None) or not .9 <= retained <= 1.1:
        raise ValueError(f"Registration/coverage review required: mask Dice={overlap:.3f}, volume ratio={retained:.3f}")
    lesion = np.zeros(brain_values.shape, dtype=np.uint8)
    lesion_volume_ratio = None
    if row.get("mask_path"):
        gt = sitk.ReadImage(row["mask_path"])
        native_gt = sitk.GetArrayFromImage(gt) > 0
        warped = sitk.Resample(sitk.Cast(gt>0,sitk.sitkUInt8), fixed, transform,
                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        lesion = sitk.GetArrayFromImage(warped).astype(np.uint8)
        if native_gt.any() and not lesion.any():
            raise ValueError("Lesion vanished during resampling")
        if native_gt.any():
            lesion_volume_ratio = float(lesion.sum()*np.prod(SPACING)/(native_gt.sum()*np.prod(gt.GetSpacing())))
        outside = int(((lesion>0) & ~brain_values).sum())
        if outside > max(3, .01*lesion.sum()):
            raise ValueError("More than 1% of resampled lesion lies outside the image-derived brain mask")
    indices = np.flatnonzero(brain_values.sum(axis=(1,2)) > 16)
    if len(indices) < 20:
        raise ValueError("Insufficient brain slice coverage")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, image=np.where(brain_values[indices], values[indices], 0).astype(np.float32),
                        brain=brain_values[indices].astype(np.uint8), lesion=lesion[indices], z_indices=indices)
    temporary.replace(output_path)
    transform_path = output_path.with_suffix(".tfm")
    sitk.WriteTransform(transform, str(transform_path))
    return {**row, "prepared_path": str(output_path.resolve()), "transform_path": str(transform_path.resolve()),
            "normalization": normalization, "brain_mask_reference_dice": overlap, "brain_volume_ratio": retained,
            "lesion_volume_ratio": lesion_volume_ratio, "slice_count": len(indices), "version": VERSION,
            "registration_strategy":strategy,"registration_attempts":attempts,
            "alignment_review":alignment_review,
            "spacing": SPACING, "shape": SHAPE, "registration_metric": None if is_reference else float(registration.GetMetricValue()),
            "registration_stop": "Identity: healthy training reference" if is_reference else registration.GetOptimizerStopConditionDescription(),
            "native_shape": list(nib.load(row["image_path"]).shape)}
