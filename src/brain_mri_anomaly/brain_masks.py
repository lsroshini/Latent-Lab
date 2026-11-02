"""HD-BET adapter with workspace-local weights and sequential inference."""
from pathlib import Path
import nibabel as nib
import numpy as np


def validate_mask(image, mask_image):
    if image.shape != mask_image.shape or not np.allclose(image.affine, mask_image.affine, atol=1e-4):
        raise ValueError("Brain mask/image geometry mismatch")
    values = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("Brain mask must be finite and binary")
    if not values.any() or values.all():
        raise ValueError("Degenerate brain mask")
    return values.astype(bool)


class BrainExtractor:
    def __init__(self, model_dir, device="cuda", threads=4):
        import HD_BET.paths
        HD_BET.paths.folder_with_parameter_files = str(Path(model_dir).resolve())
        if not (Path(model_dir) / "fold_all/checkpoint_final.pth").is_file():
            raise FileNotFoundError("HD-BET weights missing")
        import torch
        from HD_BET.hd_bet_prediction import get_hdbet_predictor
        self.predictor = get_hdbet_predictor(use_tta=False, device=torch.device(device), verbose=False)
        # Keep full-volume logits/accumulation off the 4 GB GPU. Sliding-window
        # network execution still uses CUDA; this setting changes memory placement.
        self.predictor.perform_everything_on_device = False
        torch.set_num_threads(threads)

    def extract(self, image_path, mask_path):
        image = nib.load(image_path)
        if len(image.shape) != 3 or nib.aff2axcodes(image.affine) != ("L", "A", "S"):
            raise ValueError("HD-BET batch adapter requires audited 3D LAS images")
        mask_path = Path(mask_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not mask_path.exists():
            self.predictor.predict_from_files_sequential([[str(Path(image_path).resolve())]],
                [str(mask_path.resolve())], save_probabilities=False, overwrite=False)
        return validate_mask(image, nib.load(mask_path))

    def extract_many(self,rows,output_dir,workers=2):
        outputs=[]
        for row in rows:
            image=nib.load(row["image_path"])
            if nib.aff2axcodes(image.affine)!=("L","A","S"):
                raise ValueError("Parallel extraction requires audited LAS inputs")
            outputs.append(str((Path(output_dir)/(row["scan_id"]+".nii.gz")).resolve()))
        self.predictor.predict_from_files([[r["image_path"]] for r in rows],outputs,
            save_probabilities=False,overwrite=False,num_processes_preprocessing=workers,
            num_processes_segmentation_export=workers)
