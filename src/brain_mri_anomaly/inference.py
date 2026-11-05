from pathlib import Path
import numpy as np
import torch
from .common import ROOT,PREPARED_DIR,read_json,sha256
from .models import ReconstructionModel


def load_model(checkpoint,device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload=torch.load(checkpoint,map_location=device,weights_only=True)
    model=ReconstructionModel(payload["model_config"]).to(device)
    model.load_state_dict(payload["model"]);model.eval()
    return model,device


def reconstruct_volume(model,device,images,brain,batch_size=8,progress=None):
    reconstructed=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            tensor=torch.from_numpy(np.array(images[start:start+batch_size,None],dtype=np.float32)).to(device)
            result,_,_=model(tensor,sample=False)
            reconstructed.append(result[:,0].cpu().numpy())
            if progress is not None:progress(min(start+batch_size,len(images)),len(images))
    reconstruction=np.concatenate(reconstructed)
    difference=np.abs(images-reconstruction)
    anomaly=difference*brain
    scores=[]
    for residual,mask in zip(anomaly,brain):
        values=residual[mask>0]
        k=max(1,int(np.ceil(.01*len(values))))
        scores.append(float(np.partition(values,len(values)-k)[-k:].mean()))
    # Fixed before abnormal evaluation: mean of highest 5% of slice scores.
    k=max(1,int(np.ceil(.05*len(scores))))
    volume_score=float(np.sort(scores)[-k:].mean())
    return reconstruction,difference,anomaly,np.array(scores),volume_score


def calibration_for(checkpoint):
    path=Path(checkpoint).parent/"calibration.json"
    result=read_json(path)
    if result["checkpoint_sha256"]!=sha256(checkpoint):
        raise ValueError("Calibration belongs to a different checkpoint")
    from .preprocessing import VERSION
    if result["preprocessing_version"]!=VERSION:
        raise ValueError("Checkpoint calibration requires a different preprocessing version")
    return result


def lossless_las_source(source,directory,image):
    """Preserve already-LAS bytes; otherwise permute/flip into float32 storage."""
    import nibabel as nib
    if nib.aff2axcodes(image.affine)==("L","A","S"):
        return source
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),nib.orientations.axcodes2ornt(("L","A","S")))
    oriented=image.as_reoriented(transform)
    header=oriented.header.copy();header.set_data_dtype(np.float32)
    lossless=nib.Nifti1Image(oriented.get_fdata(dtype=np.float32),oriented.affine,header)
    las=directory/"input_las.nii.gz";nib.save(lossless,las)
    return las


def cached_nifti_preparation(raw):
    import hashlib
    from .preprocessing import VERSION
    directory=ROOT/".cache/uploads"/hashlib.sha256(raw).hexdigest()/VERSION/"lossless-las-v2"
    path=directory/"prepared.npz"
    return path if path.exists() and (directory/"prepared.json").exists() else None


def prepare_nifti_bytes(raw,filename,device=None):
    """Shared local upload/CLI preparation, identical to the research pipeline."""
    import hashlib
    import nibabel as nib
    from .brain_masks import BrainExtractor
    from .preprocessing import prepare_volume,VERSION
    if len(raw)>200*1024**2:raise ValueError("Maximum volume size is 200 MB")
    if not filename.lower().endswith((".nii",".nii.gz")):raise ValueError("Expected a NIfTI volume")
    cached=cached_nifti_preparation(raw)
    if cached is not None:return cached
    extraction_device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    capability_path=ROOT/'runs/runtime-capabilities/raw-upload.json'
    if capability_path.exists():
        from .common import read_json
        capability=read_json(capability_path)
        if (capability.get('status')=='unavailable' and capability.get('torch_version')==str(torch.__version__)
                and capability.get('device')==extraction_device):
            raise RuntimeError('Fresh NIfTI preprocessing is unavailable: Windows Application Control blocked an HD-BET dependency. Use prepared examples or a previously cached volume; see docs/runtime_recovery.md.')
    key=hashlib.sha256(raw).hexdigest()
    directory=ROOT/".cache/uploads"/key/VERSION/"lossless-las-v2"
    directory.mkdir(parents=True,exist_ok=True)
    prepared=directory/"prepared.npz"
    if prepared.exists() and (directory/"prepared.json").exists():return prepared
    extension=".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
    source=directory/("original"+extension);source.write_bytes(raw)
    image=nib.load(source)
    if len(image.shape)!=3 or np.prod(image.shape)>80_000_000:
        raise ValueError("Expected a single 3D MRI volume with at most 80 million voxels")
    if not np.isfinite(image.get_fdata(dtype=np.float32)).all():raise ValueError("MRI contains nonfinite values")
    las=lossless_las_source(source,directory,image)
    extractor=BrainExtractor(ROOT/".cache/models/hd-bet/release_2.0.0",extraction_device)
    mask=directory/"brain_mask.nii.gz";extractor.extract(las,mask)
    row={"dataset":"upload","patient_id":key[:12],"scan_id":key[:12],"label":"unknown","image_path":str(las),"mask_path":"","split":"inference"}
    metadata=prepare_volume(row,str(mask),PREPARED_DIR/"reference",prepared)
    from .common import write_json
    write_json(directory/"prepared.json",metadata)
    return prepared
