"""Compare trained AE/VAE reconstructions without anomaly decisions or maps."""
import hashlib
import io
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import streamlit as st
import torch
from brain_mri_anomaly.common import ROOT,PREPARED_DIR,read_json,sha256
from brain_mri_anomaly.inference import load_model,reconstruct_volume,prepare_nifti_bytes,cached_nifti_preparation
from brain_mri_anomaly.restoration import load_restoration
from brain_mri_anomaly.image_input import prepare_image_preview

st.set_page_config(page_title='Brain MRI reconstruction explorer',layout='wide')
st.title('Brain MRI reconstruction explorer')
st.caption('Compare how the trained AE and VAE models reconstruct the same MRI.')
torch.set_num_threads(2)


def model_label(path):
    labels={
        'spatial-ae':'Spatial AE · sharper reconstruction',
        'spatial-vae':'Spatial VAE',
        'ae-baseline':'AE · baseline',
        'vae-baseline':'VAE · baseline',
        'ae-denoising64':'AE · denoising, latent 64',
        'vae-denoising64':'VAE · denoising, latent 64',
    }
    return labels.get(path.parent.name,path.parent.name)


@st.cache_resource(show_spinner=False)
def cached_model(path,fingerprint):
    checkpoint=Path(path)
    if checkpoint.parent.name in ('spatial-ae','spatial-vae'):
        return load_restoration(checkpoint),'cpu'
    return load_model(checkpoint,device='cpu')


checkpoints=sorted((ROOT/'runs/spatial-restoration-v1/models').glob('*/best.pt'))
checkpoints+=sorted((ROOT/'runs/models').glob('*/best.pt'))
if not checkpoints:
    st.info('Install the trained model artifacts to compare reconstructions. See README.md.');st.stop()
default=next((i for i,p in enumerate(checkpoints) if p.parent.name=='spatial-ae'),0)
comparison=st.sidebar.radio('View',['Single model','Compare models'])=='Compare models'
if comparison:
    selected=st.sidebar.multiselect('Models to compare',checkpoints,default=checkpoints[:2],
        format_func=model_label,max_selections=3)
    if not selected:st.info('Select at least one model to compare.');st.stop()
else:
    selected=[st.sidebar.selectbox('Model',checkpoints,index=default,format_func=model_label)]
st.sidebar.caption(f'{len(checkpoints)} trained models · spatial models listed first')
st.write('**'+('Model comparison' if comparison else model_label(selected[0]))+'**')
st.caption('All selected models use the same input image, slice and intensity scale.')

source=st.radio('MRI source',['Prepared example','Upload MRI image','Upload NIfTI volume','Upload prepared grayscale PNG'],horizontal=True)
prepared=None;images=None;brain=None;z_indices=None;input_hash=None
if source=='Prepared example':
    index_path=PREPARED_DIR/'index.json'
    if not index_path.exists():st.info('Prepared MRI examples are not installed.');st.stop()
    records=[r for r in read_json(index_path).values() if r['split'] in ('test_normal','test_abnormal')]
    if not records:st.info('Prepared examples are unavailable.');st.stop()
    row=st.selectbox('Example',records,format_func=lambda r:f"{r['scan_id']} · dataset label: {r['label']}")
    prepared=Path(row['prepared_path'])
elif source=='Upload MRI image':
    st.write('Upload a PNG or JPG. Grayscale conversion, intensity scaling and resizing happen automatically; image proportions are preserved with padding.')
    st.caption('This is a single-image reconstruction preview. Screenshot conversion does not perform the anatomical registration or full-volume preparation used during training.')
    upload=st.file_uploader('MRI image',type=['png','jpg','jpeg'])
    if upload:
        try:
            raw=upload.getvalue();images,brain,conversion=prepare_image_preview(raw)
            input_hash=hashlib.sha256(b'image-preview-v1'+raw).hexdigest()
            st.caption(f"Converted {conversion['original_size'][0]} × {conversion['original_size'][1]} image to grayscale, fitted within 192 × 192.")
        except ValueError as error:st.error(str(error));st.stop()
elif source=='Upload NIfTI volume':
    st.write('Use a structural T2 brain MRI. Previously prepared volumes can be reused from the local cache.')
    capability=ROOT/'runs/runtime-capabilities/raw-upload.json'
    if capability.exists() and read_json(capability).get('status')=='unavailable':
        st.info('Fresh NIfTI preparation is unavailable on this laptop because a brain-extraction dependency is blocked. Prepared examples and cached volumes remain available.')
    upload=st.file_uploader('MRI volume',type=['nii','gz'])
    confirmed=st.checkbox('This is a T2-weighted structural brain MRI')
    cached=cached_nifti_preparation(upload.getvalue()) if upload else None
    if upload and confirmed and st.button('Load cached preparation' if cached else 'Prepare MRI'):
        try:
            with st.spinner('Preparing MRI locally…'):
                st.session_state['reconstruction_upload']=str(prepare_nifti_bytes(upload.getvalue(),upload.name,device='cpu'))
                st.session_state['reconstruction_upload_hash']=hashlib.sha256(upload.getvalue()).hexdigest()
        except Exception as error:st.error(f'Preparation failed: {error}')
    if upload and st.session_state.get('reconstruction_upload_hash')==hashlib.sha256(upload.getvalue()).hexdigest():
        prepared=Path(st.session_state['reconstruction_upload'])
else:
    st.write('Use a grayscale 192 × 192 PNG exported with the same alignment and volume intensity normalization.')
    upload=st.file_uploader('Prepared slice',type=['png'])
    if upload:
        import cv2
        from scipy.ndimage import binary_fill_holes
        values=cv2.imdecode(np.frombuffer(upload.getvalue(),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        if values is None or values.shape!=(192,192) or values.dtype not in (np.uint8,np.uint16):
            st.error('Expected a grayscale 192 × 192 PNG with 8-bit or 16-bit values.');st.stop()
        if st.checkbox('This slice used the same alignment and volume normalization'):
            images=(values/np.iinfo(values.dtype).max).astype(np.float32)[None]
            brain=binary_fill_holes(values>0).astype(np.uint8)[None]
            input_hash=hashlib.sha256(upload.getvalue()).hexdigest()

if prepared:
    with np.load(prepared,allow_pickle=False) as volume:
        images=volume['image'];brain=volume['brain'];z_indices=volume['z_indices']
    input_hash=sha256(prepared)
if images is None:st.stop()
if images.ndim!=3 or images.shape[1:]!=(192,192) or brain.shape!=images.shape or not np.isfinite(images).all() or not brain.reshape(len(brain),-1).any(axis=1).all():
    st.error('Expected finite prepared MRI slices with matching nonempty brain masks.');st.stop()

if st.session_state.get('comparison_input')!=input_hash:
    st.session_state['comparison_input']=input_hash
    st.session_state['comparison_outputs']={}
outputs=st.session_state['comparison_outputs']
fingerprints={path:sha256(path) for path in selected}
for previous in list(outputs):
    if previous not in fingerprints.values():del outputs[previous]
for checkpoint,fingerprint in fingerprints.items():
    if fingerprint not in outputs:
        model,device=cached_model(str(checkpoint),fingerprint)
        progress=st.progress(0.,text='Reconstructing '+model_label(checkpoint)+' on CPU…');started=time.perf_counter()
        with st.spinner('Reconstructing '+model_label(checkpoint)+'…'):
            reconstruction=reconstruct_volume(model,device,images,brain,
                progress=lambda done,total:progress.progress(done/total,text=f'Reconstructed {done}/{total} slices'))[0]
        progress.empty()
        outputs[fingerprint]={'image':reconstruction,'seconds':time.perf_counter()-started,'config':asdict(model.config)}
if not comparison:
    fingerprint=fingerprints[selected[0]]
    st.session_state['reconstruction_only_output']=outputs[fingerprint]['image']
    st.session_state['reconstruction_only_key']=(input_hash,fingerprint)
    st.session_state['reconstruction_only_seconds']=outputs[fingerprint]['seconds']
st.caption(f'All {len(images)} input slices reconstructed on CPU. Slice changes reuse the existing outputs.')
slice_index=st.slider('Prepared axial slice',0,len(images)-1,len(images)//2,key='reconstruction_slice_'+input_hash[:12]) if len(images)>1 else 0
columns=st.columns(1+len(selected))
columns[0].image(np.clip(images[slice_index],0,1),caption='Automatically converted grayscale input' if source=='Upload MRI image' else 'Original prepared MRI',width='stretch')
for column,checkpoint in zip(columns[1:],selected):
    column.image(np.clip(outputs[fingerprints[checkpoint]]['image'][slice_index],0,1),caption=model_label(checkpoint),width='stretch')
st.caption('The image and slice stay the same when you select another model. A reconstruction can preserve or reproduce an abnormality.')

with st.expander('Model details'):
    st.json({path.parent.name:outputs[fingerprints[path]]['config'] for path in selected})
buffer=io.BytesIO()
exported={'reconstruction_'+path.parent.name:outputs[fingerprints[path]]['image'][slice_index] for path in selected}
if not comparison:exported['reconstruction']=outputs[fingerprints[selected[0]]]['image'][slice_index]
np.savez_compressed(buffer,image=images[slice_index],**exported,
    input_mode=source,screenshot_preview=source=='Upload MRI image',
    prepared_grid_z_index=int(z_indices[slice_index]) if z_indices is not None else -1,
    model_names=np.array([path.parent.name for path in selected]),checkpoint_sha256=np.array(list(fingerprints.values())))
st.download_button('Download selected slice and reconstructions',buffer.getvalue(),'mri_reconstruction.npz','application/octet-stream')
