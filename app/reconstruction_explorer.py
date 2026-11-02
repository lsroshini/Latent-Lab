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
checkpoint=st.sidebar.selectbox('Model',checkpoints,index=default,format_func=model_label)
st.sidebar.caption(f'{len(checkpoints)} trained models · spatial models listed first')
fingerprint=sha256(checkpoint)
model,device=cached_model(str(checkpoint),fingerprint)
is_spatial=checkpoint.parent.name in ('spatial-ae','spatial-vae')
st.write('**'+model_label(checkpoint)+'**')
st.caption('Spatial bottleneck with feature skip connections.' if is_spatial else 'Vector latent bottleneck with a convolutional encoder and upsampling decoder.')

source=st.radio('MRI source',['Prepared example','Upload NIfTI volume','Upload prepared grayscale PNG'],horizontal=True)
prepared=None;images=None;brain=None;z_indices=None;input_hash=None
if source=='Prepared example':
    index_path=PREPARED_DIR/'index.json'
    if not index_path.exists():st.info('Prepared MRI examples are not installed.');st.stop()
    records=[r for r in read_json(index_path).values() if r['split'] in ('test_normal','test_abnormal')]
    if not records:st.info('Prepared examples are unavailable.');st.stop()
    row=st.selectbox('Example',records,format_func=lambda r:f"{r['scan_id']} · dataset label: {r['label']}")
    prepared=Path(row['prepared_path'])
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

key=(input_hash,fingerprint)
if st.session_state.get('reconstruction_only_key')!=key:
    progress=st.progress(0.,text='Reconstructing MRI on CPU…');started=time.perf_counter()
    with st.spinner('Reconstructing the prepared MRI…'):
        reconstruction=reconstruct_volume(model,device,images,brain,
            progress=lambda done,total:progress.progress(done/total,text=f'Reconstructed {done}/{total} slices'))[0]
    progress.empty()
    st.session_state['reconstruction_only_output']=reconstruction
    st.session_state['reconstruction_only_key']=key
    st.session_state['reconstruction_only_seconds']=time.perf_counter()-started
reconstruction=st.session_state['reconstruction_only_output']
st.caption(f"Reconstructed all {len(images)} prepared slices in {st.session_state['reconstruction_only_seconds']:.1f} seconds on CPU.")
slice_index=st.slider('Prepared axial slice',0,len(images)-1,len(images)//2,key='reconstruction_slice_'+input_hash[:12]) if len(images)>1 else 0
left,right=st.columns(2)
left.image(np.clip(images[slice_index],0,1),caption='Original prepared MRI',width='stretch')
right.image(np.clip(reconstruction[slice_index],0,1),caption=model_label(checkpoint)+' reconstruction',width='stretch')
st.caption('The image and slice stay the same when you select another model. A reconstruction can preserve or reproduce an abnormality.')

with st.expander('Model details'):
    st.json(asdict(model.config))
buffer=io.BytesIO()
np.savez_compressed(buffer,image=images[slice_index],reconstruction=reconstruction[slice_index],
    prepared_grid_z_index=int(z_indices[slice_index]) if z_indices is not None else -1,
    model_name=checkpoint.parent.name,checkpoint_sha256=fingerprint)
st.download_button('Download selected slice and reconstruction',buffer.getvalue(),'mri_reconstruction.npz','application/octet-stream')
