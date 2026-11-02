"""Optional supervised readout page; no claim of lesion-free reconstruction."""
import hashlib
import io
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import streamlit as st
import torch
from brain_mri_anomaly.common import ROOT,PREPARED_DIR,read_json,sha256
from brain_mri_anomaly.inference import cached_nifti_preparation
from brain_mri_anomaly.supervised_inference import load_artifacts,predict_readouts,decision,POLICY_LABELS,FAMILY_LABELS
from brain_mri_anomaly.visualization import panels

st.set_page_config(page_title='Supervised AE comparison',layout='wide')
st.title('Supervised AE comparison')
st.caption('Experimental decision readouts from frozen AE outputs · reconstruction views and classification are separate results')
st.info('The AEs were trained on healthy MRI. These decision models also learned from labeled normal/abnormal development scans. They do not modify the reconstructions or establish lesion removal.')
st.warning('Healthy and glioma scans came from different datasets. Acquisition and preprocessing can influence these scores; the previously examined benchmark is not independent clinical validation.')
torch.set_num_threads(2)

@st.cache_resource(show_spinner=False)
def readouts(runtime_version,artifact_version):return load_artifacts()

try:
    runtime_version=sha256(ROOT/'src/brain_mri_anomaly/supervised_inference.py')+sha256(ROOT/'src/brain_mri_anomaly/preprocessing.py')
    artifact_version=sha256(ROOT/'runs/supervised-head-v1/selection.json')+sha256(ROOT/'runs/supervised-head-v1/fitted_models.json')
    artifacts=readouts(runtime_version,artifact_version)
except (ValueError,FileNotFoundError) as error:st.error(str(error));st.stop()
spec=artifacts[0]
family=st.sidebar.selectbox('Decision readout',list(FAMILY_LABELS),format_func=FAMILY_LABELS.get,key='supervised_family')
policy=st.sidebar.selectbox('Operating setting',list(POLICY_LABELS),format_func=POLICY_LABELS.get,key='supervised_policy')
source=st.radio('MRI source',['Prepared example','Previously prepared NIfTI'],horizontal=True)
lesion=None
if source=='Prepared example':
    records=[r for r in read_json(PREPARED_DIR/'index.json').values() if spec['example_roles'].get(r['dataset']+'_'+r['scan_id'])=='benchmark']
    st.caption('Examples come from the historical benchmark; none fitted these supervised readouts. They were examined in earlier research, so this is not fresh independent validation.')
    record=st.selectbox('Prepared MRI',records,format_func=lambda r:f"{r['scan_id']} · dataset label: {r['label']}",key='supervised_example')
    path=Path(record['prepared_path']);input_hash=sha256(path)
else:
    st.info('Fresh NIfTI preprocessing remains blocked by the existing HD-BET dependency issue. This page can reuse a volume already prepared by the main explorer; otherwise choose a prepared example.')
    upload=st.file_uploader('Previously prepared structural T2 NIfTI',type=['nii','gz'])
    if upload is None:st.stop()
    raw=upload.getvalue();path=cached_nifti_preparation(raw)
    if path is None:st.info('No compatible preparation is cached for these bytes. Choose a prepared example or resolve the preprocessing runtime first.');st.stop()
    input_hash=hashlib.sha256(raw).hexdigest()

with np.load(path,allow_pickle=False) as volume:
    image=volume['image'];brain=volume['brain']>0;z_indices=volume['z_indices']
    if 'lesion' in volume:lesion=volume['lesion']
key=(input_hash,spec['artifact_fingerprint'])
if st.session_state.get('readout_input_key')!=key:
    progress=st.progress(0.,text='Reconstructing both frozen AEs on CPU…');started=time.perf_counter()
    with st.spinner('Computing full-volume reconstruction features…'):
        prediction=predict_readouts(image,brain,artifacts,lambda current,total:progress.progress(current/total,text=f'Reconstructing slices on CPU: {current}/{total}'))
    progress.empty();st.session_state['readout_prediction']=prediction;st.session_state['readout_input_key']=key
    st.session_state['readout_seconds']=time.perf_counter()-started
prediction=st.session_state['readout_prediction'];selected=decision(spec,prediction,family,policy)
st.caption(f"Computed all {len(image)} retained prepared slices in {st.session_state['readout_seconds']:.1f} seconds on CPU. Classifier outputs are uncalibrated scores, not disease probabilities.")
columns=st.columns(3)
columns[0].metric('Classifier score',f"{selected['score']:.4f}")
columns[1].metric('Frozen decision threshold',f"{selected['threshold']:.4f}")
columns[2].metric('Experimental result','Flagged' if selected['above_threshold'] else 'Below threshold')
st.caption('The decision summarizes the complete prepared volume. A flagged result does not establish a lesion; below threshold does not rule one out.')
measured=spec['benchmark']['methods'][family]['operating_points'][policy]['scans']
st.info(f"Same-cohort benchmark for this fixed setting: {measured['tp']}/1000 abnormal scans detected, {measured['fn']}/1000 missed, and {measured['fp']}/45 healthy scans falsely flagged. The setting was chosen on separate head-validation/calibration patients; these are exploratory outcomes.")

view=st.radio('Reconstruction view',['Sharper spatial AE','Original denoising AE'],horizontal=True)
st.caption('Sharper spatial AE is a newer architecture with a spatial bottleneck and feature skips. Original denoising AE is the earlier vector model used by the reconstruction explorer. The decision classifier changes neither output; sharper images can still reproduce lesions.')
prefix='spatial' if view=='Sharper spatial AE' else 'old'
index=st.slider('Prepared axial slice',0,len(image)-1,int(np.argmax(prediction['spatial_slice_scores'])),key='readout_slice_'+input_hash[:12])
views=panels(image[index],prediction[prefix+'_reconstruction'][index],prediction[prefix+'_difference'][index],prediction[prefix+'_anomaly'][index],.25)
for col,(name,array) in zip(st.columns(5),views.items()):col.image(np.clip(array,0,1),caption=name,width='stretch')
st.caption('The residual display uses a fixed 0.5 intensity scale. These maps show reconstruction differences; they are not classifier explanations, calibrated pixel thresholds or lesion probabilities. The decision head reads summaries from both AEs and leaves these images unchanged.')
if lesion is not None and lesion.any() and st.checkbox('Show available evaluation mask'):
    st.image(lesion[index].astype(np.uint8)*255,caption='Reference mask used only for evaluation',width=400)
with st.expander('Compare experimental decision readouts'):
    rows=[]
    for name,label in FAMILY_LABELS.items():
        d=decision(spec,prediction,name,policy);m=spec['benchmark']['methods'][name]['operating_points'][policy]['scans']
        rows.append({'Readout':label,'Current score':d['score'],'Threshold':d['threshold'],'Flagged':d['above_threshold'],
            'Benchmark abnormal detected /1000':m['tp'],'Benchmark healthy false alarms /45':m['fp']})
    st.dataframe(rows,hide_index=True,width='stretch')
buffer=io.BytesIO()
np.savez_compressed(buffer,image=image[index],spatial_reconstruction=prediction['spatial_reconstruction'][index],
    old_reconstruction=prediction['old_reconstruction'][index],spatial_difference=prediction['spatial_difference'][index],
    old_difference=prediction['old_difference'][index],spatial_anomaly=prediction['spatial_anomaly'][index],
    old_anomaly=prediction['old_anomaly'][index],prepared_grid_z_index=int(z_indices[index]),
    classifier_score=selected['score'],decision_threshold=selected['threshold'],family=family,policy=policy,
    artifact_fingerprint=spec['artifact_fingerprint'],supervised=True,scores_are_calibrated_probabilities=False)
st.download_button('Download selected slice and decision',buffer.getvalue(),'supervised_ae_slice.npz','application/octet-stream')
st.download_button('Download decision and feature details',json.dumps({**selected,'all_scores':prediction['scores'],'features':prediction['features']},indent=2),
    'supervised_ae_decision.json','application/json')
with st.expander('Research details and limitations'):
    st.json(selected);st.write(spec['scope'])
    st.write('Prepared coordinates only. Background reconstruction and partial lesion reproduction remain known limitations. No supervised result proves anatomically correct healthy tissue replacement.')
