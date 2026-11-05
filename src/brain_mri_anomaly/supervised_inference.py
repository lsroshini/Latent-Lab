"""Inference-only experimental readouts; classification is distinct from restoration."""
import hashlib
from pathlib import Path
import joblib
import numpy as np
from .common import ROOT,read_json,sha256
from .restoration import load_restoration
from .inference import load_model,reconstruct_volume,calibration_for
from .hybrid import score_maps,volume_score
from .supervised_head import FEATURE_SETS,image_features,predict_head

POLICY_LABELS={'balanced':'Balanced (validation-selected)','healthy95':'Healthy calibration reference',
    'fpr10':'Validation false-alarm cap','sensitivity90':'Higher detection (validation target)'}
FAMILY_LABELS={'combined':'Image + reconstruction','reconstruction_only':'Reconstruction features',
    'image_only':'Image-only control','geometry_only':'Geometry-only control'}

RUNTIME_SOURCE_PATHS=tuple('src/brain_mri_anomaly/'+name+'.py' for name in
    ('supervised_head','restoration','models','inference','hybrid'))


def validate_runtime_sources(expected,root=ROOT):
    """Check the evaluated prediction code; research runners are not shipped.

    The original protocol retains historical training/evaluation hashes as
    documentation. Only these five entries execute within prediction. A missing
    runtime entry, missing runtime file or changed runtime source is an error.
    """
    for path in RUNTIME_SOURCE_PATHS:
        if path not in expected or not (root/path).is_file() or sha256(root/path)!=expected[path]:
            raise ValueError('Evaluated reconstruction/readout implementation changed: '+path)


def load_artifacts(directory=None):
    directory=Path(directory) if directory else ROOT/'runs/supervised-head-v1'
    if not (directory/'status.json').exists() or read_json(directory/'status.json')['status']!='complete':
        raise ValueError('A completed supervised readout study is required')
    protocol=read_json(directory/'protocol.json');manifest=read_json(directory/'fitted_models.json');selection=read_json(directory/'selection.json')
    results=read_json(directory/'results.json');protocol_hash=sha256(directory/'protocol.json')
    if selection['protocol_sha256']!=protocol_hash or manifest['protocol_sha256']!=protocol_hash or results['selection_sha256']!=sha256(directory/'selection.json'):
        raise ValueError('Readout artifact provenance mismatch')
    if selection['fitted_models_sha256']!=sha256(directory/'fitted_models.json'):raise ValueError('Readout model manifest changed')
    validate_runtime_sources(protocol['source_sha256'])
    checkpoints={'spatial_ae':ROOT/'runs/spatial-restoration-v1/models/spatial-ae/best.pt','old_ae':ROOT/'runs/models/ae-denoising64/best.pt'}
    for name,path in checkpoints.items():
        if sha256(path)!=protocol['checkpoints'][name]:raise ValueError('Evaluated AE checkpoint changed')
    calibration_for(checkpoints['old_ae'])  # Reject a changed preparation version.
    models={}
    for family in FEATURE_SETS:
        path=directory/'models'/(family+'.joblib')
        if sha256(path)!=manifest['models'][family]['model_sha256']:raise ValueError('Evaluated classifier contents changed')
        models[family]=joblib.load(path)
    fingerprint=hashlib.sha256((protocol_hash+sha256(directory/'selection.json')+sha256(directory/'fitted_models.json')).encode()).hexdigest()
    spec={'artifact_fingerprint':fingerprint,'checkpoints':protocol['checkpoints'],'selection':selection,'example_roles':protocol['partition_map'],
        'benchmark':results['benchmark'],'supervised':True,'scores_are_calibrated_probabilities':False,
        'scope':'Previously examined IXI-normal/BraTS-glioma cohort; source-confounded exploratory classification, not proof of lesion removal'}
    return spec,models,load_restoration(checkpoints['spatial_ae']),load_model(checkpoints['old_ae'],device='cpu')[0]


def reconstruct_features(images,brain,new_model,old_model,progress=None):
    images=np.asarray(images,dtype=np.float32);brain=np.asarray(brain)>0
    if images.ndim!=3 or images.shape!=brain.shape or images.shape[1:]!=(192,192) or len(images)<2 or not brain.reshape(len(brain),-1).any(axis=1).all():
        raise ValueError('Readouts require a complete normalized prepared 192x192 volume with a nonempty brain mask per slice')
    features=image_features(images,brain);n=len(images)
    def first(current,total):
        if progress:progress(current,2*n)
    def second(current,total):
        if progress:progress(n+current,2*n)
    new,difference,anomaly,slices,score=reconstruct_volume(new_model,'cpu',images,brain,batch_size=16,progress=first)
    old,old_difference,old_anomaly,old_slices,old_score=reconstruct_volume(old_model,'cpu',images,brain,batch_size=8,progress=second)
    features.update({'spatial_ae_l1':score,'old_ae_l1':old_score,'old_ae_structure':volume_score(score_maps(images,old,brain)['structure'],brain),
        'spatial_brain_mae':float(difference[brain].mean())})
    return {'features':features,'spatial_reconstruction':new,'old_reconstruction':old,'spatial_difference':difference,
        'old_difference':old_difference,'spatial_anomaly':anomaly,'old_anomaly':old_anomaly,'spatial_slice_scores':slices,'old_slice_scores':old_slices}


def predict_readouts(images,brain,artifacts,progress=None):
    spec,models,new_model,old_model=artifacts
    prediction=reconstruct_features(images,brain,new_model,old_model,progress)
    row={'features':prediction['features']}
    scores={family:float(predict_head(model,[row],family)[0]) for family,model in models.items()}
    scores.update({'structural_scalar':prediction['features']['old_ae_structure'],'spatial_scalar':prediction['features']['spatial_ae_l1']})
    prediction['scores']=scores;prediction['artifact_fingerprint']=spec['artifact_fingerprint'];return prediction


def decision(spec,prediction,family='combined',policy='balanced'):
    if family not in FEATURE_SETS or policy not in POLICY_LABELS:raise ValueError('Unsupported readout or policy')
    if prediction['artifact_fingerprint']!=spec['artifact_fingerprint']:raise ValueError('Prediction belongs to another readout configuration')
    threshold=spec['selection']['methods'][family]['thresholds'][policy];score=prediction['scores'][family]
    return {'family':family,'policy':policy,'score':score,'threshold':threshold,'above_threshold':bool(score>threshold),
        'supervised':True,'scores_are_calibrated_probabilities':False,'artifact_fingerprint':spec['artifact_fingerprint'],
        'checkpoint_sha256':spec['checkpoints'],'scope':spec['scope']}
