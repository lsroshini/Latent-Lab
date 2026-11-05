"""Small supervised readouts of frozen AE features, with explicit controls.

This module does not train or modify a reconstruction network. Scores are not
calibrated clinical probabilities. Dataset/source confounding remains possible.
"""
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_FEATURES=('intensity_mean','intensity_std','q10','q25','q50','q75','q90','q99',
                'bright_fraction','dark_fraction','gradient_mean')
GEOMETRY_FEATURES=('brain_ml','retained_slices')
RECONSTRUCTION_FEATURES=('spatial_ae_l1','old_ae_l1','old_ae_structure','spatial_brain_mae')
FEATURE_SETS={'geometry_only':GEOMETRY_FEATURES,'image_only':IMAGE_FEATURES,
              'reconstruction_only':RECONSTRUCTION_FEATURES,'combined':IMAGE_FEATURES+RECONSTRUCTION_FEATURES}
CS=(.1,1.,10.)


def image_features(image,brain):
    image=np.asarray(image);brain=np.asarray(brain)>0
    if image.ndim!=3 or image.shape!=brain.shape or not brain.any() or not np.isfinite(image).all():
        raise ValueError('Require a finite prepared volume with a nonempty brain region')
    values=image[brain].astype(np.float64);quantiles=np.quantile(values,[.1,.25,.5,.75,.9,.99])
    if values.min()<0 or values.max()>1:raise ValueError('Prepared intensities must be in [0,1]')
    gradient_sum=0.;pairs=0
    for axis in (1,2):
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        valid=brain[tuple(left)]&brain[tuple(right)]
        gradient_sum+=float(np.abs(image[tuple(left)]-image[tuple(right)])[valid].sum(dtype=np.float64));pairs+=int(valid.sum())
    result={'intensity_mean':float(values.mean()),'intensity_std':float(values.std()),
        **{k:float(v) for k,v in zip(('q10','q25','q50','q75','q90','q99'),quantiles)},
        'bright_fraction':float(np.mean(values>.8)),'dark_fraction':float(np.mean(values<.1)),
        'gradient_mean':gradient_sum/max(pairs,1),'brain_ml':float(brain.sum()*1.5**3/1000),
        'retained_slices':int(len(image))}
    return result


def matrix(rows,family):
    if family not in FEATURE_SETS:raise ValueError('Unknown feature family')
    result=np.asarray([[row['features'][key] for key in FEATURE_SETS[family]] for row in rows],dtype=np.float64)
    if result.ndim!=2 or not np.isfinite(result).all():raise ValueError('Nonfinite or missing classifier features')
    return result


def group_class_weights(rows):
    """Equal weight per patient within each class; equal total class weight."""
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[(row['dataset'],row['patient_id'])].append(i)
    counts={0:0,1:0}
    for indices in groups.values():
        labels={rows[i]['label'] for i in indices}
        if len(labels)!=1 or next(iter(labels)) not in counts:raise ValueError('Inconsistent patient labels')
        counts[next(iter(labels))]+=1
    if not all(counts.values()):raise ValueError('Fitting needs both classes')
    weights=np.empty(len(rows),dtype=np.float64)
    for indices in groups.values():
        label=rows[indices[0]]['label'];weights[indices]=1/(counts[label]*len(indices))
    return weights*(len(rows)/weights.sum())


def fit_head(rows,family,c):
    if not rows or any(row['role']!='fit' for row in rows):raise ValueError('Classifier fitting accepts fit rows only')
    weights=group_class_weights(rows);x=matrix(rows,family);y=np.array([r['label'] for r in rows])
    model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=c,solver='lbfgs',max_iter=1000,random_state=42))])
    model.fit(x,y,scale__sample_weight=weights,classifier__sample_weight=weights)
    if model.named_steps['classifier'].n_iter_[0]>=1000:raise RuntimeError('Classifier did not converge within its fixed budget')
    return model


def predict_head(model,rows,family):
    classes=model.named_steps['classifier'].classes_
    positive=int(np.flatnonzero(classes==1)[0]);scores=model.predict_proba(matrix(rows,family))[:,positive]
    if not np.isfinite(scores).all():raise ValueError('Invalid classifier score')
    return scores
