"""Frozen-reconstruction hybrid scores and bounded-error exploratory pixel metrics."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

MAP_NAMES=('l1','structure','fusion','smooth_fusion')

def score_maps(image,reconstruction,brain):
    brain=np.asarray(brain)>0
    residual=np.abs(image-reconstruction).astype(np.float32)
    structural=np.empty_like(residual)
    for i in range(len(image)):
        _,similarity=structural_similarity(image[i],reconstruction[i],data_range=1.,full=True)
        structural[i]=np.clip((1-similarity)/2,0,1)
    residual*=brain;structural*=brain
    fused=np.sqrt(residual*structural).astype(np.float32)
    denominator=gaussian_filter(brain.astype(np.float32),sigma=(0,1,1))
    smoothed=gaussian_filter(fused,sigma=(0,1,1))/np.maximum(denominator,1e-8)
    return dict(zip(MAP_NAMES,(residual,structural,fused,(np.clip(smoothed,0,1)*brain).astype(np.float32))))

def volume_score(anomaly,brain):
    scores=[]
    for values,mask in zip(anomaly,brain):
        values=values[mask>0]
        if not len(values):raise ValueError('Empty prepared brain slice')
        k=max(1,int(np.ceil(.01*len(values))))
        scores.append(float(np.partition(values,len(values)-k)[-k:].mean()))
    k=max(1,int(np.ceil(.05*len(scores))))
    return float(np.sort(scores)[-k:].mean())

def extract_latents(model,images,batch_size=32):
    device=next(model.parameters()).device;values=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            batch=np.array(images[start:start+batch_size],dtype=np.float32)
            if batch.ndim==3:batch=batch[:,None]
            features=model.encoder(torch.from_numpy(batch).to(device)).flatten(1)
            values.append(model.mean(features).cpu().numpy())
    return np.concatenate(values)

def robust_parameters(values):
    q25,median,q75=np.quantile(values,[.25,.5,.75])
    return {'median':float(median),'scale':float(max(q75-q25,1e-6))}

def normalized(values,parameters):
    return (np.asarray(values)-parameters['median'])/parameters['scale']

class BinnedPixels:
    """All voxels, quantized scores. AUROC error bound covers within-bin ordering."""
    def __init__(self,bins=65536):
        self.counts=np.zeros((2,bins),dtype=np.int64);self.bins=bins
    def update(self,scores,labels):
        values=np.asarray(scores,dtype=np.float32).ravel();labels=np.asarray(labels).ravel()
        if values.shape!=labels.shape or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError('Expected paired finite scores in [0,1]')
        if not np.isin(labels,[0,1]).all():raise ValueError('Expected binary labels')
        indices=np.minimum((values.astype(np.float64)*self.bins).astype(np.int64),self.bins-1)
        for label in (0,1):self.counts[label]+=np.bincount(indices[labels==label],minlength=self.bins)
    def result(self):
        negative,positive=self.counts.astype(np.float64);n=negative.sum();p=positive.sum()
        if not n or not p:return {'auroc':None,'auprc':None,'auroc_absolute_error_bound':None}
        auroc=float(np.sum(positive*(np.cumsum(negative)-.5*negative))/(p*n))
        bound=float(.5*np.sum(positive*negative)/(p*n))
        tp=np.cumsum(positive[::-1]);fp=np.cumsum(negative[::-1])
        precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
        ap=float(np.sum(positive[::-1]*precision)/p)
        return {'auroc':auroc,'auprc':ap,'auroc_absolute_error_bound':bound,'positive_pixels':int(p),'negative_pixels':int(n),
                'method':f'All voxels; {self.bins} uniform score bins. AUROC bounded-error approximation; AP approximate. Not the exact baseline evaluator.'}
