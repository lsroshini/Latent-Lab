import numpy as np


def panels(original,reconstruction,difference,anomaly,pixel_threshold):
    import cv2
    original_rgb=np.repeat(np.clip(original[...,None],0,1),3,axis=-1)
    scale=max(2*float(pixel_threshold),1e-6)
    intensity=np.clip(anomaly/scale,0,1)
    heat=cv2.applyColorMap((intensity*255).astype(np.uint8),cv2.COLORMAP_INFERNO)[...,::-1]/255.
    alpha=.55*intensity[...,None]
    overlay=(1-alpha)*original_rgb+alpha*heat
    return {"Original prepared MRI":original_rgb,"Reconstruction":np.clip(reconstruction,0,1),
            "Absolute difference":difference,"Anomaly heatmap":heat,"Anomaly overlay":overlay}


def save_panel_image(path,views,pixel_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(16,4))
    for ax,(name,values) in zip(axes,views.items()):
        if values.ndim==2:ax.imshow(values,cmap="gray",vmin=0,vmax=1)
        else:ax.imshow(values)
        ax.set_title(name,fontsize=10);ax.axis("off")
    fig.suptitle("Reconstruction residuals are not lesion probabilities")
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
