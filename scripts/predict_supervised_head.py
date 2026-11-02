"""Export experimental supervised AE readouts for a complete prepared volume."""
import argparse
import numpy as np
import torch
import bootstrap
from brain_mri_anomaly.common import write_json
from brain_mri_anomaly.supervised_inference import load_artifacts,predict_readouts,decision,FAMILY_LABELS,POLICY_LABELS
from brain_mri_anomaly.visualization import panels,save_panel_image
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--family',choices=list(FAMILY_LABELS),default='combined')
    parser.add_argument('--policy',choices=list(POLICY_LABELS),default='balanced');args=parser.parse_args();torch.set_num_threads(2)
    artifacts=load_artifacts();spec=artifacts[0]
    with np.load(args.prepared,allow_pickle=False) as volume:
        image=volume['image'];brain=volume['brain']>0;z=volume['z_indices'];prediction=predict_readouts(image,brain,artifacts)
        result=decision(spec,prediction,args.family,args.policy);args.output_dir.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(args.output_dir/'prediction.npz',image=image,z_indices=z,
            **{key:prediction[key] for key in ('spatial_reconstruction','old_reconstruction','spatial_difference','old_difference','spatial_anomaly','old_anomaly')},
            classifier_score=result['score'],decision_threshold=result['threshold'],family=args.family,policy=args.policy,
            artifact_fingerprint=spec['artifact_fingerprint'],supervised=True,scores_are_calibrated_probabilities=False)
        index=int(np.argmax(prediction['spatial_slice_scores']))
        save_panel_image(args.output_dir/'reconstruction.png',panels(image[index],prediction['spatial_reconstruction'][index],
            prediction['spatial_difference'][index],prediction['spatial_anomaly'][index],.25),.25)
        write_json(args.output_dir/'decision.json',{**result,'all_scores':prediction['scores'],'features':prediction['features'],
            'display_scale':.5,'preview_slice':index,'prepared_grid_z_index':int(z[index])})

if __name__=='__main__':main()
