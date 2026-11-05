import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from brain_mri_anomaly.supervised_inference import reconstruct_features,decision,validate_runtime_sources,RUNTIME_SOURCE_PATHS
from brain_mri_anomaly.common import sha256


class IdentityModel(torch.nn.Module):
    def forward(self,x,sample=False):
        mean=x.mean((2,3));return x,mean,torch.zeros_like(mean)


class SupervisedInferenceTests(unittest.TestCase):
    def test_distribution_checks_runtime_and_does_not_require_research_runners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected={}
            for name in RUNTIME_SOURCE_PATHS:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'prediction fixture');expected[name]=sha256(path)
            expected['scripts/run_historical_study.py']='historical documentation only'
            validate_runtime_sources(expected,root)
            (root/RUNTIME_SOURCE_PATHS[0]).write_bytes(b'changed prediction code')
            with self.assertRaisesRegex(ValueError,'implementation changed'):validate_runtime_sources(expected,root)
            expected.pop(RUNTIME_SOURCE_PATHS[0])
            with self.assertRaises(ValueError):validate_runtime_sources(expected,root)

    def test_full_volume_feature_extraction_does_not_change_inputs(self):
        image=np.full((2,192,192),.4,np.float32);brain=np.ones_like(image,bool);before=image.copy();updates=[]
        result=reconstruct_features(image,brain,IdentityModel(),IdentityModel(),lambda a,b:updates.append((a,b)))
        np.testing.assert_array_equal(image,before);np.testing.assert_array_equal(result['spatial_reconstruction'],image)
        for key in ('spatial_ae_l1','old_ae_l1','old_ae_structure','spatial_brain_mae'):self.assertEqual(result['features'][key],0.)
        self.assertEqual(updates[-1],(4,4))
        with self.assertRaises(ValueError):reconstruct_features(image[:1],brain[:1],IdentityModel(),IdentityModel())

    def test_decision_uses_frozen_precision_and_rejects_wrong_artifact(self):
        spec={'artifact_fingerprint':'frozen','selection':{'methods':{'combined':{'thresholds':{'balanced':.5}}}},'checkpoints':{},'scope':'fixture'}
        prediction={'artifact_fingerprint':'frozen','scores':{'combined':.5}}
        self.assertFalse(decision(spec,prediction)['above_threshold'])
        prediction['scores']['combined']=float(np.nextafter(.5,1.))
        self.assertTrue(decision(spec,prediction)['above_threshold'])
        prediction['artifact_fingerprint']='other'
        with self.assertRaises(ValueError):decision(spec,prediction)

if __name__=='__main__':unittest.main()
