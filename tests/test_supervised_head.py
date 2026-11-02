import unittest
import numpy as np
from brain_mri_anomaly.supervised_head import image_features,FEATURE_SETS,group_class_weights,fit_head,predict_head


def fixture():
    rows=[]
    for i,label in enumerate([0,0,1,1]):
        rows.append({'dataset':'fixture','patient_id':str(i),'role':'fit','label':label,
            'features':{k:.1+.6*label+.01*i for k in set(sum((list(v) for v in FEATURE_SETS.values()),[]))}})
    return rows


class SupervisedHeadTests(unittest.TestCase):
    def test_image_statistics_ignore_background_and_do_not_expose_labels(self):
        image=np.zeros((2,8,8),np.float32);brain=np.zeros_like(image,bool);brain[:,2:6,2:6]=True;image[brain]=.5
        features=image_features(image,brain);self.assertEqual(features['intensity_mean'],.5);self.assertEqual(features['gradient_mean'],0.)
        image[~brain]=.9;self.assertEqual(features,image_features(image,brain))
        forbidden={'label','patient_id','scan_id','site','dataset','role'}
        self.assertFalse(forbidden.intersection(set(sum((list(v) for v in FEATURE_SETS.values()),[]))))

    def test_repeated_scans_do_not_increase_patient_class_weight(self):
        rows=fixture();rows.append({**rows[0]});weights=group_class_weights(rows)
        self.assertAlmostEqual(weights[0]+weights[-1],weights[1])
        self.assertAlmostEqual(sum(weights[i] for i,r in enumerate(rows) if r['label']==0),sum(weights[i] for i,r in enumerate(rows) if r['label']==1))

    def test_training_rejects_evaluation_rows_and_prediction_is_reproducible(self):
        rows=fixture();model=fit_head(rows,'reconstruction_only',1.)
        first=predict_head(model,rows,'reconstruction_only');np.testing.assert_array_equal(first,predict_head(model,rows,'reconstruction_only'))
        self.assertGreater(float(first[2:].mean()),float(first[:2].mean()))
        rows[-1]['role']='benchmark'
        with self.assertRaisesRegex(ValueError,'fit rows'):fit_head(rows,'combined',1.)

if __name__=='__main__':unittest.main()
