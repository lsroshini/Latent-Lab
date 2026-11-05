import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from brain_mri_anomaly.hybrid import BinnedPixels,score_maps,volume_score,robust_parameters,normalized

class HybridTests(unittest.TestCase):
    def test_binned_auroc_bound_covers_sklearn_with_ties(self):
        rng=np.random.default_rng(73);scores=rng.random(4000).astype(np.float32)
        scores[:100]=.5;labels=rng.integers(0,2,len(scores))
        metric=BinnedPixels(64);metric.update(scores[:1700],labels[:1700]);metric.update(scores[1700:],labels[1700:])
        result=metric.result();exact=roc_auc_score(labels,scores)
        self.assertLessEqual(abs(result['auroc']-exact),result['auroc_absolute_error_bound']+1e-12)
    def test_score_maps_ignore_background_and_identical_images_have_zero_scores(self):
        image=np.full((2,16,16),.5,dtype=np.float32);brain=np.zeros_like(image);brain[:,4:12,4:12]=1
        for values in score_maps(image,image,brain).values():np.testing.assert_allclose(values,0,atol=1e-6)
        maps=score_maps(image,np.zeros_like(image),brain)
        for values in maps.values():self.assertTrue(np.isfinite(values).all());self.assertTrue((values[brain==0]==0).all())
        self.assertAlmostEqual(volume_score(maps['l1'],brain),.5)
        p=robust_parameters([2,2,2]);self.assertTrue(np.isfinite(normalized([1,2,3],p)).all())
    def test_invalid_histogram_inputs_fail(self):
        with self.assertRaises(ValueError):BinnedPixels().update([np.nan],[1])
        with self.assertRaises(ValueError):BinnedPixels().update([1.1],[0])

if __name__=='__main__':unittest.main()
