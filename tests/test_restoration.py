import unittest
import torch
from brain_mri_anomaly.restoration import SpatialRestorationModel,RestorationConfig
from brain_mri_anomaly.models import losses


class RestorationTests(unittest.TestCase):
    def test_spatial_vae_deterministic_inference_and_train_gradients(self):
        torch.manual_seed(7);model=SpatialRestorationModel(RestorationConfig(image_size=32));x=torch.rand(2,1,32,32)
        model.eval();a,mean,logvar=model(x);b,*_=model(x)
        self.assertEqual(tuple(mean.shape),(2,8,4,4));torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(bool(((a>=0)&(a<=1)).all()))
        model.train();a,mean,logvar=model(x);b,*_=model(x)
        self.assertFalse(torch.equal(a,b));loss,*_=losses(a,x,torch.ones_like(x),mean,logvar,beta=.001)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(model.mean.weight.grad).all())
        self.assertTrue(torch.isfinite(model.logvar.weight.grad).all())

    def test_ae_has_no_stochastic_latent_and_features_affect_output(self):
        model=SpatialRestorationModel(RestorationConfig(kind='ae',image_size=32));x=torch.rand(2,1,32,32)
        model.train();a,mean,logvar=model(x);b,*_=model(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertFalse(bool(logvar.any()))
        mean,_,skips=model.encode(x)
        self.assertFalse(torch.equal(model.decode(mean,skips),model.decode(torch.zeros_like(mean),skips)))
        self.assertFalse(torch.equal(model.decode(mean,skips),model.decode(mean,skips,skip_scale=0.)))
        with self.assertRaises(ValueError):model(torch.rand(2,1,40,32))

if __name__=='__main__':unittest.main()
