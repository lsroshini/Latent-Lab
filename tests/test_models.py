"""Current vector AE/VAE architecture and deterministic inference contracts."""
import unittest
import torch
from brain_mri_anomaly.models import ReconstructionModel,ModelConfig,losses


class ModelTests(unittest.TestCase):
    def test_vae_reparameterization_has_gradients_and_eval_is_deterministic(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        model=ReconstructionModel(ModelConfig(image_size=64,latent_dim=8,channels=(8,16,32,64)))
        image=torch.rand(2,1,64,64);output,mean,logvar=model(image)
        loss,_,_=losses(output,image,torch.ones_like(image),mean,logvar,.01)
        loss.backward()
        self.assertEqual(output.shape,image.shape)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.logvar.weight.grad.abs().sum().item(),0)
        model.eval();self.assertTrue(torch.equal(model(image)[0],model(image)[0]))

    def test_conventional_ae_matches_input_geometry(self):
        model=ReconstructionModel(ModelConfig(kind='ae',image_size=64,latent_dim=8,channels=(8,16,32,64)))
        image=torch.rand(1,1,64,64);model.eval()
        self.assertEqual(model(image)[0].shape,image.shape)


if __name__=='__main__':unittest.main()
