import io
import unittest
import numpy as np
from PIL import Image
from brain_mri_anomaly.image_input import prepare_image_preview


def encoded(array,mode=None):
    buffer=io.BytesIO();Image.fromarray(array,mode=mode).save(buffer,format='PNG');return buffer.getvalue()


class ImageInputTests(unittest.TestCase):
    def test_color_conversion_preserves_proportions_and_pads(self):
        image=np.zeros((100,200,3),np.uint8);image[20:80,30:170]=[200,100,50]
        output,mask,metadata=prepare_image_preview(encoded(image))
        self.assertEqual(output.shape,(1,192,192));self.assertEqual(output.dtype,np.float32)
        self.assertEqual(metadata['resized_size'],[192,96]);self.assertEqual(metadata['padding_left_top'],[0,48])
        self.assertFalse(output[:,:48].any());self.assertTrue(mask.any())
        self.assertFalse(metadata['anatomical_registration'])

    def test_16_bit_intensity_and_transparency(self):
        gradient=np.tile(np.linspace(0,65535,128,dtype=np.uint16),(128,1))
        output,_,_=prepare_image_preview(encoded(gradient))
        self.assertGreater(np.unique(output).size,50)
        rgba=np.full((40,40,4),255,np.uint8);rgba[:10,:,3]=0
        output,_,_=prepare_image_preview(encoded(rgba))
        self.assertEqual(float(output[0,0].max()),0.)

    def test_invalid_and_blank_images_are_rejected(self):
        for raw in (b'not an image',encoded(np.zeros((20,20),np.uint8))):
            with self.assertRaises(ValueError):prepare_image_preview(raw)
