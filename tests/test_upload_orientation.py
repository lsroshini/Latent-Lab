import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from brain_mri_anomaly.inference import lossless_las_source

class UploadOrientationTests(unittest.TestCase):
    def test_las_preserves_original_bytes_and_scaled_intensity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);source=directory/'source.nii.gz'
            image=nib.Nifti1Image(np.arange(60,dtype=np.int16).reshape(3,4,5),np.diag([-1.,1.,1.,1.]))
            image.header.set_slope_inter(.137,3.71);nib.save(image,source)
            before=source.read_bytes()
            result=lossless_las_source(source,directory,nib.load(source))
            self.assertEqual(result,source);self.assertEqual(result.read_bytes(),before)

    def test_reorientation_preserves_float32_values_without_integer_requantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);source=directory/'source.nii.gz'
            image=nib.Nifti1Image(np.arange(60,dtype=np.int16).reshape(3,4,5),np.eye(4))
            image.header.set_slope_inter(.137,3.71);nib.save(image,source)
            original=nib.load(source)
            expected=original.get_fdata(dtype=np.float32)[::-1,:,:]
            result=nib.load(lossless_las_source(source,directory,original))
            self.assertEqual(nib.aff2axcodes(result.affine),('L','A','S'))
            np.testing.assert_array_equal(result.get_fdata(dtype=np.float32),expected)

if __name__=='__main__':unittest.main()
