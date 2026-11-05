import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from brain_mri_anomaly.inference import cached_nifti_preparation,prepare_nifti_bytes
from brain_mri_anomaly.preprocessing import VERSION

class UploadCapabilityTests(unittest.TestCase):
    def write_block(self,root):
        path=root/'runs/runtime-capabilities/raw-upload.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'status':'unavailable','torch_version':str(torch.__version__),'device':'cpu'}))

    def test_uncached_upload_fails_before_parsing_or_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.write_block(root)
            with patch('brain_mri_anomaly.inference.ROOT',root):
                self.assertIsNone(cached_nifti_preparation(b'uncached'))
                with self.assertRaisesRegex(RuntimeError,'Application Control'):
                    prepare_nifti_bytes(b'uncached','scan.nii.gz',device='cpu')
                self.assertFalse((root/'.cache/uploads').exists())

    def test_trusted_cached_volume_remains_available_when_extraction_is_blocked(self):
        raw=b'cached-source';key=hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.write_block(root)
            directory=root/'.cache/uploads'/key/VERSION/'lossless-las-v2';directory.mkdir(parents=True)
            path=directory/'prepared.npz';np.savez_compressed(path,image=np.zeros((3,8,8),dtype=np.float32),brain=np.ones((3,8,8),dtype=np.uint8))
            (directory/'prepared.json').write_text('{}')
            with patch('brain_mri_anomaly.inference.ROOT',root):
                self.assertEqual(cached_nifti_preparation(raw),path)
                self.assertEqual(prepare_nifti_bytes(raw,'scan.nii.gz',device='cpu'),path)

if __name__=='__main__':unittest.main()
