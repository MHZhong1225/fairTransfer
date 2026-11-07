"""Corruption checks for final evidence; fixtures are not model results."""
from pathlib import Path
import tempfile
import unittest
from common import sha,write
from validate_final_outputs import verify_generation_artifacts,verify_training_endpoint


class FinalArtifactIntegrity(unittest.TestCase):
    def test_saved_noise_must_match_its_record(self):
        with tempfile.TemporaryDirectory() as name:
            image=Path(name)/'sample.png';image.write_bytes(b'fixture image')
            noise=image.with_suffix('.noise.npy');noise.write_bytes(b'actual noise fixture')
            value=dict(image_path=str(image),image_sha256=sha(image),noise_file_sha256=sha(noise),
                noise_tensor_sha256='fixture tensor digest',actual_noise_replayed=True,generation_settings={'steps':50})
            verify_generation_artifacts(value,value,True)
            noise.write_bytes(b'changed while observation still says completed')
            with self.assertRaisesRegex(ValueError,'noise file changed'):
                verify_generation_artifacts(value,value,True)

    def test_independent_noise_with_same_seed_is_not_replay(self):
        with tempfile.TemporaryDirectory() as name:
            image=Path(name)/'sample.png';image.write_bytes(b'fixture image')
            noise=image.with_suffix('.noise.npy');noise.write_bytes(b'noise')
            value=dict(image_path=str(image),image_sha256=sha(image),noise_file_sha256=sha(noise),
                noise_tensor_sha256='same digest',actual_noise_replayed=False,generation_settings={'steps':50},seed=1)
            with self.assertRaisesRegex(ValueError,'did not replay'):
                verify_generation_artifacts(value,value,True)

    def test_completed_flag_cannot_replace_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);adapter=root/'checkpoints/step_000006/adapter';adapter.mkdir(parents=True)
            (adapter/'adapter.safetensors').write_bytes(b'fixture weights')
            write(adapter/'adapter.json',dict(sha256=sha(adapter/'adapter.safetensors')))
            train=root/'train.jsonl';train.write_text('fixture train')
            dev=root/'dev.jsonl';dev.write_text('fixture dev')
            contract=dict(arm='T0',seed=7101,epochs=3,batch_images=8,train_images=16,
                train_sha256=sha(train),dev_sha256=sha(dev),technical=False)
            write(root/'contract.json',contract)
            status=dict(status='completed',step=6,final_adapter=str(adapter),
                training_contract_sha256=sha(root/'contract.json'),frozen=dict(unchanged=True,gradients_absent=True))
            write(root/'status.json',status)
            endpoint=dict(step=6,next_epoch=3,next_offset=0,checkpoint=str(adapter.parent))
            write(root/'latest.json',endpoint);write(adapter.parent/'complete.json',endpoint)
            state=dict(training=str(root),arm='T0',seed=7101);freeze=dict(train=str(train),dev=str(dev))
            verify_training_endpoint(state,freeze)
            endpoint['next_epoch']=2
            write(root/'latest.json',endpoint);write(adapter.parent/'complete.json',endpoint)
            with self.assertRaisesRegex(ValueError,'fixed final endpoint'):
                verify_training_endpoint(state,freeze)


if __name__=='__main__':
    unittest.main()
