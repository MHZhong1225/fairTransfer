"""Official BAGEL generation with auditable, exactly replayed initial noise."""
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from common import STUDY,BAGEL_WEIGHTS,now,read,rows,sha,write
from bagel_backend_v2 import DetailBagel,GENERATION,PREPROCESSING


def run(args):
    torch.set_num_threads(4)
    plan=[r for i,r in enumerate(rows(args.manifest)) if i%args.shards==args.shard]
    directory=args.output
    contract=dict(model_revision=BAGEL_WEIGHTS.name,adapter=str(args.adapter) if args.adapter else None,
        adapter_metadata_sha256=sha(args.adapter/'adapter.json') if args.adapter else None,
        manifest=str(args.manifest),manifest_sha256=sha(args.manifest),shard=args.shard,shards=args.shards,
        images=len(plan),generation=GENERATION,preprocessing=PREPROCESSING,
        replay_noise=str(args.replay_noise) if args.replay_noise else None,
        save_base_noise=str(args.save_base_noise) if args.save_base_noise else None,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('generate_images.py','bagel_backend.py','bagel_backend_v2.py')})
    if args.resume:
        if read(directory/'contract.json')!=contract:
            raise ValueError('Resume cannot change generation contract')
    else:
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'images').mkdir();(directory/'observations').mkdir();write(directory/'contract.json',contract)
    status=dict(status='loading',started_utc=now(),planned=len(plan),completed=0)
    write(directory/'status.json',status)
    try:
        backend=DetailBagel(understanding_mode='inference')
        if args.adapter:
            backend.install(0);backend.load(args.adapter)
        status['status']='generating'
        for item in plan:
            record_path=directory/'observations'/(item['id']+'.json')
            if record_path.exists():
                existing=read(record_path)
                if sha(existing['image_path'])!=existing['image_sha256']:
                    raise ValueError('Previously generated image changed')
                status['completed']+=1;continue
            image_path=directory/'images'/(item['id']+'.png')
            if image_path.exists():
                raise RuntimeError('Uncommitted image exists; preserve and reconcile it before resuming')
            replay=None;reference=None
            if args.replay_noise:
                reference=read(args.replay_noise/(item['id']+'.json'))
                if reference['prompt']!=item['prompt'] or reference['seed']!=item['seed']:
                    raise ValueError('Reference noise belongs to another generation request')
                noise_path=Path(reference['noise_path'])
                if sha(noise_path)!=reference['noise_file_sha256']:
                    raise ValueError('Reference noise file changed')
                replay=torch.from_numpy(np.load(noise_path,allow_pickle=False))
            started=time.perf_counter()
            noise,output=backend.render(item['prompt'],item['seed'],image_path,replay_noise=replay)
            if reference and output['noise_tensor_sha256']!=reference['noise_tensor_sha256']:
                raise ValueError('Actual initial noise differs from paired base noise')
            if args.save_base_noise:
                args.save_base_noise.mkdir(parents=True,exist_ok=True)
                noise_path=image_path.with_suffix('.noise.npy')
                write(args.save_base_noise/(item['id']+'.json'),dict(prompt=item['prompt'],seed=item['seed'],
                    noise_path=str(noise_path),noise_file_sha256=sha(noise_path),noise_tensor_sha256=output['noise_tensor_sha256']))
            write(record_path,{**item,**output,'seconds':time.perf_counter()-started,'completed_utc':now()})
            status.update(completed=status['completed']+1,last_id=item['id'],updated_utc=now());write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter',type=Path);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--replay-noise',type=Path)
    parser.add_argument('--save-base-noise',type=Path);parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
