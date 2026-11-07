"""Native image-to-text responses; expected facts never enter BAGEL inputs."""
import argparse
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import torch
from common import BAGEL_WEIGHTS,now,read,rows,sha,write
from bagel_backend_v2 import DetailBagel,PREPROCESSING


def native_answer(backend,row):
    observed={};original=backend.model.generate_text
    def measured(*args,**kwargs):
        last={}
        def record_head(module,inputs,output):
            last['predicted_token']=output.argmax(dim=-1).detach()
        handle=backend.model.language_model.lm_head.register_forward_hook(record_head)
        try:
            output=original(*args,**kwargs)
        finally:
            handle.remove()
        observed['native_sequence_length']=int(output.shape[0])
        observed['ended_with_eos']=bool(last['predicted_token'].reshape(-1)[0].item()==backend.native.new_token_ids['eos_token_id'])
        observed['hit_length_limit']=bool(output.shape[0]>=1024 and not observed['ended_with_eos'])
        return output
    inputs={'image_path':row.get('image_path'),'question':row['question']}
    with patch.object(backend.model,'generate_text',side_effect=measured):
        answer=backend.answer(inputs,max_length=1024)
    return answer,observed


def run(args):
    torch.set_num_threads(4)
    plan=[r for i,r in enumerate(rows(args.cases)) if i%args.shards==args.shard]
    contract=dict(model_revision=BAGEL_WEIGHTS.name,cases_sha256=sha(args.cases),cases=str(args.cases),
        adapter=str(args.adapter) if args.adapter else None,
        adapter_metadata_sha256=sha(args.adapter/'adapter.json') if args.adapter else None,
        planned=len(plan),shard=args.shard,shards=args.shards,max_length=1024,do_sample=False,
        preprocessing=PREPROCESSING,source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('run_understanding.py','bagel_backend.py','bagel_backend_v2.py')},
        input_boundary='Only question and actual image pixels are given to BAGEL; reference facts and source metadata are evaluation-only.')
    directory=args.output
    if args.resume:
        if read(directory/'contract.json')!=contract:
            raise ValueError('Resume cannot change evaluation inputs or model')
    else:
        directory.mkdir(parents=True,exist_ok=False);(directory/'observations').mkdir();write(directory/'contract.json',contract)
    status=dict(status='loading',planned=len(plan),completed=0,started_utc=now());write(directory/'status.json',status)
    try:
        backend=DetailBagel(understanding_mode='inference')
        if args.adapter:
            backend.install(0);backend.load(args.adapter)
        status['status']='evaluating'
        for row in plan:
            path=directory/'observations'/(row['id']+'.json')
            if path.exists():
                status['completed']+=1;continue
            if row.get('image_path') and sha(row['image_path'])!=row['image_sha256']:
                raise ValueError('Evaluation image changed')
            started=time.perf_counter()
            response,decoding=native_answer(backend,row)
            write(path,dict(id=row['id'],image_id=row['image_id'],response=response,decoding=decoding,
                seconds=time.perf_counter()-started,completed_utc=now()))
            status.update(completed=status['completed']+1,last_id=row['id'],updated_utc=now());write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter',type=Path);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1);parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
