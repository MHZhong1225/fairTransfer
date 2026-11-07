"""Fixed-budget BAGEL text-only SFT with identical image order across arms."""
import argparse
import math
from pathlib import Path
import random
import time
import traceback
import json
import torch
from common import STUDY,BAGEL_WEIGHTS,now,read,rows,sha,write
from bagel_tasks import TaskBagel
from bagel_backend import frozen_versions,check_frozen,GENERATION,PREPROCESSING
from supervision import make_pack
from training_schedule import training_pack


def development_loss(backend,records):
    # Monitoring only: fixed common T2 targets. No G output or checkpoint selection.
    values=[]
    with torch.no_grad():
        for row in records:
            pack=make_pack(row,'T2',0)
            loss,tokens=backend.pack_loss(pack)
            values.append(dict(id=row['id'],loss=float(loss),tokens=tokens))
    return dict(mean_image_loss=sum(r['loss'] for r in values)/len(values) if values else None,
        records=values,scope='Teacher-forced text loss under the training image transform; not an inference fairness metric.')


def save_state(backend,optimizer,directory,step,next_epoch,next_offset):
    target=directory/'checkpoints'/f'step_{step:06d}'
    target.mkdir(parents=True,exist_ok=False)
    backend.save(target/'adapter')
    torch.save(optimizer.state_dict(),target/'optimizer.pt')
    state=dict(step=step,next_epoch=next_epoch,next_offset=next_offset,checkpoint=str(target),saved_utc=now())
    write(target/'complete.json',state)
    write(directory/'latest.json',state)
    return state


def run(args):
    torch.set_num_threads(4)
    train=rows(args.train);dev=rows(args.dev) if args.dev else []
    source_names=('train_arm_v2.py','training_schedule.py','bagel_tasks.py','bagel_backend.py','bagel_backend_v2.py','supervision.py')
    contract=dict(arm=args.arm,seed=args.seed,epochs=args.epochs,batch_images=args.batch_images,
        learning_rate=1e-4,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=1.,
        lr_schedule='constant',rank=8,alpha=16,adapter_parameters=20185088,
        train_path=str(args.train),train_sha256=sha(args.train),train_images=len(train),
        dev_path=str(args.dev) if args.dev else None,dev_sha256=sha(args.dev) if args.dev else None,
        expected_updates=args.epochs*math.ceil(len(train)/args.batch_images),image_exposures=len(train)*args.epochs,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in source_names},
        model_revision=BAGEL_WEIGHTS.name,generation=GENERATION,preprocessing=PREPROCESSING,
        task='I→T cross entropy only, weighted mean of per-image mean target-token losses. A T3 image is encoded once and branches into isolated question contexts.',
        task_schedule='Permute each image’s three exposure variants independently before training, equally across arms; randomize independent QA branch order. This preserves the exact per-image target multiset and prevents a caption-then-QA curriculum confound.',
        selection='Fixed final endpoint after the prespecified number of epochs. Development loss is monitoring only. No checkpoint is selected using G or test outcomes.',
        technical=args.technical)
    directory=args.output
    if args.resume:
        if read(directory/'contract.json')!=contract:
            raise ValueError('Resume cannot change the frozen training contract')
        attempt=len(list((directory/'logs').glob('attempt_*.jsonl')))
    else:
        directory.mkdir(parents=True,exist_ok=False);(directory/'logs').mkdir()
        write(directory/'contract.json',contract);attempt=0
    status=dict(status='loading',attempt=attempt,started_utc=now(),expected_updates=contract['expected_updates'])
    write(directory/f'attempt_{attempt}_status.json',status);write(directory/'status.json',status)
    try:
        backend=TaskBagel();backend.install(args.seed);versions=frozen_versions(backend.model)
        optimizer=torch.optim.AdamW(backend.trainable.values(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        start_epoch=start_offset=step=0
        if args.resume:
            state=read(directory/'latest.json');checkpoint=Path(state['checkpoint'])
            if read(checkpoint/'complete.json')!=state:
                raise ValueError('Incomplete recovery checkpoint')
            backend.load(checkpoint/'adapter')
            optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location=backend.device,weights_only=True))
            start_epoch=state['next_epoch'];start_offset=state['next_offset'];step=state['step']
        else:
            backend.save(directory/'initial_adapter')
        status.update(status='training',step=step);write(directory/'status.json',status)
        with (directory/'logs'/f'attempt_{attempt:02d}.jsonl').open('x',buffering=1) as log:
            for epoch in range(start_epoch,args.epochs):
                ordered=list(train);random.Random(args.seed+10000*epoch).shuffle(ordered)
                begin=start_offset if epoch==start_epoch else 0
                for offset in range(begin,len(ordered),args.batch_images):
                    started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    batch=ordered[offset:offset+args.batch_images];observations=[]
                    for row in batch:
                        if sha(row['image_path'])!=row['image_sha256']:
                            raise ValueError('Training image changed')
                        pack=training_pack(row,args.arm,epoch)
                        loss,tokens=backend.pack_loss(pack)
                        weighted=loss*row.get('weight',1.)/len(batch)
                        if not torch.isfinite(weighted):
                            raise ValueError('Nonfinite text loss')
                        weighted.backward()
                        observations.append(dict(id=row['id'],loss=float(loss.detach()),target_tokens=tokens,
                            weight=row.get('weight',1.),supervision_variant=pack['supervision_variant'],question_contexts=len(pack['tasks']),tasks=[t['kind'] for t in pack['tasks']]))
                        del loss,weighted
                    norm=torch.nn.utils.clip_grad_norm_(list(backend.trainable.values()),1.)
                    if not torch.isfinite(norm):
                        raise ValueError('Nonfinite gradient norm')
                    optimizer.step();step+=1
                    next_offset=offset+len(batch);epoch_done=next_offset==len(ordered)
                    if step%100==0 or epoch_done:
                        check_frozen(backend.model,versions)
                    event=dict(step=step,epoch=epoch,offset=offset,images=observations,gradient_norm=float(norm),
                        seconds=time.perf_counter()-started,completed_utc=now())
                    log.write(json.dumps(event,ensure_ascii=False)+'\n')
                    status.update(step=step,epoch=epoch,offset=next_offset,last_loss=sum(o['loss'] for o in observations)/len(observations),updated_utc=now())
                    write(directory/'status.json',status)
                    if step%250==0 or epoch_done:
                        save_state(backend,optimizer,directory,step,epoch+1 if epoch_done else epoch,0 if epoch_done else next_offset)
                optimizer.zero_grad(set_to_none=True)
                write(directory/f'dev_epoch_{epoch+1}.json',development_loss(backend,dev))
        if step!=contract['expected_updates']:
            raise ValueError('Actual update count does not match the frozen budget')
        check_frozen(backend.model,versions)
        state=read(directory/'latest.json')
        status.update(status='completed',completed_utc=now(),final_adapter=str(Path(state['checkpoint'])/'adapter'),
            frozen=check_frozen(backend.model,versions),training_contract_sha256=sha(directory/'contract.json'))
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now());raise
    finally:
        write(directory/f'attempt_{attempt}_status.json',status);write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('T0','T1','T2','T3'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--dev',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-images',type=int,default=8)
    parser.add_argument('--technical',action='store_true')
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
