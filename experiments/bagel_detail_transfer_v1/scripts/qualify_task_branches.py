"""Actual multi-question gradient, isolation, serialization and long-target checks."""
from copy import deepcopy
from pathlib import Path
import time
import traceback
import torch
from common import STUDY,now,rows,sha,write
from bagel_tasks import TaskBagel
from bagel_backend import frozen_versions,check_frozen


def main():
    torch.set_num_threads(4)
    directory=STUDY/'results/task_branch_qualification_v1';directory.mkdir(exist_ok=False)
    # Independently inspected native scene; these are technical statements only.
    image=rows(STUDY/'data/scene_technical_v1/manifest.jsonl')[2]
    tasks=[dict(question='What device is in front of the person?',caption='A laptop is visible in front of the person.'),
        dict(question='What is visible on the person’s face?',caption='The person is wearing glasses.'),
        dict(question='Describe the visible clothing.',caption='The person is wearing a short-sleeved patterned shirt.')]
    pack=dict(image_path=image['image_path'],tasks=tasks)
    write(directory/'contract.json',dict(created_utc=now(),image_sha256=sha(image['image_path']),pack=pack,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('qualify_task_branches.py','bagel_tasks.py','bagel_backend.py','bagel_backend_v2.py')},
        seed=7100,rank=8,alpha=16,lr=1e-4,updates=3,
        acceptance='Each branch output equals an isolated image-question forward; replacing one branch answer cannot affect another; all U adapter B gradients finite/nonzero; frozen tensors unchanged; serialized outputs equal.'))
    status=dict(status='running',started_utc=now(),checks={});write(directory/'status.json',status)
    try:
        backend=TaskBagel();backend.install(7100);versions=frozen_versions(backend.model)
        with torch.no_grad():
            together=backend.pack_logits(pack)
            separate=[backend.caption_logits(dict(image_path=pack['image_path'],**task)) for task in tasks]
            changed=deepcopy(pack);changed['tasks'][0]['caption']='A blue airplane is flying over a mountain.'
            changed_outputs=backend.pack_logits(changed)
        isolated=all(torch.equal(a,b) for a,b in zip(together,separate))
        no_answer_leak=all(torch.equal(together[i],changed_outputs[i]) for i in (1,2))
        status['checks'].update(isolated_outputs_exact=isolated,other_answers_not_visible=no_answer_leak)
        if not isolated or not no_answer_leak:
            raise ValueError('Question branch isolation failed')
        del together,separate,changed_outputs
        optimizer=torch.optim.AdamW(backend.trainable.values(),lr=1e-4,weight_decay=0.)
        updates=[]
        for step in range(3):
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
            loss,count=backend.pack_loss(pack);loss.backward()
            gradients=[p.grad for name,p in backend.trainable.items() if name.endswith('.b')]
            if len(gradients)!=196 or not all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gradients):
                raise ValueError('Missing or invalid multi-question gradients')
            norm=torch.nn.utils.clip_grad_norm_(list(backend.trainable.values()),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite training value')
            optimizer.step();check_frozen(backend.model,versions)
            updates.append(dict(step=step+1,loss=float(loss.detach()),tokens=count,seconds=time.perf_counter()-started,
                peak_gib=torch.cuda.max_memory_allocated()/2**30))
        optimizer.zero_grad(set_to_none=True);backend.save(directory/'adapter')
        with torch.no_grad():
            trained=[x.cpu() for x in backend.pack_logits(pack)]
            for module in backend.modules.values():
                module.b.zero_()
            backend.load(directory/'adapter')
            restored=[x.cpu() for x in backend.pack_logits(pack)]
        exact=all(torch.equal(a,b) for a,b in zip(trained,restored))
        if not exact:
            raise ValueError('Pack checkpoint restoration failed')
        status['checks'].update(updates=updates,serialization_exact=exact,frozen=check_frozen(backend.model,versions))
        status.update(status='passed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    main()
