"""Repeat the frozen obvious visual controls in independent batched conversations."""
from pathlib import Path
import json
import traceback
import torch
from PIL import Image
from common import STUDY,now,rows,sha,write
from qualify_fact_judges import CLAIMS
from annotate_facts_v3 import verification_prompt,parse_verification
from ministral_judge import load
from ministral_batch import respond_batch


def main():
    torch.set_num_threads(4)
    directory=STUDY/'results/fact_judge_controls_v3/ministral_batch'
    directory.mkdir(parents=True,exist_ok=False)
    plan=rows(STUDY/'data/technical_blind_manifest.jsonl')[:4]
    facts=[dict(field=f,value=v,statement=s,region=r,visibility='clear') for f,v,s,r in CLAIMS]
    prompt=verification_prompt(dict(facts=facts))
    write(directory/'contract.json',dict(created_utc=now(),images=plan,claims=facts,batch_size=4,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('qualify_ministral_batch.py','ministral_judge.py','ministral_batch.py','annotate_facts_v3.py')},
        acceptance='Same 4 positive and 16 negative claims as v1, all positive YES and no negative YES.'))
    status=dict(status='running',started_utc=now());write(directory/'status.json',status)
    try:
        model,tokenizer=load()
        images=[Image.open(r['image_path']).convert('RGB') for r in plan]
        outputs=respond_batch(model,tokenizer,images,[prompt]*4)
        predictions=[dict(id=item['id'],raw=raw,parsed=parse_verification(raw,5)) for item,raw in zip(plan,outputs)]
        with (directory/'observations.jsonl').open('x') as stream:
            for result in predictions:
                stream.write(json.dumps(result)+'\n')
        positive=sum(next(r for r in p['parsed']['reviews'] if r['id']==0)['supported']=='YES' for p in predictions)
        negative=sum(r['supported']=='YES' for p in predictions for r in p['parsed']['reviews'] if r['id']!=0)
        status.update(positive_accepted=positive,false_positive=negative)
        if positive!=4 or negative!=0:
            raise ValueError('Batched obvious controls failed')
        status.update(status='passed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    main()
