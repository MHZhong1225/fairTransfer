"""Check actual visual rejection of unsupported claims on excluded portraits."""
import argparse
import json
import traceback

from common import STUDY,now,rows,sha,write
from annotate_facts_v3 import respond,verification_prompt,parse_verification

CLAIMS = [
    ('framing','visible face','The principal subject has a visible face.','face'),
    ('held_object','red umbrella','The person is visibly holding a large red umbrella overhead.','hands'),
    ('held_object','tennis racket','Both hands are visible holding a tennis racket.','hands'),
    ('background','bicycle','A full bicycle is clearly visible beside the person.','background'),
    ('accessory','opaque mask','The entire face is hidden behind a large opaque mask.','face'),
]


def main(judge):
    from PIL import Image
    import torch
    from scripts.umm_u_training_v1.review import _load_judge
    torch.set_num_threads(4)
    directory=STUDY/'results/fact_judge_controls_v1'/judge
    directory.mkdir(parents=True,exist_ok=False)
    plan=rows(STUDY/'data/technical_blind_manifest.jsonl')[:4]
    proposal=dict(facts=[dict(field=field,value=value,statement=statement,region=region,visibility='clear')
        for field,value,statement,region in CLAIMS])
    write(directory/'contract.json',dict(created_utc=now(),code_sha256=sha(__file__),judge=judge,
        images=[dict(id=r['id'],sha256=r['image_sha256']) for r in plan],claims=proposal,
        scope='Four excluded technical images visually inspected by the assistant: faces visible, umbrella/racket/bicycle/opaque face mask not visible. Not independent human gold and not a sensitivity estimate for subtle facts.',
        expected='Claim 0 YES. Claims 1-4 must not be YES; NO or UNCERTAIN is acceptable. No labels are supplied to the judge.',
        acceptance='All four positive claims accepted and no false positive among sixteen absent/unsupported claims.'))
    status=dict(status='loading',started_utc=now(),planned=4,attempted=0)
    write(directory/'status.json',status)
    try:
        model,processor=_load_judge(judge,'cuda:0');predictions=[]
        with (directory/'observations.jsonl').open('x',buffering=1) as stream:
            for item in plan:
                with Image.open(item['image_path']) as image:
                    raw,n_tokens,truncated=respond(model,processor,judge,image.convert('RGB'),verification_prompt(proposal),2048)
                parsed=parse_verification(raw,len(CLAIMS))
                result=dict(id=item['id'],raw=raw,parsed=parsed,output_tokens=n_tokens,truncated=truncated)
                predictions.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                status['attempted']+=1;write(directory/'status.json',status)
        positive=sum(next(r for r in p['parsed']['reviews'] if r['id']==0)['supported']=='YES' for p in predictions)
        false_positive=sum(r['supported']=='YES' for p in predictions for r in p['parsed']['reviews'] if r['id']!=0)
        status.update(status='passed' if positive==4 and false_positive==0 else 'failed_criterion',
            positive_accepted=positive,positive_planned=4,false_positive=false_positive,negative_planned=16,completed_utc=now())
        if status['status']!='passed':
            raise RuntimeError('Visual fact judge did not pass the prespecified obvious-control check')
    except BaseException:
        status.update(error=traceback.format_exc())
        if status['status']!='failed_criterion':
            status['status']='failed'
        raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','phi'))
    main(parser.parse_args().judge)
