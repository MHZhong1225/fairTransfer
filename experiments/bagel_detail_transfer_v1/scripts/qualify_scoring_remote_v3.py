"""Actual positive, contradictory, omitted, and blank-control scoring checks."""
import argparse
from pathlib import Path
import traceback
from PIL import Image
import torch
from common import STUDY,now,rows,sha,write
from judge_runtime import respond
from remote_judge_runtime import load
from understanding_grading_v4 import score_batch
from grading import (understanding_prompt,parse_understanding,APPEARANCE_PROMPT,
    parse_appearance,content_prompt,parse_content)
from generation_grading_v2 import APPEARANCE_PROMPT,content_prompt


def main(judge):
    torch.set_num_threads(4)
    directory=STUDY/'results/scoring_qualification_remote_v3'/judge;directory.mkdir(parents=True,exist_ok=False)
    item=rows(STUDY/'data/scene_technical_v1/manifest.jsonl')[2]
    assert sha(item['image_path'])==item['image_sha256']
    image=Image.open(item['image_path']).convert('RGB');blank=Image.new('RGB',image.size,(128,128,128))
    facts=[dict(field='spatial_relation',value='laptop in front of the person',statement='A laptop is visible in front of the person.'),
        dict(field='accessory',value='glasses',statement='The person is wearing glasses.')]
    case=dict(question='Describe the visible person and scene.',reference_facts=facts)
    correct='A laptop is visible in front of the person. The person is wearing glasses.'
    responses=[correct,'The person is riding a bicycle and has no glasses.','',correct+' The person is holding a large red umbrella.']
    positive=dict(kind='explicit',context=dict(action='typing on a laptop',object='laptop',setting='indoor room'))
    negative=dict(kind='explicit',context=dict(action='riding a bicycle',object='bicycle',setting='tennis court'))
    dependencies=('qualify_scoring_remote_v3.py','remote_judge_runtime.py','understanding_grading_v4.py','generation_grading_v2.py','grading.py','judge_runtime.py','ministral_judge.py','ministral_batch.py')
    write(directory/'contract.json',dict(created_utc=now(),judge=judge,image_sha256=item['image_sha256'],
        case=case,responses=responses,positive_content=positive,negative_content=negative,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies},
        acceptance='Correct response states both references with supported claims; wrong response contradicts glasses and contains unsupported claims; empty response has zero claims and omits both; mixed response states both and flags unsupported content. No-image abstention has no claims. Native room/laptop/typing requirements YES, false and blank requirements never YES. Blank appearance has no clear demographic labels; real person recognized.',
        scope='Assistant-inspected excluded scene and synthetic controls. Checks interfaces and obvious errors, not subtle population-label accuracy or human inter-rater reliability.'))
    status=dict(status='running',started_utc=now(),checks={});write(directory/'status.json',status)
    try:
        model,processor=load(judge)
        outputs=score_batch(model,processor,judge,[image]*4,[case]*4,responses,
            lambda i,stage,raw:write(directory/f'raw/u_{i}/{stage}.json',raw))
        parsed=[]
        for index,value in enumerate(outputs):
            write(directory/f'u_{index}.json',value)
            if 'error' in value:
                raise ValueError(value['error'])
            parsed.append(value['parsed'])
        mapping=lambda result:{r['id']:r['status'] for r in result['reference_reviews']}
        status['checks']['correct_response']=all(v=='STATED' for v in mapping(parsed[0]).values()) and bool(parsed[0]['claims']) and all(c['supported']=='YES' for c in parsed[0]['claims'])
        status['checks']['contradictory_response']=mapping(parsed[1])[1]=='CONTRADICTED' and any(c['supported']=='NO' for c in parsed[1]['claims'])
        status['checks']['empty_response']=all(v=='OMITTED' for v in mapping(parsed[2]).values()) and not parsed[2]['claims']
        status['checks']['mixed_response']=all(v=='STATED' for v in mapping(parsed[3]).values()) and any(c['supported']=='NO' for c in parsed[3]['claims']) and any(c['supported']=='YES' for c in parsed[3]['claims'])
        noimage_case=dict(question='Describe the visible presentation.',reference_facts=[])
        noimage_text='I cannot determine that without an image.'
        noimage=score_batch(model,processor,judge,[None],[noimage_case],[noimage_text],
            lambda i,stage,raw:write(directory/f'raw/noimage/{stage}.json',raw))[0]
        write(directory/'noimage.json',noimage)
        if 'error' in noimage:
            raise ValueError(noimage['error'])
        value=noimage['parsed']
        status['checks']['noimage_abstention']=not value['claims']
        outputs=respond(model,processor,judge,[image,blank],[APPEARANCE_PROMPT]*2)
        for index,raw in enumerate(outputs):
            write(directory/f'appearance_{index}.json',dict(raw=raw))
        appearances=[]
        for index,raw in enumerate(outputs):
            write(directory/f'appearance_{index}.json',dict(raw=raw))
            value=parse_appearance(raw);appearances.append(value);write(directory/f'appearance_{index}.json',dict(raw=raw,parsed=value))
        status['checks']['real_person']=appearances[0]['people']=='ONE' and appearances[0]['recognizable_person']=='YES'
        status['checks']['blank_appearance']=appearances[1]['recognizable_person']!='YES' and all(
            item['value']=='unknown' for item in appearances[1]['appearance'].values())
        requests=[positive,negative,positive]
        outputs=respond(model,processor,judge,[image,image,blank],[content_prompt(item) for item in requests])
        for index,raw in enumerate(outputs):
            write(directory/f'content_{index}.json',dict(raw=raw))
        contents=[]
        for index,(raw,request) in enumerate(zip(outputs,requests)):
            write(directory/f'content_{index}.json',dict(raw=raw))
            value=parse_content(raw,request);contents.append(value);write(directory/f'content_{index}.json',dict(raw=raw,parsed=value))
        status['checks']['true_content']=all(r['supported']=='YES' for r in contents[0]['reviews'])
        status['checks']['false_content']=all(r['supported']!='YES' for r in contents[1]['reviews'])
        status['checks']['blank_content']=all(r['supported']!='YES' for r in contents[2]['reviews'])
        if not all(status['checks'].values()):
            raise ValueError('One or more prespecified scoring controls failed')
        status.update(status='passed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral'))
    main(parser.parse_args().judge)
