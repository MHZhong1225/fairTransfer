"""Retain all raw control outputs while diagnosing strict quotation failures."""
import argparse
import os
from pathlib import Path
import subprocess
import traceback
from common import STUDY,now,rows,sha,write
from grading import U_INSTRUCTIONS,understanding_prompt,parse_understanding
import json

QUOTE_RULE='''
QUOTATION RULE: The Model response below is the only source of quotes. A quote must be a continuous exact substring, with no added subject, grammatical repair, omitted middle words or ellipsis. For example, if a response says "A person sits and holds a cup.", "holds a cup" is an exact quote but "A person holds a cup" is not. Copy text before assigning a status. Never quote the reference list as if it were the response. If a fact is not expressed, mark it OMITTED instead of inventing a quotation.
'''

COMPACT='''Evaluate the TARGET ANSWER, not the reference list. Do not answer the question yourself.
For each reference id, judge whether the TARGET ANSWER states it (STATED), explicitly contradicts it (CONTRADICTED), or does not mention it (OMITTED). Use UNCERTAIN only for undecidable correspondence. Copy an exact substring of TARGET ANSWER for STATED/CONTRADICTED; OMITTED must have an empty quote. A reference being visible in the image does not mean the answer states it.
Then list only specific visual claims actually asserted in TARGET ANSWER. Verify each claim against the image: supported must be YES, NO, or UNCERTAIN. Do not add references or visible details absent from the answer. An empty answer or an explicit abstention makes no claims. A wrong answer can have false claims; never replace them with the correct facts. Appearance is visible presentation/approximate age/skin tone, not identity or ethnicity; face/clothes alone do not prove occupation.
Return JSON with exactly these two arrays:
{"reference_reviews":[{"id":0,"status":"OMITTED","quote":""}],"claims":[{"quote":"exact answer substring","statement":"one assertion from that substring","supported":"YES","evidence":"brief image evidence"}]}
Include each supplied reference id exactly once. Empty claims must be []. All quote text must come verbatim from TARGET ANSWER, never from REFERENCE FACTS.'''


def run(judge):
    from PIL import Image
    import torch
    from remote_judge_runtime import load
    from judge_runtime import respond
    torch.set_num_threads(4)
    directory=STUDY/'results/scoring_quote_diagnostic_v2'/judge
    directory.mkdir(parents=True,exist_ok=False)
    item=rows(STUDY/'data/scene_technical_v1/manifest.jsonl')[2]
    assert sha(item['image_path'])==item['image_sha256']
    image=Image.open(item['image_path']).convert('RGB')
    facts=[dict(field='spatial_relation',value='laptop in front of the person',statement='A laptop is visible in front of the person.'),
        dict(field='accessory',value='glasses',statement='The person is wearing glasses.')]
    case=dict(question='Describe the visible person and scene.',reference_facts=facts)
    correct='A laptop is visible in front of the person. The person is wearing glasses.'
    responses=[correct,'The person is riding a bicycle and has no glasses.','',correct+' The person is holding a large red umbrella.']
    write(directory/'contract.json',dict(judge=judge,image_sha256=item['image_sha256'],case=case,responses=responses,
        extra_rule=QUOTE_RULE,source_sha256=sha(__file__),grading_sha256=sha(Path(__file__).with_name('grading.py')),
        scope='Diagnostic controls only, before formal training/evaluation. Same strict parser for both prompt versions; retain every raw output.'))
    status=dict(status='running',pid=os.getpid(),started_utc=now());write(directory/'status.json',status)
    try:
        model,processor=load(judge)
        for version in ('reordered','compact'):
            references=[dict(id=i,**f) for i,f in enumerate(facts)]
            instructions=U_INSTRUCTIONS if version=='reordered' else COMPACT
            prompts=[instructions+'\nREFERENCE FACTS (these are not the answer):\n'+json.dumps(references)+
                '\nQUESTION:\n'+case['question']+'\nTARGET ANSWER TO GRADE (JSON string):\n'+json.dumps(r)+
                '\nGrade only that TARGET ANSWER. Never attribute the reference list or image-only facts to it. An empty string has no claims and omits every reference.' for r in responses]
            outputs=respond(model,processor,judge,[image]*4,prompts)
            for i,(raw,response) in enumerate(zip(outputs,responses)):
                record=dict(raw=raw,prompt=prompts[i],response=response)
                write(directory/f'{version}_{i}.json',record)
                try:
                    record.update(status='parsed',parsed=parse_understanding(raw,case,response))
                except Exception as error:
                    record.update(status='parse_failed',error=str(error))
                write(directory/f'{version}_{i}.json',record)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral','both'))
    judge=parser.parse_args().judge
    if judge=='both':
        for name in ('qwen','ministral'):
            subprocess.run([str(STUDY/'envs'/name/'bin/python'),__file__,name],check=True)
    else:
        run(judge)
