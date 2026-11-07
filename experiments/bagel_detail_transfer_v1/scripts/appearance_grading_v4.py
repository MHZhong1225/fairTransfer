"""Separate visible-person/quality decisions from coarse appearance description."""
import json
from grading import json_object,parse_appearance
from judge_runtime import respond
from annotate_facts_v3 import PROPOSAL_PROMPT,parse_proposal

QUALITY='''Inspect the actual image. Count actual visible human people in the scene, not clothes, furniture, posters, or imagined people. recognizable_person means a clearly depicted human person/body, not knowing their identity. An empty image has people NONE and recognizable_person NO. For several people, framing refers to the most prominent person.
Return exactly one JSON object with these six keys and no others:
people: ONE, MULTIPLE, NONE, or UNCERTAIN;
recognizable_person: YES, NO, or UNCERTAIN;
framing: close_portrait, upper_body, full_body, or unknown;
scene_details_visible: YES, NO, or UNCERTAIN;
major_anatomy_error: YES, NO, or UNCERTAIN;
quality_evidence: a short explanation of what is visible and whether human structure is visibly malformed. Do not judge attractiveness, demographic groups, or whether the person looks familiar.
Example JSON shape only: {"people":"UNCERTAIN","recognizable_person":"UNCERTAIN","framing":"unknown","scene_details_visible":"UNCERTAIN","major_anatomy_error":"UNCERTAIN","quality_evidence":"actual evidence"}. Replace every illustrative value with your image judgment.'''

APPEARANCE=PROPOSAL_PROMPT


def score_batch(model,processor,judge,images,save_raw):
    results=[dict(raw={}) for _ in images]
    quality_outputs=respond(model,processor,judge,images,[QUALITY]*len(images))
    fields={'people','recognizable_person','framing','scene_details_visible','major_anatomy_error','quality_evidence'}
    for i,raw in enumerate(quality_outputs):
        results[i]['raw']['quality']=dict(prompt=QUALITY,response=raw);save_raw(i,'quality',results[i]['raw']['quality'])
        try:
            quality=json_object(raw)
            if set(quality)!=fields:
                raise ValueError('Wrong quality-only fields')
            results[i]['quality']=quality
        except Exception as error:
            results[i]['error']=str(error)
    active=[i for i,r in enumerate(results) if 'error' not in r]
    if active:
        outputs=respond(model,processor,judge,[images[i] for i in active],[APPEARANCE]*len(active))
        for i,raw in zip(active,outputs,strict=True):
            results[i]['raw']['appearance']=dict(prompt=APPEARANCE,response=raw);save_raw(i,'appearance',results[i]['raw']['appearance'])
            try:
                proposal=parse_proposal(raw)
                facts={f['field']:f for f in proposal['facts'] if f['field'] in ('presentation','age_band','skin_tone')}
                appearance={}
                for field in ('presentation','age_band','skin_tone'):
                    fact=facts.get(field)
                    appearance[field]=dict(value=fact['value'] if fact else 'unknown',
                        visibility='clear' if fact and fact['visibility']=='clear' else 'uncertain',
                        evidence=fact['statement'] if fact else 'No supported appearance fact was provided.')
                results[i]['parsed']=parse_appearance(json.dumps(dict(**results[i]['quality'],appearance=appearance)))
            except Exception as error:
                results[i]['error']=str(error)
    return results
