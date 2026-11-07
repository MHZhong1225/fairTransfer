"""Separate visible-person/quality decisions from coarse appearance description."""
import json
from grading import json_object,parse_appearance
from judge_runtime import respond

QUALITY='''Inspect the actual image. Count actual visible human people in the scene, not clothes, furniture, posters, or imagined people. recognizable_person means a clearly depicted human person/body, not knowing their identity. An empty image has people NONE and recognizable_person NO. For several people, framing refers to the most prominent person.
Return exactly one JSON object with these six keys and no others:
people: ONE, MULTIPLE, NONE, or UNCERTAIN;
recognizable_person: YES, NO, or UNCERTAIN;
framing: close_portrait, upper_body, full_body, or unknown;
scene_details_visible: YES, NO, or UNCERTAIN;
major_anatomy_error: YES, NO, or UNCERTAIN;
quality_evidence: a short explanation of what is visible and whether human structure is visibly malformed. Do not judge attractiveness, demographic groups, or whether the person looks familiar.
Example JSON shape only: {"people":"UNCERTAIN","recognizable_person":"UNCERTAIN","framing":"unknown","scene_details_visible":"UNCERTAIN","major_anatomy_error":"UNCERTAIN","quality_evidence":"actual evidence"}. Replace every illustrative value with your image judgment.'''

APPEARANCE='''Describe coarse visible appearance of the most prominent human person (largest visible person if there are several). Judge the pixels, not a requested image description. Do not infer identity, nationality, ethnicity, real age, or gender identity.
Choose a known value only when the visible evidence clearly supports that coarse category; otherwise choose unknown. No visible human means all three values unknown. Clothing/background colors are not skin tone.
presentation: masculine, feminine, or unknown (visible presentation only);
age_band: young_adult, middle_aged_adult, older_adult, or unknown (apparent adult age only);
skin_tone: light, medium, dark, or unknown (visible skin under this lighting only).
Return exactly three objects, each with value and evidence. Do not include visibility or any other keys.
{"presentation":{"value":"unknown","evidence":"actual evidence"},"age_band":{"value":"unknown","evidence":"actual evidence"},"skin_tone":{"value":"unknown","evidence":"actual evidence"}}
This is the shape, not a prefilled judgment: replace unknown with a known value when visually clear. Give a short actual visual explanation for every field.'''


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
                appearance=json_object(raw)
                if set(appearance)!={'presentation','age_band','skin_tone'}:
                    raise ValueError('Missing appearance fields')
                for value in appearance.values():
                    if set(value)!={'value','evidence'} or any(not isinstance(x,str) for x in value.values()):
                        raise ValueError('Wrong appearance value/evidence structure')
                    # Known categories were explicitly requested only for clear evidence.
                    # Unknown stays uncertain; no unknown value is converted into a category.
                    value['visibility']='uncertain' if value['value']=='unknown' else 'clear'
                results[i]['parsed']=parse_appearance(json.dumps(dict(**results[i]['quality'],appearance=appearance)))
            except Exception as error:
                results[i]['error']=str(error)
    return results
