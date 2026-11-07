"""Blind image fact proposals and separate per-fact visual verification."""
import argparse
import json
import re
import time
import traceback

from common import STUDY, now, rows, read, sha, write

FIELDS = ('presentation', 'age_band', 'skin_tone', 'framing', 'posture', 'activity',
          'held_object', 'clothing_type', 'clothing_color', 'hair', 'background',
          'accessory', 'spatial_relation')
REGIONS = ('face', 'hair', 'upper_body', 'hands', 'lower_body', 'background', 'whole_image')
CATEGORIES = {
    'presentation': ('feminine', 'masculine'),
    'age_band': ('young_adult', 'middle_aged_adult', 'older_adult'),
    'skin_tone': ('light', 'medium', 'dark'),
    'framing': ('close_portrait', 'upper_body', 'full_body'),
}
PROPOSAL_PROMPT = '''Describe only visually supported facts about the principal person in this image.
You have no source occupation or demographic labels. Do not guess their occupation, identity,
ethnicity, nationality, exact age, personality or intentions. A uniform alone does not establish
what work is being performed. Posing is not operating machinery; never invent hidden hands,
tools, patients or an activity because a workplace is suggested. Background blur is not evidence
for a specific building. Mark uncertain or invisible fields missing rather than guessing.

Return one JSON object with keys people (ONE/MULTIPLE/NONE/UNCERTAIN),
adult_subject (YES/NO/UNCERTAIN), work_action_visible (YES/NO/UNCERTAIN),
tool_visible (YES/NO/UNCERTAIN), and facts (a list).
Each fact has exactly: field, value, statement, region, visibility.
Use at most 20 facts. Controlled demographic and framing fields occur at most once;
ordinary fields may repeat for different visible objects or different concrete details.
Do not repeat the same fact. Allowed fields:
presentation, age_band, skin_tone, framing, posture, activity, held_object,
clothing_type, clothing_color, hair, background, accessory, spatial_relation.
value is a concise answer, at most 10 words. statement is one standalone literal visual
statement about the principal person or scene, at most 20 words. region must be one of
face/hair/upper_body/hands/lower_body/background/whole_image.
visibility must be clear or partial. Do not include uncertain facts.

For presentation, value must be feminine or masculine: this means visible styling only,
never gender identity or biological sex. Use the wording feminine/masculine presentation
in its statement. For age_band use young_adult/middle_aged_adult/older_adult only when the
face clearly supports a broad apparent-age impression; state apparent age, never actual age.
For skin_tone use light/medium/dark only when exposed skin and lighting permit a reliable
coarse visible tone; never infer ethnicity. For framing use close_portrait/upper_body/full_body.
For all other fields use short concrete English. Do not put demographic descriptions into
non-demographic fields. Report an actually visible action; if merely posing, say posing.
For held_object, omission is preferable when hands are cropped. Clothing and hair can be
described even when no work action exists. A spatial relation requires two visible entities.
Include all supported fields, but do not fabricate a fact to fill the list.'''


def json_object(raw):
    text = raw.strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if fence:
        text = fence.group(1)
    # Accept escaped line breaks only outside strings, as in the preserved Phi interface.
    converted, inside, escaped, index = [], False, False, 0
    while index < len(text):
        if not inside and text[index:index+2] == '\\n':
            converted.append('\n'); index += 2; continue
        char = text[index]; converted.append(char)
        if char == '"' and not escaped:
            inside = not inside
        escaped = bool(inside and char == '\\' and not escaped)
        index += 1
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    result = json.loads(''.join(converted), object_pairs_hook=unique_pairs)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def parse_proposal(raw):
    result = json_object(raw)
    if set(result) != {'people','adult_subject','work_action_visible','tool_visible','facts'}:
        raise ValueError('Wrong proposal fields')
    if result['people'] not in ('ONE','MULTIPLE','NONE','UNCERTAIN'):
        raise ValueError('Wrong people category')
    for key in ('adult_subject','work_action_visible','tool_visible'):
        if result[key] not in ('YES','NO','UNCERTAIN'):
            raise ValueError('Wrong observation category')
    if not isinstance(result['facts'], list) or len(result['facts']) > 20:
        raise ValueError('Wrong fact list')
    seen = set()
    for fact in result['facts']:
        if set(fact) != {'field','value','statement','region','visibility'}:
            raise ValueError('Wrong fact fields')
        field = fact['field']
        if field not in FIELDS or (field in CATEGORIES and field in seen):
            raise ValueError('Invalid or repeated fact field')
        seen.add(field)
        if fact['region'] not in REGIONS or fact['visibility'] not in ('clear','partial'):
            raise ValueError('Invalid evidence region or visibility')
        if any(not isinstance(fact[k], str) or not fact[k].strip() for k in ('value','statement')):
            raise ValueError('Empty or invalid fact text')
        if field in CATEGORIES and fact['value'] not in CATEGORIES[field]:
            raise ValueError('Invalid controlled category')
    return result


def verification_prompt(proposal):
    claims = [dict(id=i, field=f['field'], value=f['value'], statement=f['statement'], region=f['region'])
              for i, f in enumerate(proposal['facts'])]
    return ('''Independently inspect this image. The following proposed claims may be wrong.
For each claim, check the stated value and every detail in the statement against visible evidence.
Do not assume claims are correct. Use NO for contradicted content, UNCERTAIN for insufficient
evidence, and YES only when supported. Visible presentation is not identity, apparent age is
not actual age, and skin tone is not ethnicity. Be conservative with age and skin under lighting.
Never accept work actions or held tools on the basis of clothing or an occupation stereotype.
A scene, exact hand, object identity, or relation must actually be visible to accept it.
Return only one JSON object with keys people (ONE/MULTIPLE/NONE/UNCERTAIN),
adult_subject, work_action_visible, tool_visible (each YES/NO/UNCERTAIN), reviews (a list).
Each review has exactly id (integer), supported (YES/NO/UNCERTAIN), evidence (a short sentence).
Include every proposed id exactly once. No source labels or target model outputs are provided.
Proposed claims:\n''' + json.dumps(claims, ensure_ascii=False))


def parse_verification(raw, count):
    result = json_object(raw)
    if set(result) != {'people','adult_subject','work_action_visible','tool_visible','reviews'}:
        raise ValueError('Wrong verification fields')
    if result['people'] not in ('ONE','MULTIPLE','NONE','UNCERTAIN'):
        raise ValueError('Wrong people category')
    for key in ('adult_subject','work_action_visible','tool_visible'):
        if result[key] not in ('YES','NO','UNCERTAIN'):
            raise ValueError('Wrong observation category')
    reviews = result['reviews']
    if not isinstance(reviews,list) or len(reviews) != count:
        raise ValueError('Missing fact reviews')
    if any(set(r) != {'id','supported','evidence'} for r in reviews):
        raise ValueError('Wrong review fields')
    if any(type(r['id']) is not int for r in reviews) or {r['id'] for r in reviews} != set(range(count)):
        raise ValueError('Wrong or repeated fact ids')
    if any(r['supported'] not in ('YES','NO','UNCERTAIN') or not isinstance(r['evidence'],str) for r in reviews):
        raise ValueError('Invalid review outcome')
    return result


def respond(model, processor, judge, image, prompt, limit):
    import torch
    if judge == 'qwen':
        messages = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=limit)
        ids = output[0,inputs.input_ids.shape[1]:]
        return processor.tokenizer.decode(ids,skip_special_tokens=True), len(ids), bool(len(ids)>=limit)
    from transformers.cache_utils import DynamicCache
    prompt = processor.tokenizer.apply_chat_template([{'role':'user','content':'<|image_1|>\n'+prompt}],
        tokenize=False,add_generation_prompt=True)
    inputs = processor(prompt,[image],return_tensors='pt').to(model.device)
    cache, tokens = DynamicCache(), []
    with torch.inference_mode():
        output = model(**inputs,past_key_values=cache,use_cache=True)
        for _ in range(limit):
            token = output.logits[:,-1:].argmax(dim=-1)
            if token.item() == processor.tokenizer.eos_token_id:
                break
            tokens.append(token.item())
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'],inputs['attention_mask'].new_ones((1,1))],dim=-1)
            output = model(input_ids=token,attention_mask=inputs['attention_mask'],past_key_values=cache,use_cache=True)
    return processor.tokenizer.decode(tokens,skip_special_tokens=True),len(tokens),len(tokens)>=limit


def completed_lines(path):
    if not path.exists():
        return []
    content = path.read_text()
    return [json.loads(line) for line in content.split('\n')[:-1] if line]


def run(stage, task, limit=None):
    from PIL import Image
    from scripts.umm_u_training_v1.review import _load_judge
    import torch
    torch.set_num_threads(4)
    manifest = STUDY / 'data' / f'{"technical" if stage=="technical" else "candidate"}_blind_manifest.jsonl'
    plan = rows(manifest)
    if limit is not None:
        plan = plan[:limit]
    directory = STUDY / 'results' / ('fact_annotation_' + stage + '_v2') / task
    directory.mkdir(parents=True,exist_ok=False)
    judge = 'qwen' if task == 'propose' else 'phi'
    write(directory/'contract.json',dict(created_utc=now(),source_sha256=sha(__file__),manifest_sha256=sha(manifest),
        model=judge,planned=len(plan),limit=limit,stage=stage,decoding='greedy',max_new_tokens=2048,
        proposal_prompt=PROPOSAL_PROMPT,blindness='Only PIL image pixels and fixed task enter proposer. Verifier also sees candidate facts, but no source labels or training state. Agreement is a machine proxy, not human gold.'))
    status = dict(status='loading',judge=judge,planned=len(plan),attempted=0,started_utc=now())
    write(directory/'status.json',status)
    try:
        model,processor = _load_judge(judge,'cuda:0')
        status['status'] = 'running'
        with (directory/'observations.jsonl').open('x',buffering=1) as stream:
            proposals = {}
            for item in plan:
                if task == 'verify':
                    parent = directory.parent/'propose'
                    while item['id'] not in proposals:
                        proposals = {r['id']:r for r in completed_lines(parent/'observations.jsonl')}
                        if item['id'] in proposals:
                            break
                        if (parent/'status.json').exists() and read(parent/'status.json')['status'] in ('failed','completed'):
                            raise RuntimeError('Proposal producer ended without this planned record')
                        time.sleep(5)
                    proposed = proposals[item['id']]
                    if proposed['status'] != 'completed':
                        result = dict(id=item['id'],status='missing_proposal',proposal_status=proposed['status'])
                        stream.write(json.dumps(result)+'\n')
                        status['attempted'] += 1; write(directory/'status.json',status); continue
                result = dict(id=item['id'],started_utc=now())
                try:
                    if sha(item['image_path']) != item['image_sha256']:
                        raise ValueError('Source image changed')
                    with Image.open(item['image_path']) as image:
                        source = image.convert('RGB').copy()
                    prompt = PROPOSAL_PROMPT if task=='propose' else verification_prompt(proposed['parsed'])
                    raw,n_tokens,truncated = respond(model,processor,judge,source,prompt,2048)
                    result.update(raw=raw,output_tokens=n_tokens,truncated=truncated,prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest())
                    try:
                        parsed = parse_proposal(raw) if task=='propose' else parse_verification(raw,len(proposed['parsed']['facts']))
                        result.update(parsed=parsed,status='completed')
                    except (ValueError,TypeError,KeyError):
                        result.update(status='parse_failed',error=traceback.format_exc())
                except Exception:
                    result.update(status='runtime_failed',error=traceback.format_exc())
                result['completed_utc'] = now()
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                status['attempted'] += 1; status['last_status'] = result['status']
                write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc()); raise
    finally:
        write(directory/'status.json',status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('technical','candidate'))
    parser.add_argument('task',choices=('propose','verify'))
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    run(args.stage,args.task,args.limit)
