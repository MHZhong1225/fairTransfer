"""Derive paired supervision solely from visually verified facts."""
from collections import defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations
import re

DEMOGRAPHIC = ('presentation','age_band','skin_tone')
DETAILED_QUESTIONS = (
    'Describe the visible person, clothing, activity, objects and setting accurately.',
    'Give a detailed description of what is visibly present in this image.',
    'Describe this image using only details that are supported by the picture.',
)
BRIEF_QUESTIONS = (
    'Give a brief, neutral description of the visible person or activity.',
    'Summarize the visible activity in one short sentence.',
    'Briefly describe what the person is visibly doing, without guessing their job.',
)
QUESTIONS = {
    'presentation':('How would you describe the person’s visible presentation, if it is clear?',
        'What visible presentation does the person have? Describe appearance rather than identity.'),
    'age_band':('What broad apparent age group does the person seem to belong to?',
        'Describe the person’s apparent age in a broad adult age group, if visible.'),
    'skin_tone':('What coarse skin tone is visible on the person, if lighting permits judging it?',
        'Describe the visible skin tone without inferring ethnicity.'),
    'activity':('Describe the visible activity, held objects and spatial relations.',
        'What is the person visibly doing, and what objects or relations support that description?'),
    'appearance':('Describe the visible clothing, hair and accessories.',
        'What clothing, hair details or accessories are actually visible?'),
    'scene':('Describe the visible framing, posture and background.',
        'What does the image show about the person’s posture and surroundings?'),
    'ordinary':('Describe the visible activity, objects, clothing and scene accurately.',
        'What ordinary visual details can you verify about the person and surroundings?'),
}
DEMOGRAPHIC_STATEMENTS = {
    'presentation':{'feminine':'The person has feminine presentation.','masculine':'The person has masculine presentation.'},
    'age_band':{'young_adult':'The person appears to be a young adult.',
        'middle_aged_adult':'The person appears to be a middle-aged adult.',
        'older_adult':'The person appears to be an older adult.'},
    'skin_tone':{value:f'The person’s visible skin tone is {value}.' for value in ('light','medium','dark')},
}


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)


def neutral_statement(text):
    """Remove gendered references while retaining the verified ordinary claim."""
    text=re.sub(r'\b(?:His|Her|Their)\b','The person’s',text)
    text=re.sub(r'\b(?:his|her|their)\b','the person’s',text)
    text=re.sub(r'\b(?:He|She)\b','The person',text)
    text=re.sub(r'\b(?:he|she)\b','the person',text)
    text=re.sub(r'\b(?:man|woman|gentleman|lady|boy|girl)\b','person',text,flags=re.I)
    # Do not remove arbitrary adjectives, which could change an ordinary fact.
    # Explicit population descriptors outside their fields remain excluded/audited.
    if re.search(r'\b(?:male|female|masculine|feminine|young|elderly|middle[- ]aged|older adult|skin tone|skin color)\b',text,re.I):
        return None
    return text.strip()


def verified_facts(proposal,verification):
    reviews={r['id']:r for r in verification['reviews']}
    accepted=[];rejected=[]
    for index,original in enumerate(proposal['facts']):
        fact=dict(original,source_fact_id=index,verification=reviews[index])
        reason=None
        if reviews[index]['supported']!='YES':
            reason='not_independently_confirmed'
        elif original['field'] in DEMOGRAPHIC and original['visibility']!='clear':
            reason='population_appearance_not_clear'
        elif original['field'] in DEMOGRAPHIC:
            fact['statement']=DEMOGRAPHIC_STATEMENTS[original['field']][original['value']]
        else:
            text=neutral_statement(original['statement'])
            if text is None:
                reason='population_descriptor_in_ordinary_field'
            else:
                fact['statement']=text
        if reason:
            rejected.append(dict(fact=fact,reason=reason))
        elif fact['statement'] not in {f['statement'] for f in accepted}:
            accepted.append(fact)
    return accepted,rejected


def caption(facts):
    return ' '.join(f['statement'] for f in facts)


def select_paired_facts(identifier,facts,token_count):
    ordinary=[f for f in facts if f['field'] not in DEMOGRAPHIC]
    population=[f for f in facts if f['field'] in DEMOGRAPHIC]
    if not ordinary:
        return None
    core=[]
    for fields in (('activity','posture'),('held_object',),('background',)):
        found=next((f for field in fields for f in ordinary if f['field']==field),None)
        if found and found not in core:
            core.append(found)
    if not core:
        core=[ordinary[0]]
    extras=[f for f in ordinary if f not in core]
    # Fixed rotation avoids always omitting the same attribute when budgets are small.
    population.sort(key=lambda f:number(identifier+'/'+f['field']))
    count=min(6,len(ordinary));slots=count-len(core)
    population=population[:slots]
    remaining=slots-len(population)
    shared_extras=sorted(extras,key=lambda f:(f['field'],f['statement']))[:remaining]
    common=core+shared_extras
    options=[f for f in extras if f not in shared_extras]
    t2=common+population
    if not population:
        return dict(T1=t2,T2=t2,common_fact_ids=[f['source_fact_id'] for f in common],demographic_inserted=0)
    target=token_count(caption(t2))
    candidates=combinations(options,len(population))
    replacements=min(candidates,key=lambda selected:(abs(token_count(caption(common+list(selected)))-target),
        tuple(f['source_fact_id'] for f in selected)))
    t1=common+list(replacements)
    return dict(T1=t1,T2=t2,common_fact_ids=[f['source_fact_id'] for f in common],
        demographic_inserted=len(population),T1_tokens=token_count(caption(t1)),T2_tokens=target)


def brief_caption(facts):
    for field in ('activity','posture'):
        fact=next((f for f in facts if f['field']==field),None)
        if fact:
            return fact['statement']
    return 'A person is shown in the image.'


def question_tasks(identifier,facts,epoch):
    """Two to four isolated questions, together covering exactly the T2 facts."""
    groups=defaultdict(list)
    population=[f for f in facts if f['field'] in DEMOGRAPHIC]
    for fact in population:
        groups[fact['field']].append(fact)
    ordinary=[f for f in facts if f['field'] not in DEMOGRAPHIC]
    if len(population)==3:
        groups['ordinary']=ordinary
    else:
        for fact in ordinary:
            field=fact['field']
            group='activity' if field in ('activity','held_object','spatial_relation') else (
                'appearance' if field in ('clothing_type','clothing_color','hair','accessory') else 'scene')
            groups[group].append(fact)
        while len(groups)>4:
            merge=[key for key in ('scene','appearance','activity') if key in groups]
            groups[merge[1]].extend(groups.pop(merge[0]))
    tasks=[]
    for field,selected in groups.items():
        if not selected:
            continue
        # A merged ordinary question asks for all ordinary fields, not just its old name.
        question_key=field if field in DEMOGRAPHIC or all(
            (f['field'] in ('activity','held_object','spatial_relation') if field=='activity' else
             f['field'] in ('clothing_type','clothing_color','hair','accessory') if field=='appearance' else
             f['field'] in ('framing','posture','background')) for f in selected) else 'ordinary'
        templates=QUESTIONS[question_key]
        tasks.append(dict(question=templates[number(f'{identifier}/{epoch}/{field}')%len(templates)],
            caption=caption(selected),fact_ids=[f['source_fact_id'] for f in selected],kind='qa',field=field))
    return tasks


def make_pack(row,arm,epoch):
    identifier=row['id'];brief=number(f'{identifier}/{epoch}/brief')%5==0
    question_index=number(f'{identifier}/{epoch}/question')%3
    if brief or arm=='T0':
        question=(BRIEF_QUESTIONS if brief else DETAILED_QUESTIONS)[question_index]
        tasks=[dict(question=question,caption=row['brief_caption'],kind='brief')]
    else:
        facts=row['paired'][arm if arm in ('T1','T2') else 'T2']
        use_qa=arm=='T3' and (epoch%3==1 or (epoch%3==2 and number(identifier+'/qa')%2==0))
        tasks=question_tasks(identifier,facts,epoch) if use_qa else [dict(
            question=DETAILED_QUESTIONS[question_index],caption=caption(facts),kind='detailed',
            fact_ids=[f['source_fact_id'] for f in facts])]
        if use_qa and len(tasks)<2:
            tasks=[dict(question=DETAILED_QUESTIONS[question_index],caption=caption(facts),
                kind='detailed',fact_ids=[f['source_fact_id'] for f in facts])]
    return dict(id=identifier,image_path=row['image_path'],image_sha256=row['image_sha256'],
        tasks=tasks,weight=row.get('weight',1.),arm=arm,epoch=epoch)
