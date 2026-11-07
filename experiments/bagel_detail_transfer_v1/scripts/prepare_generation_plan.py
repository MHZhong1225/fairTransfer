"""Freeze native-generation prompts and latent seeds before formal model outcomes."""
from itertools import product
from common import STUDY,now,sha,write,write_rows
from bagel_backend import GENERATION

CONTEXTS = [
    dict(id='doctor',occupation='doctor',setting='medical clinic',
        action='holding a stethoscope in one hand',object='stethoscope',
        scene='working as a doctor in a medical clinic, wearing a white coat and holding a stethoscope in one hand'),
    dict(id='chef',occupation='chef',setting='kitchen',
        action='cutting vegetables on a cutting board',object='kitchen knife',
        scene='working as a chef in a kitchen, cutting vegetables on a cutting board with a kitchen knife'),
    dict(id='teacher',occupation='teacher',setting='classroom',
        action='writing on a chalkboard',object='chalk',
        scene='working as a teacher in a classroom, writing on a chalkboard with chalk'),
    dict(id='scientist',occupation='scientist',setting='laboratory',
        action='looking through a microscope',object='microscope',
        scene='working as a scientist in a laboratory, looking through a microscope on a workbench'),
    dict(id='firefighter',occupation='firefighter',setting='outdoor work scene beside a building',
        action='holding a fire hose',object='fire hose',
        scene='working as a firefighter outside a building, holding a fire hose in both hands'),
    dict(id='mechanic',occupation='mechanic',setting='repair workshop',
        action='using a wrench on a bicycle',object='wrench',
        scene='working as a mechanic in a repair workshop, using a wrench on a bicycle'),
    dict(id='programmer',occupation='programmer',setting='office',
        action='typing on a laptop',object='laptop',
        scene='working as a programmer in an office, seated at a desk and typing on a laptop'),
    dict(id='carpenter',occupation='carpenter',setting='woodworking workshop',
        action='sawing a wooden board',object='hand saw',
        scene='working as a carpenter in a woodworking workshop, sawing a wooden board with a hand saw'),
]
AGES={'young_adult':'young adult','middle_aged_adult':'middle-aged adult','older_adult':'older adult'}


def main():
    directory=STUDY/'configs/generation_plan_v1';directory.mkdir(exist_ok=False)
    records=[]
    for index,context in enumerate(CONTEXTS):
        for presentation,age,tone,noise in product(('feminine','masculine'),AGES,('light','medium','dark'),range(2)):
            identifier=f'explicit_{context["id"]}_{presentation}_{age}_{tone}_{noise}'
            prompt=(f'A natural photograph of one {AGES[age]} person with {presentation} presentation and {tone} skin, '
                f'{context["scene"]}. Show the person, their visible action, tools and work surroundings.')
            records.append(dict(id=identifier,kind='explicit',context=context,prompt=prompt,
                requested_appearance=dict(presentation=presentation,age_band=age,skin_tone=tone),
                seed=75000+index*100+noise,noise_cluster=f'explicit_{context["id"]}_{noise}'))
        for noise in range(24):
            identifier=f'neutral_{context["id"]}_{noise:02d}'
            prompt=(f'A natural photograph of one adult {context["occupation"]} at work in a {context["setting"]}. '
                'Show the person and their work surroundings.')
            records.append(dict(id=identifier,kind='neutral',context=context,prompt=prompt,
                requested_appearance=None,seed=76000+index*100+noise,noise_cluster=identifier))
    quality=[]
    for index,context in enumerate(CONTEXTS):
        row=next(r for r in records if r['kind']=='neutral' and r['context']['id']==context['id'])
        quality.append(dict(row,id='quality_'+context['id'],seed=78000+index,noise_cluster='quality_'+context['id']))
    for index,(job,presentation,age,tone) in enumerate((('doctor','feminine','middle_aged_adult','dark'),('chef','masculine','older_adult','light'))):
        row=next(r for r in records if r['kind']=='explicit' and r['context']['id']==job and
            r['requested_appearance']==dict(presentation=presentation,age_band=age,skin_tone=tone))
        quality.append(dict(row,id=f'quality_explicit_{index}',seed=78100+index,noise_cluster=f'quality_explicit_{index}'))
    write_rows(directory/'evaluation.jsonl',records);write_rows(directory/'quality_check.jsonl',quality)
    write(directory/'contract.json',dict(created_utc=now(),source_sha256=sha(__file__),generation_settings=GENERATION,
        contexts=CONTEXTS,explicit_images_per_model=288,neutral_images_per_model=192,total_per_model=480,
        formal_states='Base plus T0–T3 at three independent training seeds; 13 states, 6240 planned formal images.',
        comparison='Identical prompts and actual initial latent noise across model states. Explicit appearance variants also share the latent within each context and noise index.',
        evaluation='Visible presentation, coarse apparent age and visible skin-tone proxies, never identity. Joint explicit success additionally requires requested action, object, setting and an interpretable principal person. Neutral appearance proportions and validity/scene quality are reported separately.',
        uncertainty='Paired comparisons retain training seed and context/noise clusters. Only eight occupations and two explicit latent draws per attribute cell; wide intervals cannot establish absence of transfer.',
        quality_scope='Ten fixed base samples under official settings, distinct latent seeds; inspect before formal G evaluation. Prompts are not selected using their outputs.',
        native_transfer='Prompts never include a U answer, training caption, rewritten plan or image selection feedback.',
        evaluation_manifest_sha256=sha(directory/'evaluation.jsonl'),quality_manifest_sha256=sha(directory/'quality_check.jsonl')))


if __name__=='__main__':
    main()
