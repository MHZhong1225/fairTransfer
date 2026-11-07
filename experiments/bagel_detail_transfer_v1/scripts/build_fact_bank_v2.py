"""Create auditable paired targets and common weights after blind verification."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import statistics
import sys
from common import STUDY,BAGEL_REPO,BAGEL_WEIGHTS,now,read,rows,sha,write,write_rows
from supervision import DEMOGRAPHIC,verified_facts,select_paired_facts,brief_caption,make_pack


def token_counter():
    from transformers import Qwen2Tokenizer
    tokenizer=Qwen2Tokenizer.from_pretrained(str(BAGEL_WEIGHTS),local_files_only=True)
    @lru_cache(maxsize=200000)
    def count(text):
        return len(tokenizer.encode(text,add_special_tokens=False))+1
    return count


def common_weights(records):
    """Same bounded marginal raking for all arms; never fill absent joint cells."""
    cells=defaultdict(list)
    for row in records:
        if row['split']=='train':
            cells[(row['content_stratum'],row['framing'],row['work_action_visible'])].append(row)
        row['weight']=1.
    audit=[]
    for cell,group in sorted(cells.items()):
        weights={r['id']:1. for r in group}
        margins={}
        for field in DEMOGRAPHIC:
            categories=defaultdict(list)
            for row in group:
                if field in row['appearance_labels']:
                    categories[row['appearance_labels'][field]].append(row['id'])
            if len(categories)>=2 and min(map(len,categories.values()))>=3:
                margins[field]=categories
        for _ in range(20):
            for categories in margins.values():
                total=sum(weights[i] for ids in categories.values() for i in ids)
                target=total/len(categories)
                for ids in categories.values():
                    factor=target/sum(weights[i] for i in ids)
                    for identifier in ids:
                        weights[identifier]=min(4.,max(.25,weights[identifier]*factor))
        normalization=len(group)/sum(weights.values())
        for row in group:
            row['weight']=weights[row['id']]*normalization
        audit.append(dict(content_stratum=cell[0],framing=cell[1],work_action_visible=cell[2],images=len(group),
            attempted_margins={field:{k:len(v) for k,v in cats.items()} for field,cats in margins.items()},
            weighted_margins={field:{k:sum(weights[i]*normalization for i in ids) for k,ids in cats.items()} for field,cats in margins.items()},
            clipped_raw=sum(w<=.25 or w>=4. for w in weights.values()),normalization=normalization))
    return audit


def inputs(technical):
    if technical:
        return [(STUDY/'data/technical_manifest.jsonl',STUDY/'results/fact_annotation_technical_v4'),
            (STUDY/'data/scene_technical_v1/manifest.jsonl',STUDY/'results/fact_annotation_scene_technical_v5')]
    if read(STUDY/'results/annotation_queue_v5.json')['status']!='completed':
        raise RuntimeError('The complete frozen annotation pool is required before formal fact-bank construction')
    return [(STUDY/'data/combined_sources_v1/manifest.jsonl',STUDY/'results/fact_annotation_full_v5')]


def run(technical):
    sources=inputs(technical);count=token_counter();accepted=[];excluded=[];all_facts=[];hashes={}
    depictions={}
    for judge in ('qwen','ministral'):
        directory=STUDY/'results/depiction_check_v1'/judge
        if read(directory/'status.json')['status']!='completed':
            raise RuntimeError('Both independent depiction screens must finish before fact-bank construction')
        depictions[judge]={p.stem:read(p) for p in (directory/'observations').glob('*.json')}
        hashes[str(directory/'contract.json')]=sha(directory/'contract.json')
        hashes[str(directory/'status.json')]=sha(directory/'status.json')
        hashes.update({str(p):sha(p) for p in sorted((directory/'observations').glob('*.json'))})
    for manifest,directory in sources:
        for task in ('propose','verify'):
            if read(directory/task/'status.json')['status']!='completed':
                raise RuntimeError('Annotation worker has not completed')
        proposals={r['id']:r for r in rows(directory/'propose/observations.jsonl')}
        verifications={r['id']:r for r in rows(directory/'verify/observations.jsonl')}
        hashes.update({str(p):sha(p) for p in (manifest,directory/'propose/observations.jsonl',directory/'verify/observations.jsonl')})
        for item in rows(manifest):
            proposal=proposals[item['id']];verification=verifications[item['id']]
            reason=None
            depiction={judge:depictions[judge][item['id']] for judge in depictions}
            if any(r['status']!='completed' for r in depiction.values()):
                reason='depiction_judgment_unavailable'
            elif any(r['parsed']['category']!='PHOTO_LIKE_PERSON' for r in depiction.values()):
                reason='photo_like_person_not_independently_confirmed'
            elif proposal['status']!='completed' or verification['status']!='completed':
                reason='annotation_format_or_runtime_failure'
            else:
                p=proposal['parsed'];v=verification['parsed']
                facts,rejections=verified_facts(p,v)
                all_facts.append(dict(id=item['id'],facts=facts,rejected=rejections,
                    proposer_observations={k:p[k] for k in ('people','adult_subject','work_action_visible','tool_visible')},
                    verifier_observations={k:v[k] for k in ('people','adult_subject','work_action_visible','tool_visible')}))
                if p['people']!='ONE' or v['people']!='ONE':
                    reason='single_principal_person_not_agreed'
                elif 'NO' in (p['adult_subject'],v['adult_subject']):
                    reason='an_annotator_identified_a_nonadult_subject'
                elif sum(f['field'] not in DEMOGRAPHIC for f in facts)<2:
                    reason='fewer_than_two_confirmed_ordinary_facts'
                else:
                    paired=select_paired_facts(item['id'],facts,count)
                    labels={f['field']:f['value'] for f in facts if f['field'] in DEMOGRAPHIC}
                    framing=next((f['value'] for f in facts if f['field']=='framing'),'unknown')
                    action=p['work_action_visible'] if p['work_action_visible']==v['work_action_visible'] else 'DISAGREE'
                    accepted.append(dict(**item,depiction={j:r['parsed'] for j,r in depiction.items()},facts=facts,paired=paired,appearance_labels=labels,
                        framing=framing,work_action_visible=action,brief_caption=brief_caption(facts),
                        content_stratum=item.get('source_object_stratum') or item['source_family'],
                        adult_status=p['adult_subject'] if p['adult_subject']==v['adult_subject'] else 'DISAGREE'))
            if reason:
                excluded.append(dict(id=item['id'],source=item.get('source','BPO technical'),split=item['split'],reason=reason))
    # Technical images never acquire formal training status or balancing weights.
    balancing=[] if technical else common_weights(accepted)
    budget={}
    for split in sorted({r['split'] for r in accepted}):
        group=[r for r in accepted if r['split']==split]
        budget[split]={}
        for arm in ('T0','T1','T2','T3'):
            targets=[];task_counts=Counter();image_exposures=0
            for row in group:
                for epoch in range(3):
                    pack=make_pack(row,arm,epoch);image_exposures+=1
                    targets.extend(count(t['caption']) for t in pack['tasks'])
                    task_counts.update(t['kind'] for t in pack['tasks'])
            budget[split][arm]=dict(images=len(group),image_exposures=image_exposures,
                supervised_target_tokens=sum(targets),question_contexts=len(targets),task_counts=dict(task_counts),
                max_target_tokens=max(targets,default=0))
    directory=STUDY/'data'/('technical_fact_bank_v2' if technical else 'fact_bank_v2')
    directory.mkdir(exist_ok=False)
    write_rows(directory/'records.jsonl',accepted);write_rows(directory/'excluded.jsonl',excluded)
    write_rows(directory/'all_verified_facts.jsonl',all_facts)
    write(directory/'balancing.json',balancing)
    write(directory/'contract.json',dict(created_utc=now(),technical=technical,input_hashes=hashes,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in ('build_fact_bank_v2.py','supervision.py','depiction.py')},
        annotation_policy='Keep only independently confirmed facts; demographics additionally require clear visibility. Unknown/negative facts remain in the audit, never filled from source requests.',
        revision_reason='A jointly mislabeled sculpted bust in excluded technical data required an independent depiction screen before any formal training. Original labels and v1 technical fact bank are retained.',
        eligibility='Both independent depiction screens must confirm PHOTO_LIKE_PERSON. Both fact judges report ONE person; neither says nonadult; at least two ordinary verified facts. Uncertain age/adulthood is retained for ordinary visual learning, not imputed.',
        balancing='Common loss weights only: within content×framing×visible-work-action cells, rake supported appearance marginals with at least three observations per category for 20 fixed cycles, raw weights clipped [.25,4], then normalize each cell. No absent joint category is filled. All arms have identical weights and image exposures.',
        T3='The T2 facts are divided into 2–4 independent question contexts after one shared image encoding. About 40% of all exposures replace captions; other branches never read a previous gold answer. No additional image pass, epoch or G loss.',
        budget=budget,counts=dict(Counter(r['split'] for r in accepted)),exclusion_counts=dict(Counter(r['reason'] for r in excluded)),
        appearance_coverage={field:dict(Counter(r['appearance_labels'].get(field,'unknown') for r in accepted)) for field in DEMOGRAPHIC},
        source_counts=dict(Counter(r.get('source','BPO technical') for r in accepted)),
        selected_demographic_fact_counts=dict(Counter(r['paired']['demographic_inserted'] for r in accepted)),
        missingness='Known-appearance denominators differ by field; marginal balance is not intersectional balance, and generated source conditions are not labels.'))
    print(directory)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--technical',action='store_true')
    run(parser.parse_args().technical)
