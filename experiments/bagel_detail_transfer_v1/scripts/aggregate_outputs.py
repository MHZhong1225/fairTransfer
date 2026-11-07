"""Aggregate immutable native outputs and independent grades with explicit denominators."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from common import now,read,rows,sha,write,write_rows
from metrics_v2 import CATEGORIES,u_case_metrics,g_case_metrics,consensus_g,distribution_difference

JUDGES=('qwen','ministral')


def indexed(directory):
    result={}
    for path in sorted(Path(directory).rglob('observations/*.json')):
        value=read(path)
        if value['id'] in result:
            raise ValueError(f'Duplicate case {value["id"]} in {directory}')
        result[value['id']]=value
    return result


def mean(values):
    values=list(values);known=[x for x in values if x is not None]
    return dict(value=float(np.mean(known)) if known else None,defined=len(known),planned=len(values))


def fraction(values):
    values=list(values)
    return dict(value=sum(values)/len(values) if values else None,numerator=int(sum(values)),denominator=len(values))


def u_row(item,measured,grades):
    facts=item['reference_facts'];references=dict(enumerate(facts))
    row={key:item[key] for key in ('id','image_id','source','source_family','task','field','appearance_labels')}
    row.update(measured=bool(measured),response=measured.get('response'),decoding=measured.get('decoding'),judges={})
    stated={};common=[]
    for judge,grade in grades.items():
        metrics=u_case_metrics(grade)
        reviews=grade.get('parsed',{}).get('reference_reviews',[]) if metrics['scored'] else []
        stated[judge]={r['id'] for r in reviews if r['status']=='STATED'}
        groups={}
        for name,ids in (('ordinary',[i for i,f in references.items() if f['field'] not in CATEGORIES]),
                         ('demographic',[i for i,f in references.items() if f['field'] in CATEGORIES])):
            groups[name]=dict(references=len(ids),covered=sum(i in stated[judge] for i in ids),
                coverage=sum(i in stated[judge] for i in ids)/len(ids) if ids and metrics['scored'] else None)
        row['judges'][judge]=dict(metrics,status=grade.get('status','missing_grade'),groups=groups)
    for identifier,fact in enumerate(facts):
        common.append(dict(id=identifier,field=fact['field'],value=fact['value'],
            stated=all(identifier in stated[j] for j in JUDGES)))
    row['common_reference_facts']=common
    row['common_coverage']=mean(float(f['stated']) for f in common)
    row['common_groups']={name:mean(float(f['stated']) for f in common if (f['field'] in CATEGORIES)==demographic)
        for name,demographic in (('ordinary',False),('demographic',True))}
    row['all_references_confirmed']=bool(common) and all(f['stated'] for f in common)
    return row


def summarize_u(items,pairs):
    groups=defaultdict(list)
    for row in items:
        groups[(row['source'],row['task'],row['field'])].append(row)
    result=[]
    for (source,task,field),group in sorted(groups.items(),key=lambda kv:str(kv[0])):
        entry=dict(source=source,task=task,field=field,planned_cases=len(group),
            measured_cases=sum(r['measured'] for r in group),
            common_image_mean_coverage=mean(r['common_coverage']['value'] for r in group),
            common_fact_coverage=fraction(f['stated'] for r in group for f in r['common_reference_facts']),
            common_groups={name:mean(r['common_groups'][name]['value'] for r in group) for name in ('ordinary','demographic')},
            all_references_confirmed=fraction(r['all_references_confirmed'] for r in group),judges={})
        for judge in JUDGES:
            entry['judges'][judge]=dict(
                grading_status=dict(Counter(r['judges'][judge]['status'] for r in group)),
                image_mean_coverage=mean(r['judges'][judge]['coverage'] for r in group),
                image_mean_claim_precision=mean(r['judges'][judge]['claim_precision'] for r in group),
                claim_totals={key:sum(r['judges'][judge].get(key,0) or 0 for r in group)
                    for key in ('claim_count','supported_claims','contradicted_claims','uncertain_claims')},
                zero_claim_responses=sum(r['judges'][judge].get('claim_count')==0 for r in group))
        entry['generation']=dict(
            tokens=mean((r['decoding'] or {}).get('native_sequence_length') for r in group),
            empty_answers=sum(r['response']=='' for r in group),
            truncated=sum(bool((r['decoding'] or {}).get('hit_length_limit')) for r in group))
        if task in ('detailed','qa'):
            entry['appearance_subgroups']={}
            for attribute,categories in CATEGORIES.items():
                rows_by_class={category:[r for r in group if r['appearance_labels'].get(attribute)==category] for category in categories}
                scores={category:dict(cases=len(selected),
                    coverage=mean(r['common_coverage']['value'] for r in selected),
                    ordinary_coverage=mean(r['common_groups']['ordinary']['value'] for r in selected))
                    for category,selected in rows_by_class.items()}
                values=[r['coverage']['value'] for r in scores.values() if r['coverage']['value'] is not None]
                entry['appearance_subgroups'][attribute]=dict(classes=scores,
                    unknown_cases=sum(attribute not in r['appearance_labels'] for r in group),
                    descriptive_max_min_gap=max(values)-min(values) if len(values)>1 else None,
                    limitation='Content and visibility may differ between appearance classes; this descriptive gap is not an isolated causal demographic effect.')
        result.append(entry)
    lookup={r['id']:r for r in items};dependence=[]
    for pair in pairs:
        left=lookup[pair['left']];right=lookup[pair['right']]
        dependence.append(dict(pair,both_correct=left['all_references_confirmed'] and right['all_references_confirmed'],
            same_generated_text=left['response']==right['response'] if left['measured'] and right['measured'] else None))
    controls=[r for r in items if r['task']=='no_image']
    return dict(groups=result,image_dependence=dependence,
        image_dependence_summary={field:{scope:fraction(p['both_correct'] for p in dependence
            if p['field']==field and (scope=='all' or p['same_content'])) for scope in ('all','same_content')}
            for field in CATEGORIES},no_image_controls=controls)


def g_row(item,measured,grades):
    row=dict(id=item['id'],kind=item['kind'],context=item['context']['id'],noise_cluster=item['noise_cluster'],
        requested_appearance=item['requested_appearance'],measured=bool(measured),judges={})
    for judge in JUDGES:
        values=grades[judge]
        row['judges'][judge]=g_case_metrics(item,values['depiction'],values['appearance'],values['content'])
        row['judges'][judge]['grading_status']={task:g.get('status','missing_grade') for task,g in values.items()}
    row['common']=consensus_g(*(row['judges'][j] for j in JUDGES),item)
    return row


def summarize_g(items):
    result={}
    for judge in (*JUDGES,'common'):
        result[judge]={}
        for kind in ('explicit','neutral'):
            group=[r for r in items if r['kind']==kind]
            outcomes=[r['common'] if judge=='common' else r['judges'][judge] for r in group]
            summary=dict(planned=len(group),measured=sum(r['measured'] for r in group),
                scored=fraction(r['scored'] for r in outcomes),
                depiction_accepted=fraction(r['depiction_accepted'] for r in outcomes),
                valid_person=fraction(r['valid_person'] for r in outcomes),
                content={field:fraction(r['content'].get(field,False) for r in outcomes)
                    for field in (('setting','action','object') if kind=='explicit' else ('setting',))})
            if judge!='common':
                summary.update(depiction_counts=dict(Counter(r['depiction'] or 'unscored' for r in outcomes)),
                    framing=dict(Counter(r['framing'] or 'unscored' for r in outcomes)),
                    major_anatomy_error=dict(Counter(r['major_anatomy_error'] or 'unscored' for r in outcomes)),
                    scene_details_visible=dict(Counter(r['scene_details_visible'] or 'unscored' for r in outcomes)))
            else:
                summary['appearance_disagreement']={field:fraction(r['appearance_disagreement'][field] for r in outcomes) for field in CATEGORIES}
                summary['depiction_disagreement']=fraction(r['depiction_disagreement'] for r in outcomes)
            if kind=='explicit':
                summary['joint_success']=fraction(r['joint_success'] for r in outcomes)
                summary['attribute_success']={field:fraction(outcome['appearance'][field]==item['requested_appearance'][field]
                    for item,outcome in zip(group,outcomes,strict=True)) for field in CATEGORIES}
                summary['by_requested_class']={field:{category:fraction(outcome['joint_success']
                    for item,outcome in zip(group,outcomes,strict=True) if item['requested_appearance'][field]==category)
                    for category in categories} for field,categories in CATEGORIES.items()}
            else:
                cells={}
                for context in sorted({r['context'] for r in group}):
                    cells[context]={field:distribution_difference([o['appearance'][field] for r,o in zip(group,outcomes,strict=True)
                        if r['context']==context],field) for field in CATEGORIES}
                summary['by_occupation']=cells
                summary['mean_rd']={}
                for field in CATEGORIES:
                    values=[c[field]['rd'] for c in cells.values()];partial=mean(values)
                    summary['mean_rd'][field]=dict(full_occupation_mean=partial['value'] if partial['defined']==partial['planned'] else None,
                        partial_mean=partial['value'],defined_occupations=partial['defined'],planned_occupations=partial['planned'])
            result[judge][kind]=summary
    return result


def aggregate_state(state,u_plan,g_plan,pairs):
    output=Path(state['aggregate']);output.mkdir(parents=True,exist_ok=False)
    measured_u=indexed(state['understanding']);measured_g=indexed(state['generation'])
    grades={judge:{task:indexed(Path(state['grades'])/judge/task)
        for task in ('understanding','depiction','appearance','content')} for judge in JUDGES}
    # Every planned result must have a recorded grading outcome, including failure.
    for judge in JUDGES:
        for task in grades[judge]:
            planned={r['id'] for r in (u_plan if task=='understanding' else g_plan)}
            if set(grades[judge][task])!=planned:
                raise ValueError(f'Incomplete or unexpected grading cases: {state["id"]}/{judge}/{task}')
    u=[u_row(item,measured_u.get(item['id'],{}),{j:grades[j]['understanding'][item['id']] for j in JUDGES}) for item in u_plan]
    g=[g_row(item,measured_g.get(item['id'],{}),{j:{t:grades[j][t][item['id']] for t in ('depiction','appearance','content')} for j in JUDGES}) for item in g_plan]
    write_rows(output/'understanding_cases.jsonl',u);write_rows(output/'generation_cases.jsonl',g)
    summary=dict(created_utc=now(),state=state,u=summarize_u(u,pairs),g=summarize_g(g))
    write(output/'summary.json',summary)
    return summary


def main(args):
    states=rows(args.states);u_plan=rows(args.u_cases);g_plan=rows(args.g_cases);pairs=rows(args.pairs)
    for state in states:
        aggregate_state(state,u_plan,g_plan,pairs)
    write(args.states.with_suffix('.aggregation.json'),dict(completed_utc=now(),states_sha256=sha(args.states),
        manifests={str(p):sha(p) for p in (args.u_cases,args.g_cases,args.pairs)},
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in ('aggregate_outputs.py','metrics.py','metrics_v2.py')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('states','u-cases','g-cases','pairs'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
