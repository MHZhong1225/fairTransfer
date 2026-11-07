"""Prespecified paired contrasts; no selection of a best seed or checkpoint."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from common import now,rows,sha,write
from metrics_v2 import CATEGORIES,paired_explicit_interval
from paired_statistics import paired_u_interval,paired_neutral_interval

SEEDS=(7101,7102,7103)
COMPARISONS=(('T1','T0'),('T2','T1'),('T3','T2'),('T0','base'),('T1','base'),('T2','base'),('T3','base'))


def lookup(states):
    indexed={}
    for state in states:
        key=(state['arm'],state['seed'])
        if key in indexed:
            raise ValueError('Repeated model state')
        root=Path(state['aggregate'])
        indexed[key]=dict(u={r['id']:r for r in rows(root/'understanding_cases.jsonl')},
            g={r['id']:r for r in rows(root/'generation_cases.jsonl')})
    required={('base',None)}|set(product(('T0','T1','T2','T3'),SEEDS))
    if set(indexed)!=required:
        raise ValueError('Require unchanged base and every prespecified arm/seed')
    for task in ('u','g'):
        reference=set(indexed[('base',None)][task])
        if any(set(state[task])!=reference for state in indexed.values()):
            raise ValueError('Unmatched planned cases across model states')
    return indexed


def state_for(indexed,arm,seed):
    return indexed[(arm,None if arm=='base' else seed)]


def g_value(row,judge,metric):
    value=row['common'] if judge=='common' else row['judges'][judge]
    if metric.startswith('attribute:'):
        field=metric.split(':')[1]
        return value['appearance'][field]==row['requested_appearance'][field]
    if metric.startswith('content:'):
        return value['content'].get(metric.split(':')[1],False)
    return value[metric]


def explicit_arrays(indexed,arm,judge,metric):
    reference=indexed[('base',None)]['g']
    groups=defaultdict(list)
    for row in reference.values():
        if row['kind']=='explicit':
            groups[(row['context'],row['noise_cluster'])].append(row)
    contexts=sorted({c for c,n in groups});ordered=[]
    expected_cells=set(product(*CATEGORIES.values()))
    for context in contexts:
        noises=sorted(n for c,n in groups if c==context);current=[]
        for noise in noises:
            items=groups[(context,noise)]
            cells={tuple(r['requested_appearance'][f] for f in CATEGORIES):r['id'] for r in items}
            if set(cells)!=expected_cells or len(cells)!=len(items):
                raise ValueError('Explicit attribute cells are incomplete or duplicated')
            current.append([cells[cell] for cell in sorted(expected_cells)])
        ordered.append(current)
    if len({len(group) for group in ordered})!=1:
        raise ValueError('Explicit noise counts differ across contexts')
    return np.array([[[[g_value(state_for(indexed,arm,s)['g'][i],judge,metric)
        for i in noise] for noise in context] for context in ordered] for s in SEEDS],dtype=float)


def neutral_arrays(indexed,arm,judge,field):
    reference=indexed[('base',None)]['g'];groups=defaultdict(list)
    for row in reference.values():
        if row['kind']=='neutral':
            groups[row['context']].append(row['id'])
    ordered=[sorted(groups[context]) for context in sorted(groups)]
    if len({len(group) for group in ordered})!=1:
        raise ValueError('Neutral noise counts differ across contexts')
    codes={value:i for i,value in enumerate(CATEGORIES[field])}
    return np.array([[[codes.get(g_value(state_for(indexed,arm,s)['g'][i],judge,'appearance')[field],-1)
        for i in context] for context in ordered] for s in SEEDS],dtype=int)


def u_value(row,metric):
    if metric=='coverage:all':
        return row['common_coverage']['value']
    if metric.startswith('coverage:'):
        return row['common_groups'][metric.split(':')[1]]['value']
    if metric.startswith('precision:'):
        return row['judges'][metric.split(':')[1]]['claim_precision']
    return row['common_coverage']['value']


def u_arrays(indexed,arm,ids,metric):
    return np.array([[u_value(state_for(indexed,arm,s)['u'][i],metric) for i in ids] for s in SEEDS],dtype=float)


def main(args):
    states=rows(args.states);indexed=lookup(states)
    args.output.mkdir(parents=True,exist_ok=False)
    results={};reference=indexed[('base',None)]['u']
    for first,second in COMPARISONS:
        contrast=dict(first=first,second=second,g_explicit={},g_neutral={},u={})
        for judge in ('qwen','ministral','common'):
            contrast['g_explicit'][judge]={metric:paired_explicit_interval(
                explicit_arrays(indexed,first,judge,metric),explicit_arrays(indexed,second,judge,metric),draws=args.draws)
                for metric in ('joint_success','valid_person',*(f'attribute:{f}' for f in CATEGORIES),
                    'content:setting','content:action','content:object')}
            contrast['g_neutral'][judge]={field:paired_neutral_interval(
                neutral_arrays(indexed,first,judge,field),neutral_arrays(indexed,second,judge,field),field,draws=args.draws)
                for field in CATEGORIES}
        for source in sorted({r['source'] for r in reference.values() if r['image_id']}):
            contrast['u'][source]={}
            choices=[('detailed',None,m) for m in ('coverage:ordinary','coverage:demographic','precision:qwen','precision:ministral')]
            choices.extend(('qa',field,'coverage:all') for field in CATEGORIES)
            for task,field,metric in choices:
                ids=sorted(i for i,r in reference.items() if r['source']==source and r['task']==task and r['field']==field)
                if not ids:
                    continue
                clusters=[reference[i]['image_id'] if reference[i]['image_id'].startswith('coco_') else reference[i]['source_family'] for i in ids]
                key=task+':'+(field or metric)
                contrast['u'][source][key]=paired_u_interval(u_arrays(indexed,first,ids,metric),u_arrays(indexed,second,ids,metric),clusters,draws=args.draws)
        results[first+'-'+second]=contrast
        write(args.output/(first+'-'+second+'.json'),contrast)
    write(args.output/'contract.json',dict(completed_utc=now(),states_sha256=sha(args.states),draws=args.draws,
        seeds=list(SEEDS),primary='T2-T1 / G explicit / common / joint_success',
        secondary='All other endpoints and subgroup descriptions are exploratory; no best-seed or best-checkpoint selection.',
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('compare_states.py','paired_statistics.py','metrics.py','metrics_v2.py')},
        input_hashes={str(Path(s['aggregate'])/name):sha(Path(s['aggregate'])/name)
            for s in states for name in ('understanding_cases.jsonl','generation_cases.jsonl')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=10000)
    main(parser.parse_args())
