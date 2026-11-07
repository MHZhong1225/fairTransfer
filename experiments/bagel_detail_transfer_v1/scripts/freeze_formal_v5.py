"""Freeze actual eligible data only after real technical and visual checks finish."""
from collections import Counter,defaultdict
from pathlib import Path
import shutil
from common import STUDY,ROOT,BAGEL_REPO,now,read,rows,sha,write,write_rows
from prepare_u_evaluation import build
from supervision import DEMOGRAPHIC,make_pack
from build_fact_bank_v2 import token_counter


def describe_data(records,excluded):
    result={};count=token_counter()
    for split in ('train','dev','test'):
        selected=[r for r in records if r['split']==split]
        result[split]=dict(images=len(selected),sources={})
        for source in sorted({r['source'] for r in selected}):
            group=[r for r in selected if r['source']==source]
            result[split]['sources'][source]=dict(images=len(group),source_families=len({r['source_family'] for r in group}),
                framing=dict(Counter(r['framing'] for r in group)),
                visible_work_action=dict(Counter(r['work_action_visible'] for r in group)),
                adult_status=dict(Counter(r['adult_status'] for r in group)),
                appearance={field:{'unweighted':dict(Counter(r['appearance_labels'].get(field,'unknown') for r in group)),
                    'weighted':{category:sum(r.get('weight',1.) for r in group if r['appearance_labels'].get(field,'unknown')==category)
                        for category in sorted({r['appearance_labels'].get(field,'unknown') for r in group})}} for field in DEMOGRAPHIC},
                demographic_inserted=dict(Counter(r['paired']['demographic_inserted'] for r in group)),
                known_fact_counts=dict(Counter(f['field'] for r in group for f in r['facts'])),
                weight_min=min(r.get('weight',1.) for r in group),weight_max=max(r.get('weight',1.) for r in group))
        if split=='train':
            result[split]['actual_budgets']={}
            for arm in ('T0','T1','T2','T3'):
                lengths=[];question_tokens=0;weighted_tokens=0.;contexts=0;types=Counter()
                for row in selected:
                    for variant in range(3):
                        pack=make_pack(row,arm,variant)
                        for task in pack['tasks']:
                            tokens=count(task['caption']);lengths.append(tokens)
                            weighted_tokens+=tokens*row['weight'];question_tokens+=count(task['question'])-1
                            contexts+=1;types[task['kind']]+=1
                result[split]['actual_budgets'][arm]=dict(images=len(selected),image_exposures=len(selected)*3,
                    target_tokens=sum(lengths),weighted_target_tokens=weighted_tokens,
                    question_tokens=question_tokens,question_contexts=contexts,task_kinds=dict(types))
            budgets=result[split]['actual_budgets']
            result[split]['target_budget_relative_differences']={first+'-'+second:
                budgets[first]['target_tokens']/budgets[second]['target_tokens']-1 for first,second in (('T2','T1'),('T3','T2'))}
    result['exclusion_reasons']=dict(Counter(r['reason'] for r in excluded))
    result['excluded_by_source']=dict(Counter(r['source'] for r in excluded))
    result['limitations']=[
        'Appearance and facts are machine judgments. Shared mistakes and group-dependent missingness remain possible.',
        'BPO source occupation family is metadata, not a visible-occupation label or independent identity.',
        'No cross-split exact or prespecified near-duplicate candidates were found in the source audit; this is not proof of identity independence.',
        'The eleven prespecified near-duplicate candidate pairs are all within training and retained. They may increase repeated-content exposure equally across arms.',
        'Question-context counts and input tokens differ in T3. Same image exposures and similar target budgets do not imply equal FLOPs.',
        'Targets and group weights are identical in their image pool across arms; absent or uncertain demographic combinations are not invented.']
    return result


def main():
    remote_grading=(STUDY/'configs/remote_formal_grading_v4.json').exists()
    required={str(STUDY/path):status for path,status in (
        ('results/backend_qualification_v3/status.json','passed'),
        ('results/task_branch_qualification_v1/status.json','passed'),
        ('results/mini_scoring_queue_v5.json','completed'),
        ('results/data_quality_queue_v2_qwen.json','completed'),
        ('results/annotation_queue_v5.json','completed'),
        ('results/base_quality_v1/quality_review_v2.json','reviewed_for_formal_evaluation'))}
    for judge in ('qwen','ministral'):
        required[str(STUDY/f'results/scoring_qualification_v6/{judge}/status.json')]='passed'
        required[str(STUDY/f'results/depiction_check_v1/{judge}/status.json')]='completed'
    for path,status in required.items():
        if not Path(path).exists() or read(path).get('status')!=status:
            raise RuntimeError(f'Actual qualification is incomplete: {path}')
    if remote_grading:
        from remote_grading_v4 import qualify_remote_scoring
        qualify_remote_scoring()
        required[str(STUDY/'results/remote_grading_preflight_v4/status.json')]='passed'
    quality=read(STUDY/'results/base_quality_v1/quality_review_v2.json')
    planned_quality={r['id'] for r in rows(STUDY/'configs/generation_plan_v1/quality_check.jsonl')}
    if {r['id'] for r in quality['observations']}!=planned_quality:
        raise ValueError('Every fixed base quality image needs a recorded quality review')
    directory=STUDY/'configs/formal_v1'
    if directory.exists():
        raise RuntimeError('A formal freeze cannot be overwritten; inspect the existing version')
    records=rows(STUDY/'data/fact_bank_v2/records.jsonl');excluded=rows(STUDY/'data/fact_bank_v2/excluded.jsonl')
    technical=rows(STUDY/'data/technical_manifest.jsonl')+rows(STUDY/'data/scene_technical_v1/manifest.jsonl')
    if {r['image_sha256'] for r in records}&{r['image_sha256'] for r in technical}:
        raise ValueError('Technical and formal source images overlap')
    split_hashes={s:{r['image_sha256'] for r in records if r['split']==s} for s in ('train','dev','test')}
    if any(not hashes for hashes in split_hashes.values()) or any(split_hashes[a]&split_hashes[b]
            for a,b in (('train','dev'),('train','test'),('dev','test'))):
        raise ValueError('Empty or overlapping formal split')
    directory.mkdir()
    for split in ('train','dev','test'):
        write_rows(directory/(split+'.jsonl'),[r for r in records if r['split']==split])
    cases,pairs=build([r for r in records if r['split']=='test'])
    write_rows(directory/'u_cases.jsonl',cases);write_rows(directory/'u_pairs.jsonl',pairs)
    audit=describe_data(records,excluded);write(directory/'data_audit.json',audit)
    shutil.copy2(STUDY/'reports/experiment_protocol.md',directory/'protocol_at_freeze.md')
    analysis=read(STUDY/'configs/analysis_plan_v2.json')
    analysis.update(implementation_status='Aggregation, paired U/neutral/explicit intervals and the real five-state technical scoring pipeline are implemented and checked before this formal freeze.',
        original_plan_sha256=sha(STUDY/'configs/analysis_plan_v2.json'))
    write(directory/'analysis_at_freeze.json',analysis)
    root=STUDY/'results/formal_v1';states=[]
    combinations=[('base',None)]+[(arm,seed) for seed in (7101,7102,7103)
        for arm in (('T0','T1','T2','T3') if seed!=7102 else ('T3','T2','T1','T0'))]
    for index,(arm,seed) in enumerate(combinations):
        identifier='base' if arm=='base' else f'{arm}_seed{seed}'
        state_root=root/identifier
        states.append(dict(id=identifier,arm=arm,seed=seed,worker=0 if arm=='base' else (index-1)%2,
            training=str(state_root/'training') if arm!='base' else None,
            understanding=str(state_root/'understanding'),generation=str(state_root/'generation'),
            grades=str(state_root/'grades'),aggregate=str(state_root/'aggregate')))
    write_rows(directory/'states.jsonl',states)
    files=list((STUDY/'scripts').glob('*.py'))+list(directory.glob('*'))
    files += [STUDY/'configs/generation_plan_v1'/name for name in ('evaluation.jsonl','contract.json')]
    files += [STUDY/path for path in ('data/fact_bank_v2/contract.json','data/fact_bank_v2/balancing.json',
        'data/fact_bank_v2/records.jsonl','data/expanded_overlap_audit_v1.json',
        'configs/model_revision_verification_v1.json','configs/judge_environment_freeze.txt')]
    files += [STUDY/'configs/scoring_revision_v4.json',STUDY/'configs/base_quality_review_revision_v2.json',
        STUDY/'results/base_quality_v1/quality_review.json',
        STUDY/'results/base_quality_v1/codex_visual_review_v1.json']
    files += [ROOT/'scripts/understanding_generation_sft_v3/backend.py']
    files += list((ROOT/'scripts/shared_prefix_fairness').glob('*.py'))
    for subdirectory in ('modeling','data','inferencer'):
        files += list((BAGEL_REPO/subdirectory).rglob('*.py'))
    files += [BAGEL_REPO/'inferencer.py']
    files += [Path(path) for path in required]
    if remote_grading:
        files += [STUDY/'configs/remote_formal_grading_v4.json',STUDY/'results/remote_grading_preflight_v4/model_verification.json']
    freeze=dict(created_utc=now(),status='frozen_before_formal_training_and_test',
        grading_location='remote_gpu' if remote_grading else 'local_gpus',
        base_quality_review=str(STUDY/'results/base_quality_v1/quality_review_v2.json'),
        remote_grading_queue_status=str(STUDY/'results/remote_formal_grading_queue_v5.json'),
        scoring_revision=str(STUDY/'configs/scoring_revision_v4.json'),
        states_manifest=str(directory/'states.jsonl'),train=str(directory/'train.jsonl'),dev=str(directory/'dev.jsonl'),
        u_cases=str(directory/'u_cases.jsonl'),u_pairs=str(directory/'u_pairs.jsonl'),
        g_cases=str(STUDY/'configs/generation_plan_v1/evaluation.jsonl'),base_noise=str(root/'base_noise'),
        required_statuses=required,file_hashes={str(path):sha(path) for path in sorted(set(files))},
        exact_images={r['image_path']:r['image_sha256'] for r in records},
        selection='Fixed final epoch, all three seeds, all frozen held-out cases. No formal-result-based parser, sample, prompt or checkpoint changes.')
    write(directory/'freeze.json',freeze)
    print(directory/'freeze.json')


if __name__=='__main__':
    main()
