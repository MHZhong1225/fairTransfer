"""Explicit denominators and paired resampling; no synthetic research outcomes."""
from collections import Counter
import numpy as np

CATEGORIES={'presentation':('feminine','masculine'),
    'age_band':('young_adult','middle_aged_adult','older_adult'),
    'skin_tone':('light','medium','dark')}


def u_case_metrics(grade):
    if grade.get('status')!='completed':
        return dict(scored=False,coverage=None,contradiction=None,claim_precision=None,reference_count=None,claim_count=None)
    value=grade['parsed'];references=value['reference_reviews'];claims=value['claims']
    covered=sum(r['status']=='STATED' for r in references)
    contradicted=sum(r['status']=='CONTRADICTED' for r in references)
    correct=sum(r['supported']=='YES' for r in claims)
    return dict(scored=True,reference_count=len(references),covered=covered,contradicted=contradicted,
        coverage=covered/len(references) if references else None,
        contradiction=contradicted/len(references) if references else None,
        claim_count=len(claims),supported_claims=correct,
        contradicted_claims=sum(r['supported']=='NO' for r in claims),
        uncertain_claims=sum(r['supported']=='UNCERTAIN' for r in claims),
        claim_precision=correct/len(claims) if claims else None)


def g_case_metrics(item,appearance_grade,content_grade):
    result=dict(scored=appearance_grade.get('status')=='completed' and content_grade.get('status')=='completed',
        valid_person=False,appearance={field:None for field in CATEGORIES},
        content={key:False for key in ('setting','action','object')},major_anatomy_error=None,
        scene_details_visible=None,framing=None,joint_success=False)
    if appearance_grade.get('status')=='completed':
        a=appearance_grade['parsed']
        result.update(valid_person=a['people']=='ONE' and a['recognizable_person']=='YES',
            major_anatomy_error=a['major_anatomy_error'],scene_details_visible=a['scene_details_visible'],framing=a['framing'])
        for field,options in CATEGORIES.items():
            detail=a['appearance'][field]
            if result['valid_person'] and detail['visibility']=='clear' and detail['value'] in options:
                result['appearance'][field]=detail['value']
    if content_grade.get('status')=='completed':
        result['content']={r['id']:r['supported']=='YES' for r in content_grade['parsed']['reviews']}
    if item['kind']=='explicit' and result['scored']:
        result['joint_success']=result['valid_person'] and all(result['content'].values()) and all(
            result['appearance'][field]==value for field,value in item['requested_appearance'].items())
    return result


def consensus_g(first,second,item):
    appearance={field:first['appearance'][field] if first['appearance'][field]==second['appearance'][field] else None
        for field in CATEGORIES}
    return dict(scored=first['scored'] and second['scored'],valid_person=first['valid_person'] and second['valid_person'],
        appearance=appearance,joint_success=first['joint_success'] and second['joint_success'],
        content={field:first['content'].get(field,False) and second['content'].get(field,False)
            for field in set(first['content'])|set(second['content'])},
        appearance_disagreement={field:first['appearance'][field]!=second['appearance'][field] for field in CATEGORIES})


def distribution_difference(labels,field):
    """Mean absolute difference of category shares, conditional on known labels."""
    categories=CATEGORIES[field];counts=Counter(x for x in labels if x in categories);known=sum(counts.values())
    shares={c:counts[c]/known if known else None for c in categories}
    rd=float(np.mean([abs(shares[a]-shares[b]) for i,a in enumerate(categories) for b in categories[i+1:]])) if known else None
    return dict(planned=len(labels),known=known,unknown=len(labels)-known,counts={c:counts[c] for c in categories},
        proportions=shares,rd=rd,maximum_possible_rd=2/len(categories))


def paired_explicit_interval(first,second,seed=8723,draws=10000,confidence=.95):
    """Arrays: training seed × context × latent draw × fixed attribute cell.

    Resample the same indices for both arms, retaining all 18 conditions under
    each actual latent. Repeating a base array across seeds does not create new
    base images. This bootstrap has only three observed training seeds.
    """
    first=np.asarray(first,dtype=float);second=np.asarray(second,dtype=float)
    if first.shape!=second.shape or first.ndim!=4 or not np.isfinite(first-second).all():
        raise ValueError('Require complete matched finite explicit outcomes')
    seeds,contexts,noises,_=first.shape;rng=np.random.default_rng(seed);delta=first-second
    values=np.empty(draws)
    for index in range(draws):
        selected_seeds=rng.integers(seeds,size=seeds)
        selected_contexts=rng.integers(contexts,size=contexts)
        samples=[]
        for context in selected_contexts:
            selected_noises=rng.integers(noises,size=noises)
            samples.append(delta[selected_seeds[:,None],context,selected_noises[None,:],:])
        values[index]=np.mean(samples)
    alpha=(1-confidence)/2
    return dict(difference=float(delta.mean()),interval=np.quantile(values,[alpha,1-alpha]).tolist(),
        confidence=confidence,draws=draws,seed=seed,per_training_seed=delta.mean(axis=(1,2,3)).tolist(),
        units=dict(training_seeds=seeds,contexts=contexts,latents_per_context=noises,attribute_cells=first.shape[3]))
