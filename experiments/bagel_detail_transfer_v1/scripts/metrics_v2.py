"""Same denominators, additionally excluding nonhuman depictions from valid people."""
from metrics import (CATEGORIES,u_case_metrics,distribution_difference,paired_explicit_interval,
    g_case_metrics as original_g_case_metrics,consensus_g as original_consensus_g)


def g_case_metrics(item,depiction_grade,appearance_grade,content_grade):
    result=original_g_case_metrics(item,appearance_grade,content_grade)
    depicted=depiction_grade.get('status')=='completed'
    category=depiction_grade['parsed']['category'] if depicted else None
    accepted=category=='PHOTO_LIKE_PERSON'
    result.update(depiction=category,depiction_accepted=accepted,scored=result['scored'] and depicted,
        valid_person=result['valid_person'] and accepted,joint_success=result['joint_success'] and accepted)
    if not result['valid_person']:
        result['appearance']={field:None for field in CATEGORIES}
    return result


def consensus_g(first,second,item):
    result=original_consensus_g(first,second,item)
    result.update(depiction_accepted=first['depiction_accepted'] and second['depiction_accepted'],
        depiction_disagreement=first['depiction']!=second['depiction'])
    return result
