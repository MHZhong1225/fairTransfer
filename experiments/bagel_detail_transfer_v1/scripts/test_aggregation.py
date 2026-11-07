"""Denominator fixtures deliberately include failures, missing labels and empty responses."""
import unittest
from itertools import product
import numpy as np
from aggregate_outputs import u_row,summarize_u,g_row,summarize_g
from compare_states import explicit_arrays,neutral_arrays,u_arrays,SEEDS
from metrics import CATEGORIES


class AggregationTests(unittest.TestCase):
    def test_u_failed_grade_remains_in_common_denominator(self):
        item=dict(id='a',image_id='image',source='fixture',source_family='family',task='detailed',field=None,
            appearance_labels={'presentation':'feminine'},reference_facts=[
                dict(field='presentation',value='feminine',source_fact_id=4),
                dict(field='held_object',value='book',source_fact_id=7)])
        correct=dict(status='completed',parsed=dict(reference_reviews=[dict(id=0,status='STATED'),dict(id=1,status='OMITTED')],claims=[]))
        result=u_row(item,dict(response='',decoding={'native_sequence_length':0}),dict(qwen=correct,ministral={'status':'parse_failed'}))
        self.assertEqual(result['common_coverage']['value'],0.)
        self.assertEqual(result['common_groups']['demographic']['planned'],1)
        self.assertEqual(result['judges']['qwen']['coverage'],.5)
        self.assertIsNone(result['judges']['qwen']['claim_precision'])
        report=summarize_u([result],[])
        self.assertEqual(report['groups'][0]['common_fact_coverage']['denominator'],2)
        self.assertEqual(report['groups'][0]['judges']['qwen']['zero_claim_responses'],1)

    def test_neutral_missing_occupation_never_looks_balanced(self):
        item=dict(id='a',kind='neutral',context={'id':'doctor'},noise_cluster='doctor_0',requested_appearance=None)
        grades={j:{t:{} for t in ('depiction','appearance','content')} for j in ('qwen','ministral')}
        row=g_row(item,{},grades);report=summarize_g([row])
        neutral=report['common']['neutral']
        self.assertEqual(neutral['valid_person']['denominator'],1)
        self.assertIsNone(neutral['mean_rd']['presentation']['full_occupation_mean'])
        self.assertEqual(neutral['by_occupation']['doctor']['presentation']['unknown'],1)
        self.assertEqual(neutral['mean_rd']['presentation']['defined_occupations'],0)

    def test_comparison_axes_preserve_every_cell_and_base_pairing(self):
        generated={}
        for context,noise,values in product(('a','b'),range(2),product(*CATEGORIES.values())):
            identifier=context+str(noise)+'_'.join(values)
            appearance=dict(zip(CATEGORIES,values))
            generated[identifier]=dict(id=identifier,kind='explicit',context=context,noise_cluster=f'{context}_{noise}',
                requested_appearance=appearance,common=dict(appearance=appearance,joint_success=values[0]=='feminine'))
        for context,noise in product(('a','b'),range(2)):
            identifier='neutral_'+context+str(noise)
            generated[identifier]=dict(id=identifier,kind='neutral',context=context,
                common=dict(appearance={'presentation':CATEGORIES['presentation'][noise]}))
        base=dict(g=generated,u={'u':{'common_coverage':{'value':.5}}})
        indexed={('base',None):base,**{('T1',seed):base for seed in SEEDS}}
        explicit=explicit_arrays(indexed,'base','common','joint_success')
        self.assertEqual(explicit.shape,(3,2,2,18));self.assertEqual(explicit.mean(),.5)
        np.testing.assert_array_equal(explicit,explicit_arrays(indexed,'T1','common','joint_success'))
        neutral=neutral_arrays(indexed,'base','common','presentation')
        self.assertEqual(neutral.shape,(3,2,2))
        np.testing.assert_array_equal(neutral[0],[[0,1],[0,1]])
        np.testing.assert_array_equal(u_arrays(indexed,'base',['u'],'coverage:all'),[[.5],[.5],[.5]])


if __name__=='__main__':
    unittest.main()
