"""Small arithmetic fixtures validate denominators; they are not experiment data."""
import unittest
import numpy as np
from metrics import distribution_difference,u_case_metrics,paired_explicit_interval


class MetricChecks(unittest.TestCase):
    def test_unknowns_are_not_balanced_votes(self):
        result=distribution_difference(['feminine','masculine',None,None],'presentation')
        self.assertEqual(result['planned'],4);self.assertEqual(result['known'],2)
        self.assertEqual(result['unknown'],2);self.assertEqual(result['rd'],0.)
        self.assertIsNone(distribution_difference([None,None],'presentation')['rd'])

    def test_ranges_differ_by_number_of_groups(self):
        self.assertEqual(distribution_difference(['feminine']*3,'presentation')['rd'],1.)
        self.assertAlmostEqual(distribution_difference(['light']*3,'skin_tone')['rd'],2/3)

    def test_empty_answer_does_not_have_perfect_precision(self):
        result=u_case_metrics(dict(status='completed',parsed=dict(reference_reviews=[dict(status='OMITTED')],claims=[])))
        self.assertEqual(result['coverage'],0.);self.assertIsNone(result['claim_precision'])
        self.assertFalse(u_case_metrics(dict(status='parse_failed'))['scored'])

    def test_paired_resampling_keeps_exact_cancellation(self):
        values=np.arange(3*4*2*6,dtype=float).reshape(3,4,2,6)
        result=paired_explicit_interval(values,values,draws=100)
        self.assertEqual(result['difference'],0.);self.assertEqual(result['interval'],[0.,0.])
        constant=paired_explicit_interval(values+2,values,draws=100)
        self.assertEqual(constant['interval'],[2.,2.])


if __name__=='__main__':
    unittest.main()
