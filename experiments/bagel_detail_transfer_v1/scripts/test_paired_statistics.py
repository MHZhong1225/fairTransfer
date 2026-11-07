"""Arithmetic fixtures, not experimental model outcomes."""
import unittest
import numpy as np
from paired_statistics import paired_u_interval,paired_neutral_interval,neutral_rd_array
from metrics_v2 import g_case_metrics


class StatisticsTests(unittest.TestCase):
    def test_image_cluster_pairing_and_constant_effect(self):
        second=np.array([[.1,.3,.4,.5],[.2,.3,.5,.6],[.1,.2,.3,.5]])
        result=paired_u_interval(second+.1,second,['a','a','b','c'],draws=400)
        np.testing.assert_allclose(result['interval'],[.1,.1],atol=1e-14)
        self.assertEqual(result['matched_defined_images_per_seed'],[4]*3)
        self.assertEqual(result['units']['source_clusters'],3)

    def test_undefined_claim_precision_is_not_perfect(self):
        first=np.array([[np.nan,.4],[.3,np.nan]])
        second=np.array([[.9,.2],[.1,.8]])
        result=paired_u_interval(first,second,['a','b'],draws=400)
        self.assertEqual(result['matched_defined_images_per_seed'],[1,1])
        self.assertGreater(result['invalid_draws'],0)
        np.testing.assert_allclose(result['interval'],[.2,.2])

    def test_neutral_pairing_and_undefined_cells(self):
        labels=np.array([[[0,1,0,1],[1,1,0,0]],[[0,0,0,1],[1,1,1,0]]])
        result=paired_neutral_interval(labels,labels,'presentation',draws=400)
        self.assertEqual(result['interval'],[0.,0.])
        missing=np.full_like(labels,-1)
        result=paired_neutral_interval(missing,labels,'presentation',draws=400)
        self.assertIsNone(result['difference']); self.assertIsNone(result['interval'])
        self.assertEqual(result['invalid_draws'],400)
        values,counts=neutral_rd_array(np.array([[0,0,-1],[0,1,-1]]),2)
        np.testing.assert_allclose(values,[1,0]); np.testing.assert_array_equal(counts,[2,2])

    def test_sculpture_cannot_count_as_success(self):
        item=dict(kind='explicit',requested_appearance=dict(presentation='feminine',age_band='young_adult',skin_tone='light'))
        appearance=dict(status='completed',parsed=dict(people='ONE',recognizable_person='YES',
            major_anatomy_error='NO',scene_details_visible='YES',framing='upper_body',
            appearance={k:dict(value=v,visibility='clear') for k,v in item['requested_appearance'].items()}))
        content=dict(status='completed',parsed=dict(reviews=[dict(id=k,supported='YES') for k in ('setting','action','object')]))
        statue=dict(status='completed',parsed=dict(category='SCULPTURE_OR_MANNEQUIN'))
        result=g_case_metrics(item,statue,appearance,content)
        self.assertFalse(result['joint_success']); self.assertFalse(result['valid_person'])
        self.assertTrue(all(x is None for x in result['appearance'].values()))


if __name__=='__main__':
    unittest.main()
