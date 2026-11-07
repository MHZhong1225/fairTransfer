"""Scientific invariants on the real excluded technical fact bank."""
import unittest
from collections import Counter
from common import STUDY,rows
from supervision import make_pack,DEMOGRAPHIC
from build_fact_bank import common_weights


class SupervisionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=rows(STUDY/'data/technical_fact_bank_v1/records.jsonl')

    def test_common_context_and_equal_fact_counts(self):
        for row in self.records:
            a,b=row['paired']['T1'],row['paired']['T2']
            self.assertEqual(len(a),len(b))
            self.assertFalse(any(f['field'] in DEMOGRAPHIC for f in a))
            shared=set(row['paired']['common_fact_ids'])
            self.assertTrue(shared)
            self.assertTrue(shared.issubset({f['source_fact_id'] for f in a}))
            self.assertTrue(shared.issubset({f['source_fact_id'] for f in b}))

    def test_qa_replaces_targets_without_duplicating_facts(self):
        actual_qa=0
        for row in self.records:
            for epoch in range(3):
                pack=make_pack(row,'T3',epoch)
                if pack['tasks'][0]['kind']=='qa':
                    actual_qa+=1
                    self.assertTrue(2<=len(pack['tasks'])<=4)
                    ids=[i for t in pack['tasks'] for i in t['fact_ids']]
                    self.assertEqual(Counter(ids),Counter(f['source_fact_id'] for f in row['paired']['T2']))
        self.assertGreater(actual_qa,0)

    def test_common_image_and_brief_exposure(self):
        for row in self.records:
            for epoch in range(3):
                packs=[make_pack(row,arm,epoch) for arm in ('T0','T1','T2','T3')]
                self.assertEqual(len({p['image_path'] for p in packs}),1)
                if packs[1]['tasks'][0]['kind']=='brief':
                    self.assertEqual(len({p['tasks'][0]['caption'] for p in packs}),1)

    def test_weights_preserve_context_mass_and_do_not_fill_missing_groups(self):
        records=[dict(id=str(i),split='train',content_stratum='same',framing='upper_body',
            work_action_visible='YES',appearance_labels={'presentation':'feminine' if i<3 else 'masculine'}) for i in range(12)]
        records+=[dict(id='missing',split='train',content_stratum='different',framing='full_body',
            work_action_visible='YES',appearance_labels={})]
        common_weights(records)
        self.assertAlmostEqual(sum(r['weight'] for r in records[:3]),6.)
        self.assertAlmostEqual(sum(r['weight'] for r in records[3:12]),6.)
        self.assertEqual(records[-1]['weight'],1.)
        self.assertEqual(records[-1]['appearance_labels'],{})


if __name__=='__main__':
    unittest.main()
