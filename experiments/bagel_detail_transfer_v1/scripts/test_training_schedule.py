"""Check randomized exposure order preserves the already audited fact/token budget."""
from collections import Counter
import unittest
from common import STUDY,rows
from supervision import make_pack
from training_schedule import training_pack


def targets(packs):
    return Counter((task['question'],task['caption']) for pack in packs for task in pack['tasks'])


class ScheduleChecks(unittest.TestCase):
    def test_same_targets_and_same_common_exposure_for_all_arms(self):
        records=rows(STUDY/'data/technical_fact_bank_v1/records.jsonl')
        qa_epochs=set()
        for row in records:
            for arm in ('T0','T1','T2','T3'):
                before=[make_pack(row,arm,e) for e in range(3)]
                after=[training_pack(row,arm,e) for e in range(3)]
                self.assertEqual(targets(before),targets(after))
                self.assertEqual(sorted(p['supervision_variant'] for p in after),[0,1,2])
                if arm=='T3':
                    qa_epochs.update(p['physical_epoch'] for p in after if p['tasks'][0]['kind']=='qa')
            for epoch in range(3):
                self.assertEqual(len({training_pack(row,arm,epoch)['supervision_variant'] for arm in ('T0','T1','T2','T3')}),1)
        self.assertEqual(qa_epochs,{0,1,2})


if __name__=='__main__':
    unittest.main()
