"""Protect fact provenance, missing evidence and ambiguous demographic labels."""
import json
import unittest
from annotate_facts_v3 import parse_proposal,parse_verification


def fact(field,value):
    return dict(field=field,value=value,statement='A test statement.',region='face',visibility='clear')


def proposal(facts):
    return json.dumps(dict(people='ONE',adult_subject='YES',work_action_visible='NO',tool_visible='NO',facts=facts))


class FactSchemaTests(unittest.TestCase):
    def test_unsupported_field_does_not_remove_other_image_facts(self):
        result=parse_proposal(proposal([fact('hair','short'),fact('invented_category','value')]))
        self.assertEqual([f['field'] for f in result['facts']],['hair'])
        self.assertEqual(len(result['rejected_facts']),1)

    def test_conflicting_demographic_values_are_both_excluded(self):
        result=parse_proposal(proposal([fact('presentation','feminine'),fact('presentation','masculine'),fact('hair','short')]))
        self.assertEqual([f['field'] for f in result['facts']],['hair'])
        self.assertEqual(len(result['rejected_facts']),2)

    def test_distinct_objects_are_preserved_but_exact_duplicates_not(self):
        a=fact('accessory','glasses');b=fact('accessory','earring')
        result=parse_proposal(proposal([a,b,a]))
        self.assertEqual(len(result['facts']),2)
        self.assertEqual(len(result['rejected_facts']),1)

    def test_uncertain_evidence_never_becomes_a_positive_fact(self):
        a=fact('skin_tone','light');a['visibility']='uncertain'
        self.assertEqual(parse_proposal(proposal([a]))['facts'],[])

    def test_every_fact_needs_its_own_review(self):
        result=dict(people='ONE',adult_subject='YES',work_action_visible='NO',tool_visible='NO',
                    reviews=[dict(id=0,supported='YES',evidence='visible')])
        with self.assertRaises(ValueError):
            parse_verification(json.dumps(result),2)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_proposal('{"people":"ONE","people":"NONE"}')


if __name__=='__main__':
    unittest.main()
