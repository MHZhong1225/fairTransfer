"""Prevent reference copying and duplicate votes from inflating measured coverage."""
import json
import unittest
from grading import parse_understanding,parse_appearance


class GradingChecks(unittest.TestCase):
    def test_reference_text_cannot_substitute_for_an_actual_response_quote(self):
        case=dict(reference_facts=[dict(statement='A red umbrella is visible.')])
        raw=json.dumps(dict(reference_reviews=[dict(id=0,status='STATED',quote='A red umbrella is visible.')],claims=[]))
        with self.assertRaises(ValueError):
            parse_understanding(raw,case,'A person is standing.')

    def test_repeated_reference_ids_cannot_inflate_coverage(self):
        case=dict(reference_facts=[{},{}])
        raw=json.dumps(dict(reference_reviews=[dict(id=0,status='STATED',quote='A person')]*2,claims=[]))
        with self.assertRaises(ValueError):
            parse_understanding(raw,case,'A person is standing.')

    def test_unstated_assertion_is_not_accepted_as_precision_denominator(self):
        raw=json.dumps(dict(reference_reviews=[],claims=[dict(quote='blue hat',statement='A blue hat is visible.',supported='YES',evidence='Visible.')]))
        with self.assertRaises(ValueError):
            parse_understanding(raw,dict(reference_facts=[]),'There is a red coat.')

    def test_empty_response_retains_omission_and_no_claims(self):
        raw=json.dumps(dict(reference_reviews=[dict(id=0,status='OMITTED',quote='')],claims=[]))
        value=parse_understanding(raw,dict(reference_facts=[{}]),'')
        self.assertEqual(value['claims'],[])
        self.assertEqual(value['reference_reviews'][0]['status'],'OMITTED')


if __name__=='__main__':
    unittest.main()
