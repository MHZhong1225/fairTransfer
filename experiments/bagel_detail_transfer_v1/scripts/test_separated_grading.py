"""Data-flow checks, not experimental model observations."""
import json
import unittest
from unittest.mock import patch
from understanding_grading_v4 import score_batch


class SeparatedGradingTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(reference_facts=[dict(field='accessory',value='glasses',statement='REFERENCE_ONLY_MARKER')])
        self.answer='The person wears glasses.'

    def test_empty_answer_does_not_invent_observations(self):
        with patch('understanding_grading_v4.respond') as model:
            result=score_batch(None,None,'qwen',[object()],[self.case],[''],lambda *args:None)[0]
        model.assert_not_called()
        self.assertEqual(result['parsed'],dict(reference_reviews=[dict(id=0,status='OMITTED',quote='')],claims=[]))

    def test_reference_text_never_reaches_visual_claim_check(self):
        pixels=object();calls=[];saved=[]
        outputs=[dict(reference_reviews=[dict(id=0,status='STATED',quote=self.answer)]),
            dict(claims=[dict(quote=self.answer,statement='The person wears glasses.')]),
            dict(reviews=[dict(id=0,supported='YES',evidence='Visible glasses.')])]
        def respond(model,processor,judge,images,prompts):
            calls.append((images,prompts))
            return [json.dumps(outputs[len(calls)-1])]
        with patch('understanding_grading_v4.respond',side_effect=respond):
            value=score_batch(None,None,'qwen',[pixels],[self.case],[self.answer],lambda *args:saved.append(args))[0]
        self.assertNotIn('error',value)
        self.assertEqual(calls[0][0],[None]);self.assertEqual(calls[1][0],[None])
        self.assertIs(calls[2][0][0],pixels)
        self.assertIn('REFERENCE_ONLY_MARKER',calls[0][1][0])
        self.assertNotIn('REFERENCE_ONLY_MARKER',calls[1][1][0])
        self.assertNotIn('REFERENCE_ONLY_MARKER',calls[2][1][0])
        self.assertEqual([s[1] for s in saved],['coverage','extraction','support'])

    def test_reference_copy_fails_and_raw_is_saved(self):
        raw=json.dumps(dict(reference_reviews=[dict(id=0,status='STATED',quote='REFERENCE_ONLY_MARKER')]))
        saved=[]
        with patch('understanding_grading_v4.respond',return_value=[raw]) as model:
            value=score_batch(None,None,'qwen',[object()],[self.case],[self.answer],lambda *args:saved.append(args))[0]
        self.assertIn('error',value);self.assertEqual(model.call_count,1)
        self.assertEqual(saved[0][2]['response'],raw)

    def test_no_reference_abstention_needs_no_reference_or_image_judgment(self):
        with patch('understanding_grading_v4.respond',return_value=['{"claims":[]}']) as model:
            value=score_batch(None,None,'qwen',[None],[dict(reference_facts=[])],
                ['I cannot determine this without an image.'],lambda *args:None)[0]
        self.assertEqual(model.call_count,1)
        self.assertEqual(value['parsed'],dict(reference_reviews=[],claims=[]))


if __name__=='__main__':
    unittest.main()
