"""Literal evidence selection checks; fixture outputs are not research results."""
import json
import unittest
from unittest.mock import patch
from understanding_grading_v5 import answer_spans,score_batch


class AnswerSpanGradingTests(unittest.TestCase):
    def test_spans_keep_original_text(self):
        answer='  A person sits.\nThey hold a cup\nA final phrase'
        spans=answer_spans(answer)
        self.assertEqual([x['id'] for x in spans],[0,1,2])
        self.assertTrue(all(x['text'] in answer for x in spans))

    def test_evidence_is_selected_not_regenerated(self):
        answer='A person sits and holds a cup.'
        case=dict(question='What is the person holding?',reference_facts=[
            dict(field='held_object',value='cup',statement='REFERENCE_ONLY')])
        responses=[dict(reference_reviews=[dict(id=0,status='STATED',answer_span=0)]),
            dict(claims=[dict(answer_span=0,statement='A person holds a cup.')]),
            dict(reviews=[dict(id=0,supported='YES',evidence='A visible cup.')])]
        calls=[]
        def respond(model,processor,judge,images,prompts):
            calls.append((images,prompts));return [json.dumps(responses[len(calls)-1])]
        with patch('understanding_grading_v5.respond',side_effect=respond):
            value=score_batch(None,None,'qwen',[object()],[case],[answer],lambda *args:None)[0]
        self.assertNotIn('error',value)
        self.assertEqual(value['parsed']['claims'][0]['quote'],answer)
        self.assertIn(case['question'],calls[0][1][0]);self.assertIn(case['question'],calls[1][1][0])
        self.assertNotIn('REFERENCE_ONLY',calls[1][1][0]);self.assertNotIn('REFERENCE_ONLY',calls[2][1][0])

    def test_invented_span_fails_before_visual_check(self):
        case=dict(reference_facts=[dict(field='activity',value='sit',statement='The person sits.')])
        raw=json.dumps(dict(reference_reviews=[dict(id=0,status='STATED',answer_span=4)]));saved=[]
        with patch('understanding_grading_v5.respond',return_value=[raw]) as model:
            result=score_batch(None,None,'qwen',[object()],[case],['The person sits.'],lambda *args:saved.append(args))[0]
        self.assertIn('error',result);self.assertEqual(model.call_count,1)
        self.assertEqual(saved[0][2]['response'],raw)

    def test_empty_remains_omitted_without_model_fabrication(self):
        case=dict(reference_facts=[dict(field='activity',value='sit',statement='The person sits.')])
        with patch('understanding_grading_v5.respond') as model:
            result=score_batch(None,None,'qwen',[object()],[case],[' '],lambda *args:None)[0]
        model.assert_not_called();self.assertEqual(result['parsed']['claims'],[])
        self.assertEqual(result['parsed']['reference_reviews'][0]['status'],'OMITTED')


if __name__=='__main__':
    unittest.main()
