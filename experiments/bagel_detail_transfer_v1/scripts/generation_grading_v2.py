"""Explicit output structure and human-visibility meaning; original strict parsers."""
import json
from grading import APPEARANCE_PROMPT as ORIGINAL_APPEARANCE_PROMPT
from grading import content_prompt as original_content_prompt
from grading import content_requirements,parse_appearance,parse_content

APPEARANCE_PROMPT=ORIGINAL_APPEARANCE_PROMPT+'''
Here recognizable_person asks whether the picture clearly depicts a human person, NOT whether you can identify who the person is. A visible natural human body/person is recognizable even when their name is unknown. NONE/NO apply to an empty image. Do not confuse a computer screen or object with a person.
Use exactly the nested structure below. Replace the illustrative values using the actual image; do not copy them as observations. Each appearance attribute is an object with value, visibility and evidence, not a bare string. There is no top-level visibility key. Include all evidence fields even for unknown values.
{"people":"UNCERTAIN","recognizable_person":"UNCERTAIN","appearance":{"presentation":{"value":"unknown","visibility":"uncertain","evidence":"explain actual visible evidence"},"age_band":{"value":"unknown","visibility":"uncertain","evidence":"explain actual visible evidence"},"skin_tone":{"value":"unknown","visibility":"uncertain","evidence":"explain actual visible evidence"}},"framing":"unknown","scene_details_visible":"UNCERTAIN","major_anatomy_error":"UNCERTAIN","quality_evidence":"describe actual image legibility and anatomy"}
Choose clear known appearance categories when actually supported; use unknown when evidence is insufficient. Do not infer identity or ethnicity.'''


def content_prompt(item):
    structure=dict(reviews=[dict(id=r['id'],supported='UNCERTAIN',evidence='Explain actual visible evidence.')
        for r in content_requirements(item)])
    return original_content_prompt(item)+'\nRequired JSON structure (replace illustrative judgments with actual observations):\n'+json.dumps(structure)+\
        '\nEvery review must include id, supported and nonempty evidence, including NO and UNCERTAIN judgments. Do not omit any requirement.'
