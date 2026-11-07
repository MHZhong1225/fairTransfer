"""Separate image-medium checks prevent sculpture material becoming a skin label."""
from annotate_facts_v3 import json_object

PROMPT='''What kind of subject is actually depicted in this image? Inspect material, eyes, hair, clothing and scene evidence. Do not classify demographics or infer occupation. A human-shaped sculpture is not a natural-looking human person.
Return only one JSON object with exactly category and evidence.
category must be one of:
PHOTO_LIKE_PERSON: a photographic or photorealistic depiction of a natural-looking human person. AI-generated photorealistic people count here; their existence in the real world is irrelevant. A partially cropped or obscured real-looking person may count here without a clear face.
SCULPTURE_OR_MANNEQUIN: a statue, carved bust, doll or mannequin; look for sculpted strands, uniform plaster/marble material, blank carved eyes, artificial joints or clearly rigid sculptural surfaces. Do not label white stone as human skin.
ILLUSTRATED_PERSON: a drawn, painted, cartoon or clearly non-photorealistic illustration of a human.
NO_PERSON: neither a person nor a human-like depiction is visible.
UNCERTAIN: the depiction type cannot be determined confidently.
evidence must be one concise sentence about actual visible material or depiction cues. Do not infer that an ordinary smooth face is a statue merely because it is attractive or may be AI-generated. Use UNCERTAIN when the distinction is unclear.'''
CATEGORIES=('PHOTO_LIKE_PERSON','SCULPTURE_OR_MANNEQUIN','ILLUSTRATED_PERSON','NO_PERSON','UNCERTAIN')


def parse(raw):
    value=json_object(raw)
    if set(value)!={'category','evidence'} or value['category'] not in CATEGORIES or not isinstance(value['evidence'],str):
        raise ValueError('Invalid depiction judgment')
    return value
