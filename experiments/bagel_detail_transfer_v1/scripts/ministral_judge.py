"""Independent BF16 visual verifier using the documented Mistral tokenizer."""
import base64
import io
from pathlib import Path

MODEL = Path('/home/ubuntu/zmh/models/Ministral-3-8B-Instruct-2512-BF16')


def load():
    import torch
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
    tokenizer = MistralCommonBackend.from_pretrained(str(MODEL), local_files_only=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(str(MODEL),
        dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', local_files_only=True).eval()
    return model, tokenizer


def respond(model, tokenizer, image, prompt, limit=2048):
    import torch
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    messages = [{'role':'user','content':[{'type':'text','text':prompt},
        {'type':'image_url','image_url':{'url':url}}]}]
    inputs = tokenizer.apply_chat_template(messages,return_tensors='pt',return_dict=True)
    inputs['input_ids'] = inputs['input_ids'].to(model.device)
    if 'attention_mask' in inputs:
        inputs['attention_mask'] = inputs['attention_mask'].to(model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(model.device,dtype=torch.bfloat16)
    sizes = [inputs['pixel_values'].shape[-2:]]
    with torch.inference_mode():
        output = model.generate(**inputs,image_sizes=sizes,do_sample=False,max_new_tokens=limit)
    ids = output[0,inputs['input_ids'].shape[1]:]
    return tokenizer.decode(ids,skip_special_tokens=True),len(ids),len(ids)>=limit
