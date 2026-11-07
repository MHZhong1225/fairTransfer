"""Independent chat inputs, left text padding and native image-size masking."""
import base64
import io


def respond_batch(model, tokenizer, images, prompts, limit=2048):
    import torch
    from torch.nn import functional as F
    items = []
    for image,prompt in zip(images,prompts,strict=True):
        buffer=io.BytesIO();image.convert('RGB').save(buffer,format='PNG')
        url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        messages=[{'role':'user','content':[{'type':'text','text':prompt},
            {'type':'image_url','image_url':{'url':url}}]}]
        items.append(tokenizer.apply_chat_template(messages,return_tensors='pt',return_dict=True))
    width=max(item['input_ids'].shape[1] for item in items)
    sizes=[item['pixel_values'].shape[-2:] for item in items]
    height=max(s[0] for s in sizes); image_width=max(s[1] for s in sizes)
    ids=torch.cat([F.pad(item['input_ids'],(width-item['input_ids'].shape[1],0),value=tokenizer.pad_token_id)
        for item in items]).to(model.device)
    mask=torch.cat([F.pad(torch.ones_like(item['input_ids']),(width-item['input_ids'].shape[1],0))
        for item in items]).to(model.device)
    pixels=torch.cat([F.pad(item['pixel_values'],(0,image_width-size[1],0,height-size[0]))
        for item,size in zip(items,sizes)]).to(model.device,dtype=torch.bfloat16)
    with torch.inference_mode():
        output=model.generate(input_ids=ids,attention_mask=mask,pixel_values=pixels,
            image_sizes=sizes,do_sample=False,max_new_tokens=limit)
    return [tokenizer.decode(row[width:],skip_special_tokens=True) for row in output]
