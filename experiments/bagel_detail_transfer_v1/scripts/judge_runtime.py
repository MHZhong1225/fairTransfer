"""Independent visual/text evaluation with original model preprocessing."""
import torch


def load(judge):
    if judge=='qwen':
        from scripts.umm_u_training_v1.review import _load_judge
        model,processor=_load_judge('qwen','cuda:0')
        processor.tokenizer.padding_side='left'
        return model,processor
    from ministral_judge import load as load_ministral
    return load_ministral()


def respond(model,processor,judge,images,prompts,limit=4096):
    if judge=='qwen':
        conversations=[]
        for image,prompt in zip(images,prompts,strict=True):
            content=[] if image is None else [dict(type='image',image=image)]
            conversations.append([dict(role='user',content=content+[dict(type='text',text=prompt)])])
        inputs=processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',padding=True).to(model.device)
        with torch.inference_mode():
            output=model.generate(**inputs,do_sample=False,max_new_tokens=limit)
        return processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
    from ministral_batch import respond_batch
    from torch.nn import functional as F
    results=[None]*len(images)
    for with_image in (True,False):
        indices=[i for i,image in enumerate(images) if (image is not None)==with_image]
        if not indices:
            continue
        if with_image:
            output=respond_batch(model,processor,[images[i] for i in indices],[prompts[i] for i in indices],limit)
        else:
            items=[processor.apply_chat_template([dict(role='user',content=prompts[i])],
                return_tensors='pt',return_dict=True) for i in indices]
            width=max(item['input_ids'].shape[1] for item in items)
            ids=torch.cat([F.pad(item['input_ids'],(width-item['input_ids'].shape[1],0),value=processor.pad_token_id) for item in items]).to(model.device)
            mask=torch.cat([F.pad(torch.ones_like(item['input_ids']),(width-item['input_ids'].shape[1],0)) for item in items]).to(model.device)
            with torch.inference_mode():
                tokens=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=limit)
            output=[processor.decode(row[width:],skip_special_tokens=True) for row in tokens]
        for index,text in zip(indices,output,strict=True):
            results[index]=text
    return results
