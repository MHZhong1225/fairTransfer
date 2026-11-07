"""Measure repeatability before attributing any BF16 gradient discrepancy."""
from contextlib import contextmanager
import json
from common import STUDY,now,rows,sha,write
from bagel_backend import DetailBagel
import torch
from safetensors.torch import save_file


@contextmanager
def attention_determinism(enabled):
    import modeling.bagel.qwen2_navit as qwen
    previous=qwen.flash_attn_varlen_func
    def attention(*args,**kwargs):
        kwargs['deterministic']=enabled
        return previous(*args,**kwargs)
    qwen.flash_attn_varlen_func=attention
    try:
        yield
    finally:
        qwen.flash_attn_varlen_func=previous


def compare(a,b):
    result=[]
    for name in a:
        if a[name].norm()>0:
            d=a[name]-b[name]
            result.append(dict(name=name,max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a[name].norm()),
                cosine=float(torch.nn.functional.cosine_similarity(a[name].flatten(),b[name].flatten(),dim=0))))
    return dict(exact=sum(torch.equal(a[k],b[k]) for k in a),total=len(a),
        largest=sorted(result,key=lambda x:-x['relative_l2'])[:8])


def main():
    torch.set_num_threads(4)
    directory=STUDY/'results/gradient_diagnosis_v1';directory.mkdir(exist_ok=False)
    item=rows(STUDY/'data/technical_manifest.jsonl')[17]
    row=dict(image_path=item['image_path'],question='Describe the visible person and background.',
        caption='A person is shown in the image. The face and clothing are visible against a background.')
    write(directory/'contract.json',dict(created_utc=now(),code_sha256=sha(__file__),row=row,
        purpose='Technical repeated-backward diagnostic; no optimizer updates and no formal image.'))
    backend=DetailBagel();backend.install(7100)
    observations={};gradients={}
    for deterministic in (False,True):
        for checkpoint in (False,True):
            for repeat in range(2):
                label=f'det{int(deterministic)}_ckpt{int(checkpoint)}_rep{repeat}'
                backend.activation_checkpointing=checkpoint
                backend.model.zero_grad(set_to_none=True)
                with attention_determinism(deterministic):
                    loss=backend.caption_loss(row);loss.backward()
                gradients[label]={k:p.grad.detach().cpu().clone() for k,p in backend.trainable.items()}
                observations[label]=dict(loss=float(loss.detach()))
                save_file(gradients[label],str(directory/(label+'.safetensors')))
                del loss
                backend.model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    for deterministic in (False,True):
        prefix=f'det{int(deterministic)}_'
        for ca,cb in ((0,0),(1,1),(0,1)):
            a=prefix+f'ckpt{ca}_rep0';b=prefix+f'ckpt{cb}_rep1'
            observations[a+'__'+b]=compare(gradients[a],gradients[b])
    write(directory/'result.json',dict(status='completed',completed_utc=now(),observations=observations))


if __name__=='__main__':
    main()
