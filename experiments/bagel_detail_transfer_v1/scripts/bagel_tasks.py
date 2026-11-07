"""One visual exposure with separate question contexts and text-only targets."""
from copy import copy
import torch
from torch.nn import functional as F
from bagel_backend_v2 import DetailBagel,deterministic_attention_backward
from bagel_backend import checkpoint_decoders
from scripts.shared_prefix_fairness.prefix_forward import append_image,append_text,token_logits


def fork_image_context(context):
    """Share differentiable image tensors; isolate each branch's cache containers."""
    branch=dict(context)
    branch['kv_lens']=list(context['kv_lens'])
    branch['ropes']=list(context['ropes'])
    branch['past_key_values']=copy(context['past_key_values'])
    branch['past_key_values'].key_cache=dict(context['past_key_values'].key_cache)
    branch['past_key_values'].value_cache=dict(context['past_key_values'].value_cache)
    return branch


class TaskBagel(DetailBagel):
    def pack_logits(self,pack):
        """No question branch can read any other branch's prompt or gold answer."""
        with torch.autocast('cuda',dtype=torch.bfloat16), deterministic_attention_backward(), \
                checkpoint_decoders(self.model,self.activation_checkpointing):
            context=self.native.init_gen_context()
            if pack.get('image_path'):
                append_image(self.native,self._image(pack),context)
            outputs=[]
            for task in pack['tasks']:
                branch=fork_image_context(context)
                append_text(self.native,task['question'],branch)
                targets=self.target_ids(task)
                inputs=[self.native.new_token_ids['bos_token_id']]+targets[:-1]
                outputs.append(token_logits(self.native,branch,inputs))
            return outputs

    def pack_loss(self,pack):
        outputs=self.pack_logits(pack)
        total=None;count=0
        for task,logits in zip(pack['tasks'],outputs,strict=True):
            targets=torch.tensor(self.target_ids(task),device=self.device)
            loss=F.cross_entropy(logits,targets,reduction='sum')
            total=loss if total is None else total+loss
            count+=len(targets)
        if total is None or not count:
            raise ValueError('An image exposure needs at least one supervised target')
        return total/count,count
