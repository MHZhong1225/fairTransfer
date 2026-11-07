"""BAGEL I→T supervision on all U attention/MLP LoRA, native G inference."""
from contextlib import contextmanager
from pathlib import Path
import json

from common import STUDY
import torch
from torch.nn import functional as F

from scripts.understanding_generation_sft_v3.backend import (
    BagelSFT, ValueLoRA, GENERATION, PREPROCESSING, checkpoint_decoders, tensor_hash)
from scripts.shared_prefix_fairness.prefix_forward import append_image, append_text, token_logits

U_SITES = ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
           'mlp.gate_proj','mlp.up_proj','mlp.down_proj')


class DetailBagel(BagelSFT):
    def install(self, seed, rank=8, alpha=16):
        """Same seven U-only projections in every layer for every training group."""
        if self.modules:
            raise ValueError('Install adapters once per model; load checkpoints thereafter')
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.model.requires_grad_(False)
        self.model.language_model.config.freeze_und = False
        self.model.config.llm_config.freeze_und = False
        for index, layer in enumerate(self.model.language_model.model.layers):
            layer.freeze_und = False
            layer.self_attn.config.freeze_und = False
            for site in U_SITES:
                parent, attribute = site.rsplit('.',1)
                owner = layer.get_submodule(parent)
                module = ValueLoRA(getattr(owner,attribute),generator,rank=rank)
                module.alpha = float(alpha)
                setattr(owner,attribute,module)
                self.modules[f'language_model.model.layers.{index}.{site}'] = module
        self.trainable = {name:p for name,p in self.model.named_parameters() if p.requires_grad}
        self.scope = 'u_attention_mlp'
        return self.trainable

    def caption_logits(self, row):
        """One causal teacher-forced answer query; no prompt tokens in the loss."""
        with torch.autocast('cuda',dtype=torch.bfloat16), checkpoint_decoders(self.model,self.activation_checkpointing):
            context = self.native.init_gen_context()
            if row.get('image_path'):
                append_image(self.native,self._image(row),context)
            append_text(self.native,row['question'],context)
            inputs = [self.native.new_token_ids['bos_token_id']] + self.target_ids(row)[:-1]
            return token_logits(self.native,context,inputs)

    def token_loss(self, row):
        targets = torch.tensor(self.target_ids(row),device=self.device)
        logits = self.caption_logits(row)
        return F.cross_entropy(logits,targets,reduction='sum'),len(targets)

    @torch.no_grad()
    def answer(self, row, max_length=256):
        """Official context construction and native greedy answer decoding."""
        context = self.native.init_gen_context()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if row.get('image_path'):
                context = self.native.update_context_image(self._image(row),context,vae=False,vit=True)
            context = self.native.update_context_text(row['question'],context)
            return self.native.gen_text(context,max_length=max_length,do_sample=False)

    @contextmanager
    def zero_updates(self):
        """Temporary exact base condition, retaining identical adapter structure."""
        saved = {name:module.b.detach().clone() for name,module in self.modules.items()}
        with torch.no_grad():
            for module in self.modules.values():
                module.b.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                for name,module in self.modules.items():
                    module.b.copy_(saved[name])

    def load(self, directory):
        metadata = json.loads((Path(directory)/'adapter.json').read_text())
        current = {name:{'rank':m.rank,'alpha':m.alpha} for name,m in self.modules.items()}
        if metadata['sites'] != current:
            raise ValueError('Checkpoint adapter locations or scaling differ')
        return super().load(directory)


def frozen_versions(model):
    return {name:p._version for name,p in model.named_parameters() if not p.requires_grad}


def check_frozen(model, versions):
    parameters = dict(model.named_parameters())
    changed = [name for name,version in versions.items() if parameters[name]._version!=version]
    gradients = [name for name in versions if parameters[name].grad is not None]
    if changed or gradients:
        raise ValueError(f'Frozen parameters changed or received gradients: {changed}, {gradients}')
    return dict(parameters=len(versions),unchanged=True,gradients_absent=True)
