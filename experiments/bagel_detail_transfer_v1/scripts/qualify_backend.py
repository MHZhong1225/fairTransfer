"""Real full-input BAGEL checks before formal training; no scientific labels."""
import gc
from pathlib import Path
import time
import traceback

from common import ROOT,STUDY,BAGEL_REPO,BAGEL_WEIGHTS,now,read,rows,sha,write
import torch
from safetensors.torch import save_file
from bagel_backend import DetailBagel,GENERATION,PREPROCESSING,U_SITES,frozen_versions,check_frozen,tensor_hash
from scripts.understanding_generation_sft_v3.backend import BagelSFT
from scripts.shared_prefix_fairness.prefix_forward import append_text


def difference(a,b):
    delta = (a.detach().float()-b.detach().float()).abs()
    return dict(exact=torch.equal(a,b),maximum=float(delta.max()),mean=float(delta.mean()))


def prefix_fingerprint(backend):
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        context = backend.native.init_gen_context()
        append_text(backend.native,'A photograph of a person holding a red umbrella.',context)
        cache = context['past_key_values']
        return {f'{kind}/{index}':tensor_hash(values[index])
                for kind,values in [('key',cache.key_cache),('value',cache.value_cache)] for index in values}


def main():
    directory = STUDY/'results/backend_qualification_v1'
    directory.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    technical = rows(STUDY/'data/technical_manifest.jsonl')[0]
    row = dict(image_path=technical['image_path'],question='Describe the visible person and background.',
        caption='A person is shown in the image. The face and clothing are visible against a background.')
    dependencies = [Path(__file__),Path(__file__).with_name('bagel_backend.py'),Path(__file__).with_name('common.py'),
        ROOT/'scripts/understanding_generation_sft_v3/backend.py']
    dependencies += [ROOT/'scripts/shared_prefix_fairness'/name for name in ('runtime.py','config.py','prefix_forward.py','adapters.py')]
    dependencies += [BAGEL_REPO/name for name in ('inferencer.py','modeling/bagel/bagel.py','modeling/bagel/qwen2_navit.py','data/transforms.py','data/configs/example.yaml')]
    contract = dict(created_utc=now(),seed=7100,rank=8,alpha=16,lr=1e-4,
        task='Technical text cross-entropy only, excluded BPO image; literal caption is a numerical fixture, not an image truth claim.',
        row=row,image_sha256=sha(row['image_path']),generation=GENERATION,preprocessing=PREPROCESSING,
        sites=U_SITES,model_revision=BAGEL_WEIGHTS.name,source_hashes={str(p):sha(p) for p in dependencies},
        acceptance=dict(zero_adapter_exact=True,bulk_vs_native_max_abs=.15,bulk_vs_native_mean_abs=.015,
            recomputation_forward_exact=True,gradient_rtol=1e-4,gradient_atol=1e-6,
            all_B_gradients_finite_positive=True,frozen_gradients_absent=True,
            save_reload_weights_and_outputs_exact=True,actual_G_prefix_changed=True,
            restored_G_pixels_and_noise_exact=True))
    write(directory/'contract.json',contract)
    official = read(ROOT/'experiments/official_generation_check_20261031/bagel/result.json')
    if GENERATION != official['settings']:
        raise ValueError('Official generation settings changed')
    status = dict(status='running',started_utc=now(),checks={})
    write(directory/'status.json',status)
    try:
        backend = DetailBagel(device='cuda:0',understanding_mode='train',activation_checkpointing=True)
        with torch.no_grad():
            base = backend.caption_logits(row).cpu()
        backend.install(7100)
        versions = frozen_versions(backend.model)
        with torch.no_grad():
            zero = backend.caption_logits(row).cpu()
            native = backend.native_caption_logits(row).cpu()
            sequential = BagelSFT.caption_logits(backend,row).cpu()
        status['checks']['zero_adapter'] = difference(base,zero)
        status['checks']['native_alignment'] = difference(zero,native)
        status['checks']['sequential_alignment'] = difference(zero,sequential)
        save_file({'base':base,'zero_adapter':zero,'native_forced':native,'sequential_teacher':sequential},str(directory/'initial_logits.safetensors'))
        if not torch.equal(base,zero):
            raise ValueError('Zero adapter changed base output')
        alignment = status['checks']['native_alignment']
        if alignment['maximum']>.15 or alignment['mean']>.015:
            raise ValueError('Bulk teacher forcing failed native alignment')
        write(directory/'status.json',status)
        comparisons = {}
        for enabled in (False,True):
            backend.activation_checkpointing = enabled
            backend.model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                logits = backend.caption_logits(row)
                targets = torch.tensor(backend.target_ids(row),device=backend.device)
                loss = torch.nn.functional.cross_entropy(logits,targets)
                loss.backward()
                torch.cuda.synchronize()
                gradients = {name:p.grad.detach().cpu().clone() for name,p in backend.trainable.items()}
                comparisons[enabled] = dict(logits=logits.detach().cpu(),gradients=gradients)
                name = 'checkpoint_on' if enabled else 'checkpoint_off'
                save_file(gradients,str(directory/(name+'_gradients.safetensors')))
                status['checks'][name] = dict(status='completed',loss=float(loss),seconds=time.perf_counter()-started,
                    peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    positive_B=sum(bool(torch.isfinite(grad).all() and grad.norm()>0) for key,grad in gradients.items() if key.endswith('.b')),
                    expected_B=len(backend.modules),frozen=check_frozen(backend.model,versions))
                del logits,loss
            except torch.cuda.OutOfMemoryError:
                status['checks']['checkpoint_off' if not enabled else 'checkpoint_on'] = dict(status='out_of_memory',error=traceback.format_exc())
                if 'logits' in locals():
                    del logits
                if 'loss' in locals():
                    del loss
                if enabled:
                    raise
            finally:
                backend.model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                write(directory/'status.json',status)
        if False in comparisons:
            a,b = comparisons[False],comparisons[True]
            exact_forward = torch.equal(a['logits'],b['logits'])
            gradient_match = all(torch.allclose(a['gradients'][name],b['gradients'][name],rtol=1e-4,atol=1e-6) for name in backend.trainable)
            status['checks']['recomputation_parity'] = dict(forward_exact=exact_forward,gradients_close=gradient_match)
            if not exact_forward or not gradient_match:
                raise ValueError('Activation recomputation changed forward/gradients')
        if status['checks']['checkpoint_on']['positive_B'] != len(backend.modules):
            raise ValueError('Not all U adapter B matrices received finite nonzero gradients')
        backend.activation_checkpointing = True
        prefix_before = prefix_fingerprint(backend)
        initial_hashes = {name:tensor_hash(p) for name,p in backend.trainable.items()}
        optimizer = torch.optim.AdamW(backend.trainable.values(),lr=1e-4,weight_decay=0.)
        steps = []
        for step in range(3):
            started = time.perf_counter();optimizer.zero_grad(set_to_none=True)
            loss = backend.caption_loss(row);loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(backend.trainable.values()),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite training value')
            optimizer.step();check_frozen(backend.model,versions)
            steps.append(dict(step=step+1,loss=float(loss),gradient_norm=float(norm),seconds=time.perf_counter()-started))
        optimizer.zero_grad(set_to_none=True)
        prefix_after = prefix_fingerprint(backend)
        changed = sum(prefix_before[k]!=prefix_after[k] for k in prefix_before)
        if not changed:
            raise ValueError('U updates did not affect native G text condition')
        status['checks']['training'] = dict(steps=steps,trainable_parameters=sum(p.numel() for p in backend.trainable.values()),
            modules=len(backend.modules),initial_parameter_hashes=initial_hashes,
            actual_generation_condition_cache_changes=changed,total_condition_tensors=len(prefix_before),
            frozen=check_frozen(backend.model,versions))
        backend.save(directory/'adapter')
        with torch.no_grad():
            trained_logits = backend.caption_logits(row).cpu()
        noise,generated = backend.render('A photograph of one adult person holding a red umbrella outdoors.',7100,directory/'trained.png')
        with torch.no_grad():
            for module in backend.modules.values():
                module.b.zero_()
        backend.load(directory/'adapter')
        with torch.no_grad():
            reloaded_logits = backend.caption_logits(row).cpu()
        _,restored = backend.render('A photograph of one adult person holding a red umbrella outdoors.',7199,directory/'restored.png',replay_noise=noise)
        status['checks']['reload'] = dict(logits_exact=torch.equal(trained_logits,reloaded_logits),
            generation_pixels_exact=generated['image_sha256']==restored['image_sha256'],
            generation_noise_exact=generated['noise_tensor_sha256']==restored['noise_tensor_sha256'])
        if not all(status['checks']['reload'].values()):
            raise ValueError('Saved checkpoint did not reproduce outputs')
        status.update(status='passed',completed_utc=now(),contract_sha256=sha(directory/'contract.json'))
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),completed_utc=now());raise
    finally:
        write(directory/'status.json',status)


if __name__ == '__main__':
    main()
