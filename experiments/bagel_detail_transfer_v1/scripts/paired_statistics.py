"""Paired inference with training seeds and observed source clusters retained."""
import numpy as np
from metrics import CATEGORIES


def summarize_draws(point, values, per_seed, seed, draws, confidence):
    finite = np.asarray(values)[np.isfinite(values)]
    alpha = (1 - confidence) / 2
    return dict(difference=point, interval=np.quantile(finite, [alpha, 1-alpha]).tolist() if len(finite) else None,
        per_training_seed=per_seed, valid_draws=len(finite), invalid_draws=draws-len(finite),
        draws=draws, seed=seed, confidence=confidence)


def paired_u_interval(first, second, clusters, seed=8723, draws=10000, confidence=.95):
    """Arrays are seed × image; undefined per-answer precision stays undefined.

    Comparison is on images with defined values for both methods within a seed.
    Missingness and all-source counts must accompany this conditional contrast.
    Each sampled source cluster retains all of its images and tasks. Equal seed
    weight is used, including when the count of defined images differs by seed.
    """
    first = np.asarray(first, dtype=float); second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != len(clusters):
        raise ValueError('Require paired seed × image values and one cluster per image')
    if not first.size or np.isinf(first).any() or np.isinf(second).any():
        raise ValueError('Empty or infinite observations')
    names = sorted(set(clusters)); columns = {name:i for i,name in enumerate(names)}
    valid = np.isfinite(first) & np.isfinite(second)
    delta = np.where(valid, first-second, 0.)
    sums = np.zeros((first.shape[0], len(names))); counts = np.zeros_like(sums)
    for image, cluster in enumerate(clusters):
        sums[:,columns[cluster]] += delta[:,image]
        counts[:,columns[cluster]] += valid[:,image]
    total_counts = counts.sum(1)
    per_seed = [float(sums[s].sum()/n) if n else None for s,n in enumerate(total_counts)]
    point = float(np.mean(per_seed)) if all(x is not None for x in per_seed) else None
    rng = np.random.default_rng(seed); values = np.full(draws,np.nan)
    for offset in range(0,draws,256):
        size = min(256,draws-offset)
        seed_weights = rng.multinomial(first.shape[0],np.full(first.shape[0],1/first.shape[0]),size)
        cluster_weights = rng.multinomial(len(names),np.full(len(names),1/len(names)),size)
        numerator = cluster_weights @ sums.T; denominator = cluster_weights @ counts.T
        means = np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        good = ((denominator>0)|(seed_weights==0)).all(1)
        values[offset:offset+size] = np.where(good,(means*seed_weights).sum(1)/first.shape[0],np.nan)
    result = summarize_draws(point,values,per_seed,seed,draws,confidence)
    result.update(units=dict(training_seeds=first.shape[0],images=first.shape[1],source_clusters=len(names)),
        matched_defined_images_per_seed=valid.sum(1).tolist(),
        first_defined_images_per_seed=np.isfinite(first).sum(1).tolist(),
        second_defined_images_per_seed=np.isfinite(second).sum(1).tolist(),
        estimand='Equal-seed mean of paired defined-image differences, source-cluster bootstrap; not an unconditional estimate when either method has undefined values.')
    return result


def neutral_rd_array(labels, categories):
    """Return RD over the final noise dimension, and known-image counts."""
    counts = np.stack([(labels==c).sum(axis=-1) for c in range(categories)],axis=-1)
    known = counts.sum(-1)
    shares = np.divide(counts,known[...,None],out=np.zeros_like(counts,dtype=float),where=known[...,None]>0)
    values = np.mean(np.stack([abs(shares[...,a]-shares[...,b]) for a in range(categories)
        for b in range(a+1,categories)],axis=-1),axis=-1)
    return np.where(known>0,values,np.nan),known


def paired_neutral_interval(first, second, field, seed=8723, draws=10000, confidence=.95):
    """Integer categories (unknown=-1), seed × occupation × matched noise.

    Recalculate proportions after paired resampling. A replicate with any
    sampled seed/occupation lacking known labels has an undefined full mean.
    """
    first = np.asarray(first,dtype=int); second = np.asarray(second,dtype=int)
    if first.shape!=second.shape or first.ndim!=3 or not first.size:
        raise ValueError('Require complete paired seed × occupation × noise arrays')
    categories = len(CATEGORIES[field])
    if any(((array < -1)|(array >= categories)).any() for array in (first,second)):
        raise ValueError('Invalid category code')
    seeds,contexts,noises = first.shape
    difference = neutral_rd_array(first,categories)[0]-neutral_rd_array(second,categories)[0]
    per_seed = [float(x.mean()) if np.isfinite(x).all() else None for x in difference]
    point = float(difference.mean()) if np.isfinite(difference).all() else None
    rng = np.random.default_rng(seed); values = np.full(draws,np.nan)
    for offset in range(0,draws,128):
        size = min(128,draws-offset)
        selected_seeds = rng.integers(seeds,size=(size,seeds))
        selected_contexts = rng.integers(contexts,size=(size,contexts))
        selected_noises = rng.integers(noises,size=(size,contexts,noises))
        index = (selected_seeds[:,:,None,None],selected_contexts[:,None,:,None],selected_noises[:,None,:,:])
        delta = neutral_rd_array(first[index],categories)[0]-neutral_rd_array(second[index],categories)[0]
        valid = np.isfinite(delta).all(axis=(1,2))
        values[offset:offset+size] = np.where(valid,delta.mean(axis=(1,2)),np.nan)
    result = summarize_draws(point,values,per_seed,seed,draws,confidence)
    result.update(units=dict(training_seeds=seeds,occupations=contexts,noises_per_occupation=noises),
        first_known_counts=neutral_rd_array(first,categories)[1].tolist(),
        second_known_counts=neutral_rd_array(second,categories)[1].tolist(),
        direction='Negative means lower conditional RD. Interpret with known-person coverage and generation utility.')
    return result
