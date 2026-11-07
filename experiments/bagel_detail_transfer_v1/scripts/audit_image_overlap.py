"""Conservative exact and perceptual overlap screen before data freezing."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.fft import dctn
from common import STUDY,now,rows,sha,write


def main():
    started = time.perf_counter()
    plan = rows(STUDY/'data/candidate_manifest.jsonl') + rows(STUDY/'data/technical_manifest.jsonl')
    features, hashes = [], []
    for row in plan:
        with Image.open(row['image_path']) as image:
            rgb = np.asarray(image.convert('RGB').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)/255
        gray = rgb @ np.array([.299,.587,.114],dtype=np.float32)
        frequency = dctn(gray,norm='ortho')[:8,:8].reshape(-1)
        bitmask = frequency > np.median(frequency[1:])
        hashes.append(int.from_bytes(np.packbits(bitmask).tobytes(),'big'))
        features.append(rgb)
    features = np.stack(features)
    exact = defaultdict(list)
    for row in plan:
        exact[row['image_sha256']].append(row)
    exact_groups = [[dict(id=r['id'],split=r['split'],source_family=r['source_family']) for r in group]
                    for group in exact.values() if len(group)>1]
    near = []
    for first in range(len(plan)):
        possible = [second for second in range(first+1,len(plan)) if (hashes[first]^hashes[second]).bit_count()<=4]
        if not possible:
            continue
        errors = ((features[possible]-features[first])**2).mean(axis=(1,2,3))
        for second,error in zip(possible,errors):
            if error<=.001:
                a,b = plan[first],plan[second]
                near.append(dict(first_id=a['id'],second_id=b['id'],first_split=a['split'],second_split=b['split'],
                    first_family=a['source_family'],second_family=b['source_family'],rgb32_mse=float(error),
                    phash_distance=(hashes[first]^hashes[second]).bit_count(),same_sha256=a['image_sha256']==b['image_sha256']))
    write(STUDY/'data/initial_overlap_audit.json',dict(created_utc=now(),source_sha256=sha(__file__),images=len(plan),
        input_manifests={str(STUDY/'data'/n):sha(STUDY/'data'/n) for n in ('candidate_manifest.jsonl','technical_manifest.jsonl')},
        exact_duplicate_groups=exact_groups,near_candidates=near,
        crossing_splits=[r for r in near if r['first_split']!=r['second_split']],seconds=time.perf_counter()-started,
        method='Exact file SHA256 plus 64-bit DCT pHash Hamming <=4 AND 32x32 RGB mean squared difference <=0.001 (pixels scaled 0-1). Candidate screen, not proof of no identity/semantic overlap. Original pixels are untouched.',
        action='Any cross-split candidate must be resolved by source-connected grouping before the formal data freeze; do not inspect model outcomes to choose the destination.'))


if __name__=='__main__':
    main()
