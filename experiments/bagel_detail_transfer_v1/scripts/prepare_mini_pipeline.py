"""Fix excluded technical train/dev examples for an end-to-end pipeline check."""
import random
from common import STUDY,now,rows,sha,write,write_rows
from prepare_u_evaluation import build


def main():
    path=STUDY/'data/technical_fact_bank_v1/records.jsonl';records=rows(path)
    scenes=[r for r in records if r['id'].startswith('coco_')]
    portraits=[r for r in records if not r['id'].startswith('coco_')]
    random.Random(7100).shuffle(portraits)
    train=[dict(r,split='train',weight=1.) for r in portraits[:16-len(scenes)]+scenes]
    dev=[dict(r,split='dev',weight=1.) for r in portraits[16-len(scenes):24-len(scenes)]]
    if {r['id'] for r in train}&{r['id'] for r in dev}:
        raise ValueError('Technical train/dev overlap')
    directory=STUDY/'data/mini_pipeline_v1';directory.mkdir(exist_ok=False)
    write_rows(directory/'train.jsonl',train);write_rows(directory/'dev.jsonl',dev)
    cases,pairs=build(dev);write_rows(directory/'u_cases.jsonl',cases);write_rows(directory/'u_image_pairs.jsonl',pairs)
    generation=[r for r in rows(STUDY/'configs/generation_plan_v1/quality_check.jsonl') if r['id']=='quality_explicit_0']
    write_rows(directory/'g_cases.jsonl',generation)
    write(directory/'contract.json',dict(created_utc=now(),source_sha256=sha(__file__),records_sha256=sha(path),
        train_images=len(train),dev_images=len(dev),u_cases=len(cases),g_images_per_state=len(generation),
        seed=7100,epochs=3,batch_images=8,
        scope='All source images excluded from the formal pool. Four actual small training runs, native U inference and one native G image each qualify the pipeline; these sample sizes cannot establish fairness or transfer.'))


if __name__=='__main__':
    main()
