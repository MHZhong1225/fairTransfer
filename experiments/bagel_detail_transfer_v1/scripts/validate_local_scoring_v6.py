"""Audit already completed actual local controls without rerunning GPU work."""
from common import STUDY,now,read,sha,write


def main():
    evidence={}
    for judge in ('qwen','ministral'):
        root=STUDY/'results/scoring_qualification_v6'/judge
        status=read(root/'status.json');contract=read(root/'contract.json')
        if status['status']!='passed' or len(status['checks'])!=11 or not all(status['checks'].values()):
            raise ValueError('The complete actual controls have not passed')
        if any(sha(STUDY/'scripts'/name)!=digest for name,digest in contract['source_hashes'].items()):
            raise ValueError('Qualification implementation changed after measurement')
        evidence[judge]=dict(status_sha256=sha(root/'status.json'),contract_sha256=sha(root/'contract.json'),checks=status['checks'])
    write(STUDY/'results/scoring_controls_audit_v6.json',dict(status='completed',checked_utc=now(),evidence=evidence,
        scope='Checks real pre-existing GPU control outputs and matching implementation. Does not replace or relabel any failed run.'))


if __name__=='__main__':
    main()
