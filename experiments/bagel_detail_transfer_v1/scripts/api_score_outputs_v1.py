"""Run the frozen scorer with an API-only judge loader and responder."""
import sys
import argparse
from pathlib import Path
from qwen_api_runtime import QwenAPI
import judge_runtime
import remote_judge_runtime


def respond(model, processor, judge, images, prompts, limit=4096):
    return model.respond(images, prompts, limit)


def load(judge):
    return QwenAPI(), None


judge_runtime.respond = respond
remote_judge_runtime.load = load

import score_outputs_remote_v4

score_outputs_remote_v4.respond = respond
score_outputs_remote_v4.load = load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge', choices=('qwen', 'ministral'), required=True)
    parser.add_argument('--task', choices=('understanding', 'depiction', 'appearance', 'content'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--measured', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    score_outputs_remote_v4.run(parser.parse_args())
