"""Explicit Wait-d phases. Invoking --help never loads data or trains."""
import argparse
import json
from pathlib import Path

from reflexml.wait_d import inspect_run, preflight, run, write_json


def main():
    parser = argparse.ArgumentParser(description='ReflexML Wait-d v0.1')
    parser.add_argument('mode', choices=('preflight', 'run', 'validate', 'analyze'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor
    dataset = FashionMNIST(repo / 'data', train=True, download=False, transform=ToTensor())
    if args.mode == 'preflight':
        _, manifest = preflight(repo, dataset)
        write_json(args.output, manifest)
        print('PREFLIGHT PASS; no training performed')
    elif args.mode == 'run':
        print(run(repo, dataset, args.output))
    else:
        print(json.dumps(inspect_run(repo, dataset, args.output,
                                     analyze=args.mode == 'analyze'), sort_keys=True))


if __name__ == '__main__':
    main()
