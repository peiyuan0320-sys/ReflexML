"""Explicit entrypoint for the frozen MNIST screening; review before execution."""
import argparse

from reflexml.collapse_external import run_replication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="New output directory; never overwritten")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--run-screening", action="store_true",
                        help="Explicitly launch N=6 (requires separately reviewed execution authorization)")
    parser.add_argument("--expand-from", help="Original N=6 AMBIGUOUS output; one N=12 expansion")
    args = parser.parse_args()
    if args.run_screening == bool(args.expand_from):
        parser.error("Choose exactly one of --run-screening or --expand-from")
    result = run_replication(args.data_root, args.output, download=args.download,
                             initial_output=args.expand_from)
    print(result["classification"])


if __name__ == "__main__":
    main()
