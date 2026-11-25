import argparse
import json

from outage_replay.frozen import analyze_frozen


def main():
    parser = argparse.ArgumentParser(description="Recompute all four methods on one frozen buffer")
    parser.add_argument("source", help="frozen.csv with adjacent frozen.metadata.json")
    parser.add_argument("output")
    parser.add_argument("--beta", type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(analyze_frozen(args.source, args.output, beta=args.beta), indent=2))


if __name__ == "__main__":
    main()
