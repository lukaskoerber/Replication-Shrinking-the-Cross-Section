"""
CLI for the paper replication:

    python -m kns.reproduce              # run every section
    python -m kns.reproduce fig2a_ff25   # run a single section (+ its prerequisites)
    python -m kns.reproduce --list       # list available sections

Also installed as the `kns-reproduce` console script.
"""

import argparse

from kns.reproduce import SECTIONS, reproduce_all, run_section


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kns.reproduce",
                                     description="Reproduce the KNS (2020) replication.")
    parser.add_argument("sections", nargs="*",
                        help="section name(s) to run; default = all")
    parser.add_argument("--list", action="store_true", help="list sections and exit")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    if args.list:
        for name, _module, deps in SECTIONS:
            dep_note = f"  (after: {', '.join(deps)})" if deps else ""
            print(f"{name}{dep_note}")
        return 0

    if args.sections:
        for name in args.sections:
            run_section(name)
    else:
        reproduce_all(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
