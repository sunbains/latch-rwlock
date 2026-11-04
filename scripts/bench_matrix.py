#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run rw_bench over a thread/write-ratio grid for several lock implementations.

  scripts/bench_matrix.py --out results.csv glibc= latch=build/liblatch_interpose.so

Each argument is label=path-to-LD_PRELOAD-library; an empty path uses the system
pthread_rwlock. An optional label=library@benchmark selects a different benchmark
binary, as long as it accepts the same arguments and prints a SYNOPSIS line.
"""
import argparse
import csv
import os
import re
import subprocess

SYNOPSIS = re.compile(r"SYNOPSIS:.*: (\d+) Ops .*Spread=([\d.]+) Utilization=([\d.]+)")
STATS = re.compile(r"JFI=([\d.]+)")


def run_once(library, bench, common, threads, write_pct, limit):
    env = dict(os.environ)
    if library:
        env["LD_PRELOAD"] = os.path.abspath(library)
    command = [bench, *common.split(), f"+{threads}@{write_pct}"]
    try:
        out = subprocess.run(command, env=env, capture_output=True, text=True, timeout=limit).stdout
    except subprocess.TimeoutExpired:
        return None
    match = SYNOPSIS.search(out)
    if not match or "Post validation : OK" not in out:
        return None
    jain = STATS.search(out)
    return int(match[1]), float(match[2]), float(match[3]), float(jain[1]) if jain else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locks", nargs="+", help="label=library[@benchmark]")
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--bench", default="build/rw_bench")
    parser.add_argument("--threads", default="1,8,24,48,96,192,384")
    parser.add_argument("--writes", default="0,1,5,50,100")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--duration", type=int, default=3000, help="milliseconds")
    parser.add_argument("--args", default="ALength=64 NCSLength=50 WCSLength=4 RCSLength=10")
    args = parser.parse_args()

    common = f"{args.args} _Duration={args.duration}"
    limit = args.duration / 1000 + 45
    with open(args.out, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow("lock threads write_pct rep ops_per_sec spread utilization jain".split())
        # Interleaved, so slow drift in the machine's state affects all locks equally.
        for rep in range(args.reps):
            for threads in args.threads.split(","):
                for write_pct in args.writes.split(","):
                    for spec in args.locks:
                        label, target = spec.split("=", 1)
                        library, _, bench = target.partition("@")
                        result = run_once(library, bench or args.bench, common, threads, write_pct, limit)
                        if result is None:
                            writer.writerow([label, threads, write_pct, rep, "", "", "", ""])
                        else:
                            ops, spread, utilization, jain = result
                            writer.writerow([label, threads, write_pct, rep, round(ops * 1000 / args.duration),
                                             spread, utilization, jain])
                        stream.flush()


if __name__ == "__main__":
    main()
