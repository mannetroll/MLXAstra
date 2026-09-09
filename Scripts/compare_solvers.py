#!/usr/bin/env python3
"""Compare release solver throughput at 512² state / 768² nonlinear resolution.

Build both apps first, then run this script from a terminal with Metal access.
No third-party Python packages are needed. This is a whole-solver comparison;
the integrators, initial spectra, timestep rules, and snapshot costs differ.
"""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTES = [
    "Both runs use 512² state and 768² nonlinear grids, unforced flow, "
    "viscosity approximately 0.00015, and eight steps per batch.",
    "Integrators, timestep rules, seeds, and initial spectra differ; this is "
    "not a matched-trajectory or isolated FFT benchmark.",
    "Astra excludes the requested warmup steps. Metal's existing headless "
    "API has no stepping warmup; initialization and field statistics are untimed.",
    "Astra includes a field snapshot and GPU-buffer handoff per batch. "
    "Metal's headless timing includes no display snapshots.",
    "Runs are sequential with alternating solver order. Close other GPU "
    "workloads before running; this script does not stop other applications.",
]


def require_metal():
    if sys.platform != "darwin":
        raise RuntimeError("This comparison requires macOS and a Metal GPU.")
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.argtypes = []
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    if not metal.MTLCreateSystemDefaultDevice():
        raise RuntimeError(
            "Metal device unavailable in this process. Run the same command "
            "from a macOS terminal with Metal access. No benchmarks were run "
            "and no historical results were substituted."
        )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def positive_number(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def read_result(solver, prefix, args):
    if solver == "astra":
        report = json.loads(prefix.with_suffix(".stdout.txt").read_text())
        expected = {"grid": 512, "padded_grid": 768,
                    "dealiasing": "three-halves-padding", "steps": args.steps,
                    "warmup": args.warmup, "batch_steps": 8,
                    "preset": "decaying", "finite": True}
        for key, value in expected.items():
            if report.get(key) != value:
                raise ValueError(
                    f"Astra {key} must be {value!r}, got {report.get(key)!r}. "
                    "Rebuild the current padded solver before comparing."
                )
        rate = report["updates_per_second"]
        milliseconds = report["mean_ms"]
    else:
        with prefix.with_suffix(".csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 1:
            raise ValueError("Expected exactly one Metal benchmark CSV row")
        report = rows[0]
        if (int(report["N"]), int(report["padded"]), int(report["steps"])) != (512, 768, args.steps):
            raise ValueError("Metal benchmark returned the wrong grids or step count")
        for key in ("E0", "E_final", "Z_final"):
            if not math.isfinite(float(report[key])):
                raise ValueError(f"Non-finite Metal diagnostic: {key}")
        rate = report["steps_per_s"]
        milliseconds = report["ms_per_step"]
    return {"solver": solver,
            "steps_per_second": positive_number(rate, "steps/s"),
            "ms_per_step": positive_number(milliseconds, "ms/step"),
            "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    derived = Path(os.environ.get("ASTRA_DERIVED_DATA", "/tmp/MLXAstraDerived"))
    parser.add_argument("--astra-binary", type=Path, default=derived /
                        "Build/Products/Release/MLXAstra.app/Contents/MacOS/MLXAstra")
    parser.add_argument("--metal-binary", type=Path, default=ROOT.parent /
                        "Metal2DTurbo/.build-release/Build/Products/Release/"
                        "Metal2DTurbo.app/Contents/MacOS/Metal2DTurbo")
    parser.add_argument("--output-dir", type=Path, help="New directory for raw output and summaries")
    parser.add_argument("--runs", type=int, default=3, help="Runs per solver (default: 3)")
    parser.add_argument("--steps", type=int, default=3000, help="Timed steps per run (default: 3000)")
    parser.add_argument("--warmup", type=int, default=30, help="Astra-only warmup steps (default: 30)")
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or not 1 <= args.steps <= 100_000 or not 0 <= args.warmup <= 1000:
        parser.error("Require 1–100 runs, 1–100000 steps, and 0–1000 warmup steps")
    require_metal()
    binaries = {"metal": args.metal_binary.expanduser().resolve(),
                "astra": args.astra_binary.expanduser().resolve()}
    for name, binary in binaries.items():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Missing executable for {name}: {binary}. Build the release app first.")
    now = datetime.now(timezone.utc)
    output = (args.output_dir or ROOT / ".build" /
              f"solver-comparison-{now:%Y%m%dT%H%M%S%fZ}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"created_utc": now.isoformat(), "host": platform.platform(),
                "state_grid": 512, "padded_grid": 768, "batch_steps": 8,
                "target_viscosity": 0.00015, "runs_per_solver": args.runs,
                "timed_steps": args.steps, "astra_warmup": args.warmup,
                "metal_warmup": 0, "notes": NOTES,
                "binaries": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for name, path in binaries.items()}}
    write_json(output / "metadata.json", metadata)
    print(f"Results: {output}", flush=True)
    print("Whole-solver throughput; different integrators, initialization, and timing overhead.", flush=True)
    records = []
    for index in range(args.runs):
        order = ("metal", "astra") if index % 2 == 0 else ("astra", "metal")
        for solver in order:
            prefix = output / f"{index + 1:02d}-{solver}"
            command = [str(binaries[solver]), "--benchmark", "--steps", str(args.steps)]
            if solver == "metal":
                command += ["--resolution", "512", "--reynolds", "6666.666666667",
                            "--k0", "4", "--cfl", "1.5", "--label", "padded-comparison",
                            "--csv", str(prefix.with_suffix(".csv"))]
            else:
                command += ["--grid", "512", "--preset", "decaying", "--batch", "8",
                            "--warmup", str(args.warmup)]
            write_json(prefix.with_suffix(".command.json"), command)
            print(f"Run {index + 1}/{args.runs}: {solver}", flush=True)
            with prefix.with_suffix(".stdout.txt").open("w") as stdout, \
                    prefix.with_suffix(".stderr.txt").open("w") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600)
            if result.returncode:
                raise RuntimeError(f"{solver} exited {result.returncode}; inspect {prefix}.stderr.txt")
            record = read_result(solver, prefix, args)
            record["run"] = index + 1
            records.append(record)
            write_json(output / "runs.json", records)
            print(f"  {record['steps_per_second']:.1f} steps/s", flush=True)
    medians = {solver: {key: statistics.median(row[key] for row in records if row["solver"] == solver)
                        for key in ("steps_per_second", "ms_per_step")}
               for solver in binaries}
    summary = {"medians": medians, "metal_over_astra_steps_per_second":
               medians["metal"]["steps_per_second"] / medians["astra"]["steps_per_second"],
               "notes": NOTES}
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["solver", "median_steps_per_second", "median_ms_per_step"])
        for solver, values in medians.items():
            writer.writerow([solver, values["steps_per_second"], values["ms_per_step"]])
    for solver, values in medians.items():
        print(f"{solver}: {values['steps_per_second']:.1f} steps/s median ({values['ms_per_step']:.3f} ms/step)")
    print(f"Metal/Astra throughput ratio: {summary['metal_over_astra_steps_per_second']:.3f}×")
    for note in NOTES:
        print(note)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        sys.exit(1)
