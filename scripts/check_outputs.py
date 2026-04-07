#!/usr/bin/env python3
"""
check_outputs.py — Automated correctness checker for Assignment 3.

Compares each student output image against the corresponding reference image.
Reports pass/fail per image and an overall summary.

Usage:
    python3 scripts/check_outputs.py \\
        --output   <output_dir>    \\
        --reference <reference_dir> \\
        --tolerance <max_diff>      (default: 2)
"""
import argparse
import os
import sys
import numpy as np

def load_pgm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        if header != 'P5':
            raise ValueError(f"{path}: not a P5 PGM file")
        # skip comments
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != w * h:
        raise ValueError(f"{path}: expected {w*h} bytes, got {data.size}")
    return data.reshape(h, w)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',    required=True, help='Directory of student outputs')
    parser.add_argument('--reference', required=True, help='Directory of reference outputs')
    parser.add_argument('--tolerance', type=int, default=2, help='Max allowed pixel difference')
    args = parser.parse_args()

    # Map: stem → reference path for the three output types
    ref_map = {}
    for fname in os.listdir(args.reference):
        if fname.endswith('.pgm'):
            ref_map[fname] = os.path.join(args.reference, fname)

    # Find student outputs
    student_files = sorted(
        f for f in os.listdir(args.output) if f.endswith('.pgm')
    )

    if not student_files:
        print(f"[check] No .pgm files found in {args.output}")
        sys.exit(1)

    passed = 0
    failed = 0
    missing = 0

    print(f"\n{'Image':<45} {'Stage':<12} {'MaxDiff':>8} {'MeanDiff':>10} {'Result':>8}")
    print("-" * 90)

    for fname in student_files:
        student_path = os.path.join(args.output, fname)

        # Derive which reference file this corresponds to.
        # Student output: img_01_gradient_h_out.pgm
        # Reference files: img_01_gradient_h_blurred.pgm
        #                  img_01_gradient_h_edges.pgm
        #                  img_01_gradient_h_equalized.pgm
        stem = fname.replace('_out.pgm', '')

        # The final student output corresponds to the equalised reference.
        stage = 3
        if '/stage-1' in args.reference:
            stage = 1
            ref_name = stem + '_blurred.pgm'
        elif '/stage-2' in args.reference:
            stage = 2
            ref_name = stem + '_edges.pgm'
        else:
            stage = 3
            ref_name = stem + '_equalized.pgm'
        ref_path = ref_map.get(ref_name)

        if ref_path is None:
            print(f"  {'[MISSING REF]':<45} {ref_name}")
            missing += 1
            continue

        try:
            student = load_pgm(student_path).astype(np.int32)
            ref     = load_pgm(ref_path).astype(np.int32)
        except Exception as e:
            print(f"  {fname:<45} [ERROR] {e}")
            failed += 1
            continue

        if student.shape != ref.shape:
            print(f"  {fname:<45} [SHAPE MISMATCH] student={student.shape} ref={ref.shape}")
            failed += 1
            continue

        diff     = np.abs(student - ref)
        max_diff = int(diff.max())
        mean_diff = float(diff.mean())
        ok       = max_diff <= args.tolerance
        result   = "PASS" if ok else "FAIL"

        print(f"  {fname:<45} {stage:<12} {max_diff:>8} {mean_diff:>10.2f} {result:>8}")

        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 90)
    print(f"\nResults: {passed} passed, {failed} failed, {missing} missing reference")
    print(f"Tolerance: ±{args.tolerance} intensity levels\n")

    if failed > 0 or missing > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
