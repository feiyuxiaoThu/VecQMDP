#!/usr/bin/env python3
"""
Benchmark Analysis: VecQMDP vs DESPOT on RockSample(15,15)

Parses per-simulation log files and reports:
  - Discounted value
  - Total run time (excl. step 0)
  - Simulation steps (excl. step 0)
  - Avg plan time per step (excl. step 0)
  - Avg simulation throughput (excl. step 0)

Usage:
    python3 examples/rock_sample/benchmark/analyze.py
    python3 examples/rock_sample/benchmark/analyze.py --log-dir examples/rock_sample/benchmark/logs
    python3 examples/rock_sample/benchmark/analyze.py \
        --vecqmdp-dir examples/rock_sample/benchmark/logs/vecqmdp_multithreaded_3000_8_static \
        --despot-dir  examples/rock_sample/benchmark/logs/despot_1000ms_512
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants (must match the run scripts)
# ---------------------------------------------------------------------------
ROCK_BITS = [
    20953, 3649,  820,  24300, 9013,
    8025,  7315,  4573, 24133, 3359,
    22175, 24271, 29235,17871, 2849,
    19350, 13826, 1042, 977,   3071,
]

# ---------------------------------------------------------------------------
# VecQMDP log parser
# ---------------------------------------------------------------------------

def parse_vecqmdp_log(path: Path):
    text = path.read_text(errors='replace')

    # 1. Fix regex: add a third capture group ([0-9.]+) to match the number before #/ms
    step_pat = re.compile(
        r'--- Q-values\s+step=(\d+)\s+best=\S+\s+\[([0-9.]+)\s+ms,\s+([0-9.]+)\s+#/ms\]'
    )

    plan_ms_by_step: dict[int, float] = {}
    throughput_by_step: dict[int, float] = {} # Stores throughput values read from the log

    for m in step_pat.finditer(text):
        step = int(m.group(1))
        t_ms = float(m.group(2))
        tp   = float(m.group(3)) # Extract the #/ms value from the log
        
        plan_ms_by_step[step] = t_ms
        throughput_by_step[step] = tp

    if not plan_ms_by_step:
        return None

    # Final results block (unchanged)
    dr_m    = re.search(r'Discounted reward\s*:\s*([0-9.\-]+)', text)
    steps_m = re.search(r'Steps taken\s*:\s*(\d+)', text)
    wall_m  = re.search(r'Total wall time\s*:\s*([0-9.]+)\s*ms', text)

    if not (dr_m and steps_m and wall_m):
        return None

    steps_total = int(steps_m.group(1))
    
    # 2. Retrieve values directly from the dict; no manual calculation needed
    plan_ms_list = [plan_ms_by_step.get(s, 0.0) for s in range(steps_total)]
    throughput_list = [throughput_by_step.get(s, 0.0) for s in range(steps_total)]

    return {
        'discounted_reward':   float(dr_m.group(1)),
        'total_wall_ms':       float(wall_m.group(1)),
        'steps_total':         steps_total,
        'plan_ms_per_step':    plan_ms_list,
        'throughput_per_step': throughput_list,
    }


# ---------------------------------------------------------------------------
# DESPOT log parser
# ---------------------------------------------------------------------------

def parse_despot_log(path: Path):
    """
    Returns a dict with:
      discounted_reward   : float
      steps_total         : int
      plan_ms_per_step    : list[float]  index 0 = step 0
      throughput_per_step : list[float]  #/ms
    or None on parse failure.
    """
    text = path.read_text(errors='replace')

    # Per-step: "---Round 0 Step N---" followed later by
    # "[PlanningLoop] Time spent in RunStep(): T ms, simulation throughput X#/ms"
    #
    # We collect (step_idx, time_ms, throughput) in document order.
    step_header_pat = re.compile(
        r'---+Round\s+\d+\s+Step\s+(\d+)---+'
    )
    planning_pat = re.compile(
        r'\[PlanningLoop\]\s+Time spent in RunStep\(\):\s*([0-9.]+)ms,\s+'
        r'simulation throughput\s*([0-9.]+)#/ms'
    )

    headers = [(m.start(), int(m.group(1))) for m in step_header_pat.finditer(text)]
    plannings = [(m.start(), float(m.group(1)), float(m.group(2)))
                 for m in planning_pat.finditer(text)]

    if not headers or not plannings:
        return None

    # Match each planning entry to the most-recent header before it
    step_data: dict[int, tuple[float, float]] = {}
    h_idx = 0
    for (p_pos, t_ms, thr) in plannings:
        while h_idx + 1 < len(headers) and headers[h_idx + 1][0] < p_pos:
            h_idx += 1
        step_no = headers[h_idx][1]
        step_data[step_no] = (t_ms, thr)

    # Total steps and discounted reward from summary block
    term_m = re.search(
        r'Simulation terminated in (\d+) steps', text)
    dr_m   = re.search(
        r'Total discounted reward\s*=\s*([0-9.\-]+)', text)

    if not (term_m and dr_m):
        return None

    steps_total = int(term_m.group(1))
    
    # range(steps_total) covers all steps from 0 to steps_total - 1
    plan_ms_full = [step_data.get(s, (0.0, 0.0))[0] for s in range(steps_total)]
    thr_full     = [step_data.get(s, (0.0, 0.0))[1] for s in range(steps_total)]

    # --- Key change: drop the last frame ---
    # Use Python slice [:-1] to remove the last element
    plan_ms_list    = plan_ms_full[:-1]
    throughput_list = thr_full[:-1]
    # Update the total step count to reflect the removal
    new_steps_total = max(0, steps_total - 1)

    return {
        'discounted_reward':   float(dr_m.group(1)),
        'steps_total':         steps_total,
        'plan_ms_per_step':    plan_ms_list,
        'throughput_per_step': throughput_list,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics (exclude step 0)
# ---------------------------------------------------------------------------

def compute_stats(parsed: dict) -> dict:
    """Derive aggregate stats, excluding step 0 from time/throughput metrics."""
    plan_excl  = parsed['plan_ms_per_step'][1:]    # drop step 0
    thr_excl   = parsed['throughput_per_step'][1:]  # drop step 0
    steps_excl = len(plan_excl)

    total_time_excl  = sum(plan_excl)
    avg_plan_excl    = (total_time_excl / steps_excl) if steps_excl > 0 else 0.0
    avg_thr_excl     = (sum(thr_excl)   / len(thr_excl)) if thr_excl else 0.0

    return {
        'discounted_reward': parsed['discounted_reward'],
        'total_time_excl_ms': total_time_excl,
        'steps_excl':         steps_excl,
        'avg_plan_ms_excl':   avg_plan_excl,
        'avg_throughput_excl': avg_thr_excl,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def mean(lst):
    return sum(lst) / len(lst) if lst else float('nan')

def fmt_f(v, digits=2):
    return f'{v:.{digits}f}'


def print_per_sim_table(solver_name: str,
                        results: list[Optional[dict]],
                        rock_bits_list: list[int]):
    """Print a per-simulation table."""
    col_w = [6, 12, 14, 10, 12, 16]
    header = (
        f"{'Sim':>{col_w[0]}} "
        f"{'RockBits':>{col_w[1]}} "
        f"{'DiscReward':>{col_w[2]}} "
        f"{'Steps':>{col_w[3]}} "
        f"{'TotalMs*':>{col_w[4]}} "
        f"{'AvgPlanMs*':>{col_w[5]}} "
        f"{'AvgThr*(#/ms)':>14}"
    )
    sep = '-' * (sum(col_w) + 6 + 14 + 1)

    print(f"\n{'='*len(sep)}")
    print(f"  {solver_name}  (excl. step 0 for time/steps/plan/throughput)")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)

    for i, (r, rb) in enumerate(zip(results, rock_bits_list)):
        if r is None:
            print(f"  {i:>4}  {'N/A':>10}  [log not found or parse error]")
            continue
        s = compute_stats(r)
        print(
            f"{i:>{col_w[0]}} "
            f"{rb:>{col_w[1]}} "
            f"{s['discounted_reward']:>{col_w[2]}.3f} "
            f"{s['steps_excl']:>{col_w[3]}} "
            f"{s['total_time_excl_ms']:>{col_w[4]}.1f} "
            f"{s['avg_plan_ms_excl']:>{col_w[5]}.1f} "
            f"{s['avg_throughput_excl']:>14.1f}"
        )
    print(sep)


def print_summary(solver_name: str, results: list[Optional[dict]]):
    """Print overall mean ± across all valid simulations."""
    valid = [r for r in results if r is not None]
    if not valid:
        print(f"  [{solver_name}] No valid logs found.\n")
        return

    stats_list = [compute_stats(r) for r in valid]

    dr_vals   = [s['discounted_reward']    for s in stats_list]
    tm_vals   = [s['total_time_excl_ms']   for s in stats_list]
    st_vals   = [s['steps_excl']           for s in stats_list]
    pl_vals   = [s['avg_plan_ms_excl']     for s in stats_list]
    thr_vals  = [s['avg_throughput_excl']  for s in stats_list]

    print(f"\n  [{solver_name}] Summary over {len(valid)} simulation(s):")
    print(f"    Avg discounted reward   : {mean(dr_vals):.3f}")
    print(f"    Avg total time* (ms)    : {mean(tm_vals):.1f}")
    print(f"    Avg steps*              : {mean(st_vals):.1f}")
    print(f"    Avg plan time/step* (ms): {mean(pl_vals):.1f}")
    print(f"    Avg throughput* (#/ms)  : {mean(thr_vals):.1f}")
    print(f"  (* = step 0 excluded)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze VecQMDP vs DESPOT benchmark logs')
    parser.add_argument('--log-dir', default=None,
                        help='Base log directory (default: <script_dir>/logs). '
                             'Ignored when --vecqmdp-dir and --despot-dir are both set.')
    parser.add_argument('--vecqmdp-dir', default=None,
                        help='Directory containing VecQMDP per-simulation log files '
                             '(sim_00.log ... sim_NN.log). '
                             'Overrides the subdirectory derived from --log-dir.')
    parser.add_argument('--despot-dir', default=None,
                        help='Directory containing DESPOT per-simulation log files '
                             '(sim_00.log ... sim_NN.log). '
                             'Overrides the subdirectory derived from --log-dir.')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    log_base   = Path(args.log_dir) if args.log_dir else (script_dir / 'logs')

    vecqmdp_dir = Path(args.vecqmdp_dir) if args.vecqmdp_dir else (log_base / 'vecqmdp_multithreaded_3000_8_static')
    despot_dir  = Path(args.despot_dir)  if args.despot_dir  else (log_base / 'despot_1000ms_512')

    num_sims = len(ROCK_BITS)

    # ---- Parse VecQMDP logs ----
    vecqmdp_results: list[Optional[dict]] = []
    for i in range(num_sims):
        log_path = vecqmdp_dir / f'sim_{i:02d}.log'
        if not log_path.exists():
            print(f"[WARN] VecQMDP log missing: {log_path}", file=sys.stderr)
            vecqmdp_results.append(None)
        else:
            r = parse_vecqmdp_log(log_path)
            if r is None:
                print(f"[WARN] VecQMDP parse error: {log_path}", file=sys.stderr)
            vecqmdp_results.append(r)

    # ---- Parse DESPOT logs ----
    despot_results: list[Optional[dict]] = []
    for i in range(num_sims):
        log_path = despot_dir / f'sim_{i:02d}.log'
        if not log_path.exists():
            print(f"[WARN] DESPOT log missing: {log_path}", file=sys.stderr)
            despot_results.append(None)
        else:
            r = parse_despot_log(log_path)
            if r is None:
                print(f"[WARN] DESPOT parse error: {log_path}", file=sys.stderr)
            despot_results.append(r)

    # ---- Per-simulation comparison table ----
    print_per_sim_table('VecQMDP', vecqmdp_results, ROCK_BITS)
    print_per_sim_table('DESPOT',  despot_results,  ROCK_BITS)

    # ---- Summaries ----
    print('\n' + '='*60)
    print('  AGGREGATE SUMMARY')
    print('='*60)
    print_summary('VecQMDP', vecqmdp_results)
    print_summary('DESPOT',  despot_results)

    # ---- Side-by-side comparison ----
    v_valid = [(i, compute_stats(r)) for i, r in enumerate(vecqmdp_results) if r is not None]
    d_valid = [(i, compute_stats(r)) for i, r in enumerate(despot_results)  if r is not None]
    common  = set(i for i, _ in v_valid) & set(i for i, _ in d_valid)

    if common:
        print('\n' + '='*70)
        print('  HEAD-TO-HEAD (simulations where both solvers have valid logs)')
        print('='*70)
        hdr = (
            f"{'Sim':>4}  {'RockBits':>10}  "
            f"{'VQ_Reward':>10}  {'DS_Reward':>10}  "
            f"{'VQ_PlanMs*':>11}  {'DS_PlanMs*':>11}  "
            f"{'VQ_Thr*':>9}  {'DS_Thr*':>9}"
        )
        print(hdr)
        print('-' * len(hdr))
        vd = {i: s for i, s in v_valid}
        dd = {i: s for i, s in d_valid}
        for i in sorted(common):
            v = vd[i]
            d = dd[i]
            print(
                f"{i:>4}  {ROCK_BITS[i]:>10}  "
                f"{v['discounted_reward']:>10.3f}  {d['discounted_reward']:>10.3f}  "
                f"{v['avg_plan_ms_excl']:>11.1f}  {d['avg_plan_ms_excl']:>11.1f}  "
                f"{v['avg_throughput_excl']:>9.1f}  {d['avg_throughput_excl']:>9.1f}"
            )
        print('-' * len(hdr))

        # Means over common sims
        v_dr  = mean([vd[i]['discounted_reward']    for i in common])
        d_dr  = mean([dd[i]['discounted_reward']    for i in common])
        v_pl  = mean([vd[i]['avg_plan_ms_excl']     for i in common])
        d_pl  = mean([dd[i]['avg_plan_ms_excl']     for i in common])
        v_thr = mean([vd[i]['avg_throughput_excl']  for i in common])
        d_thr = mean([dd[i]['avg_throughput_excl']  for i in common])

        print(f"{'MEAN':>4}  {'':>10}  "
              f"{v_dr:>10.3f}  {d_dr:>10.3f}  "
              f"{v_pl:>11.1f}  {d_pl:>11.1f}  "
              f"{v_thr:>9.1f}  {d_thr:>9.1f}")
        print()
        print(f"  VecQMDP / DESPOT reward ratio : {v_dr/d_dr:.3f}" if d_dr != 0 else '')
        print(f"  VecQMDP / DESPOT plan-time ratio: {v_pl/d_pl:.3f}" if d_pl != 0 else '')
        print(f"  VecQMDP / DESPOT throughput ratio: {v_thr/d_thr:.3f}" if d_thr != 0 else '')

    print()


if __name__ == '__main__':
    main()
