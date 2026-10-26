"""Throughput benchmark for the duel sim.

Reports env-ticks/s and agent-steps/s (2 agents per env) with random actions,
including observation building, which is what PPO actually pays for.

    python tools/bench_sim.py
    python tools/bench_sim.py --envs 256 512 1024 --ticks 400
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mc_pvp.sim import DuelEnv


def bench(envs: int, ticks: int, warmup: int = 60, seed: int = 0) -> dict:
    env = DuelEnv(envs, seed=seed)
    rng = np.random.default_rng(seed)
    act = np.zeros((envs, 2, DuelEnv.ACTION_DIM), np.float32)

    for _ in range(warmup):
        env.step(act)

    t0 = time.perf_counter()
    for _ in range(ticks):
        a = rng.uniform(-1.0, 1.0, (envs, 2, DuelEnv.ACTION_DIM)).astype(np.float32)
        env.step(a)
    dt = time.perf_counter() - t0

    env_ticks = envs * ticks
    return {
        "envs": envs,
        "ticks": ticks,
        "seconds": dt,
        "env_ticks_s": env_ticks / dt,
        "agent_steps_s": 2 * env_ticks / dt,
        "epochs_per_sec": envs / dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--ticks", type=int, default=300)
    args = ap.parse_args()

    print(f"{'envs':>6} {'env-ticks/s':>14} {'agent-steps/s':>15} {'x realtime':>12}")
    print("-" * 52)
    for n in args.envs:
        r = bench(n, args.ticks)
        # one env-tick == 1/20 s of game time
        speedup = r["env_ticks_s"] / 20.0
        print(f"{r['envs']:6d} {r['env_ticks_s']:14,.0f} {r['agent_steps_s']:15,.0f} "
              f"{speedup:11,.0f}x")


if __name__ == "__main__":
    main()
