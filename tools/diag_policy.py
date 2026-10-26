"""Diagnose a trained checkpoint: how does it actually fight?

    python tools/diag_policy.py [runs/ppo.pt]

Reports the things that matter for PvP, not just button rates:
  * how often it is inside attack range
  * how often its crosshair is actually on the target
  * damage traded per episode
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mc_pvp.bots import ScriptedBot
from mc_pvp.ppo import PPO
from mc_pvp.sim import REACH, DuelEnv
from train import aim_delta, pack_action, to_t


def main(path="runs/ppo.pt", n=16, ticks=1200):
    env = DuelEnv(n, seed=7, max_ticks=20 * 45)
    obs = env.reset()
    net = PPO.load(path, obs.shape[-1])
    bot = ScriptedBot(n, np.random.default_rng(7))

    atk, jump, sprint, blk, eat, spl = [], [], [], [], [], []
    in_reach, aimed, dists, aim_err = [], [], [], []
    dmg_for = np.zeros(n)
    dmg_against = np.zeros(n)
    seen = np.zeros(n, bool)

    for _ in range(ticks):
        with torch.no_grad():
            cont, disc, _, _ = net.act(to_t(obs[:, 0]), deterministic=True)
        cont, disc = cont.numpy(), disc.numpy()

        dist = np.hypot(env.px[:, 0] - env.px[:, 1], env.pz[:, 0] - env.pz[:, 1])
        err_yaw, _ = aim_delta(env, np.zeros(n, np.int64))
        in_reach.append((dist <= REACH).mean())
        aimed.append((np.abs(err_yaw) < 15.0).mean())
        dists.append(dist.mean())
        aim_err.append(np.abs(err_yaw).mean())

        act = bot.act(env)
        act[:, 0] = pack_action(cont, disc)
        atk.append(disc[:, 2].mean())
        jump.append(disc[:, 0].mean())
        sprint.append(disc[:, 1].mean())
        if disc.shape[1] > 3:
            blk.append(disc[:, 3].mean())
            eat.append(disc[:, 5].mean())
            spl.append(disc[:, 6:].max(axis=1).mean())

        obs, rew, done, info = env.step(act, record_events=False)
        for f in info.get("finished", []):
            i = f["env"]
            if seen[i]:
                continue
            seen[i] = True
            dmg_for[i] = f["dmg"][0]
            dmg_against[i] = f["dmg"][1]

    m = seen if seen.any() else np.ones(n, bool)
    print(f"attack {np.mean(atk):.2f}  jump {np.mean(jump):.2f}  sprint {np.mean(sprint):.2f}")
    if blk:
        print(f"block {np.mean(blk):.2f}  eat {np.mean(eat):.2f}  splash {np.mean(spl):.2f}")
    print(f"in reach {np.mean(in_reach)*100:5.1f}% of ticks"
          f" | aimed (<15 deg) {np.mean(aimed)*100:5.1f}%"
          f" | mean aim error {np.mean(aim_err):5.1f} deg")
    print(f"mean distance {np.mean(dists):.2f} blocks")
    print(f"dmg dealt {dmg_for[m].mean():5.1f} | taken {dmg_against[m].mean():5.1f}"
          f" | duels {m.sum()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/ppo.pt")
