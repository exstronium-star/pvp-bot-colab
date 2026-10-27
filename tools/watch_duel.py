"""Watch a sim duel as a top-down SVG replay.

    python tools/watch_duel.py [checkpoint] [out.svg] [--ticks N] [--seed S]

Records policy (seat 0, orange) vs scripted (seat 1, cyan): trails, hits
sized by damage, shield-disable stars, gapple/splash markers, and a
health-over-time strip. Open the SVG in a browser.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from mc_pvp.bots import ScriptedBot
from mc_pvp.ppo import PPO
from mc_pvp.sim import DuelEnv
from train import aim_delta, pack_action, to_t


def run_duel(ckpt: str, max_ticks: int, seed: int):
    env = DuelEnv(1, seed=seed)
    obs = env.reset()
    net = PPO.load(ckpt, obs.shape[-1])
    bot = ScriptedBot(1, np.random.default_rng(seed + 1))

    T = {"px": [], "pz": [], "hp": [], "blk": [], "eat": [], "gap": [], "heal": [],
         "hits": [], "dis": [], "spl": [], "outcome": ""}
    prev_off = env.shield_off.copy()
    prev_heal = env.n_heal.copy()
    prev_gap = env.n_gap.copy()
    prev_eat = env.eating.copy()
    winner = -1
    for t in range(max_ticks):
        with torch.no_grad():
            cont, disc, _, _ = net.act(to_t(obs[:, 0]), deterministic=True)
        act = bot.act(env)
        act[:, 0] = pack_action(cont.numpy(), disc.numpy())
        obs, rew, done, info = env.step(act, record_events=True)
        for f in info.get("finished", []):
            pass
        T["px"].append(env.px[0].copy())
        T["pz"].append(env.pz[0].copy())
        T["hp"].append(env.health[0].copy())
        T["blk"].append(env.blocking[0].copy())
        T["eat"].append(env.eating[0].copy())
        T["gap"].append(env.n_gap[0].copy())
        T["heal"].append(env.n_heal[0].copy())
        for e in info.get("events", []):
            T["hits"].append((t, e["attacker"], e["damage"], e["x"], e["z"]))
        # shield disable started
        if env.shield_off[0, 0] == 100 and prev_off[0, 0] < 100:
            T["dis"].append((t, 0, float(env.px[0, 0]), float(env.pz[0, 0])))
        if env.shield_off[0, 1] == 100 and prev_off[0, 1] < 100:
            T["dis"].append((t, 1, float(env.px[0, 1]), float(env.pz[0, 1])))
        # splash thrown
        if env.n_heal[0, 0] < prev_heal[0, 0]:
            T["spl"].append((t, 0, float(env.px[0, 0]), float(env.pz[0, 0])))
        if env.n_heal[0, 1] < prev_heal[0, 1]:
            T["spl"].append((t, 1, float(env.px[0, 1]), float(env.pz[0, 1])))
        prev_off = env.shield_off.copy()
        prev_heal = env.n_heal.copy()
        if done[0]:
            h = env.health[0]
            winner = 0 if (h[0] > 0 and h[1] <= 0) else (1 if h[1] > 0 and h[0] <= 0 else -1)
            break
    T["outcome"] = ("policy wins" if winner == 0 else
                    "scripted wins" if winner == 1 else "draw/timeout")
    for k in ("px", "pz", "hp", "blk", "eat", "gap", "heal"):
        T[k] = np.array(T[k])
    return T, env


def svg(T, env, path: str):
    A = env.arena
    W, H = 760, 980
    top, map_s = 90, 700  # map square
    strip_y, strip_h = 820, 120

    def X(x):
        return 30 + (x + A) / (2 * A) * map_s

    def Z(z):
        return top + (z + A) / (2 * A) * map_s

    n = len(T["px"])
    step = max(1, n // 400)
    cols = ["#ff9e2c", "#35d0ff"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="#101418"/>',
         f'<text x="30" y="34" fill="#fff" font-size="22" font-family="monospace">'
         f'policy (orange) vs scripted (cyan) - {T["outcome"]} - {n} ticks</text>',
         f'<text x="30" y="58" fill="#8a93a0" font-size="13" font-family="monospace">'
         f'circle = hit (size ~ dmg) | star = shield disable | + = splash heal | '
         f'ring = eating</text>',
         f'<rect x="30" y="{top}" width="{map_s}" height="{map_s}" '
         f'fill="#181d24" stroke="#3a4350"/>']
    # spawn markers
    for p in (0, 1):
        s.append(f'<circle cx="{X(T["px"][0, p]):.1f}" cy="{Z(T["pz"][0, p]):.1f}" '
                 f'r="5" fill="none" stroke="{cols[p]}" stroke-dasharray="3,2"/>')
    # trails
    for p in (0, 1):
        pts = " ".join(f"{X(x):.1f},{Z(z):.1f}"
                       for x, z in zip(T["px"][::step, p], T["pz"][::step, p]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{cols[p]}" '
                 f'stroke-width="2" opacity="0.75"/>')
        ex, ez = T["px"][-1, p], T["pz"][-1, p]
        s.append(f'<circle cx="{X(ex):.1f}" cy="{Z(ez):.1f}" r="6" fill="{cols[p]}"/>')
    # hits
    for t, atk, dmg, x, z in T["hits"]:
        r = 2 + dmg * 3
        s.append(f'<circle cx="{X(x):.1f}" cy="{Z(z):.1f}" r="{r:.1f}" '
                 f'fill="none" stroke="{cols[atk]}" stroke-width="2"/>')
    # disables
    for t, p, x, z in T["dis"]:
        s.append(f'<text x="{X(x):.1f}" y="{Z(z):.1f}" fill="#ffe14d" font-size="16" '
                 f'text-anchor="middle">★</text>')
    # splashes
    for t, p, x, z in T["spl"]:
        s.append(f'<text x="{X(x):.1f}" y="{Z(z):.1f}" fill="#7dff8a" font-size="16" '
                 f'text-anchor="middle">+</text>')
    # eating rings (subsampled)
    for i in range(0, n, 20):
        for p in (0, 1):
            if T["eat"][i, p] > 0:
                x, z = T["px"][i, p], T["pz"][i, p]
                s.append(f'<circle cx="{X(x):.1f}" cy="{Z(z):.1f}" r="9" fill="none" '
                         f'stroke="{cols[p]}" stroke-dasharray="2,2"/>')

    # health strip
    s.append(f'<rect x="30" y="{strip_y}" width="{map_s}" height="{strip_h}" fill="#181d24"/>')
    for p in (0, 1):
        pts = " ".join(
            f"{30 + i / max(n - 1, 1) * map_s:.1f},{strip_y + strip_h - h / 20 * strip_h:.1f}"
            for i, h in enumerate(T["hp"][::max(1, n // 300), p]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{cols[p]}" stroke-width="2"/>')
    s.append(f'<text x="30" y="{strip_y + strip_h + 22}" fill="#8a93a0" font-size="13" '
             f'font-family="monospace">policy hp {T["hp"][-1, 0]:.1f} | '
             f'scripted hp {T["hp"][-1, 1]:.1f} | gapples left '
             f'{int(T["gap"][-1, 0])}/{int(T["gap"][-1, 1])} | heal pots left '
             f'{int(T["heal"][-1, 0])}/{int(T["heal"][-1, 1])}</text>')
    # damage totals
    d0 = sum(d for _, a, d, _, _ in T["hits"] if a == 0)
    d1 = sum(d for _, a, d, _, _ in T["hits"] if a == 1)
    s.append(f'<text x="30" y="{strip_y + strip_h + 44}" fill="#8a93a0" font-size="13" '
             f'font-family="monospace">dmg dealt {d0:.1f} / taken {d1:.1f} | '
             f'hits {sum(1 for _, a, _, _, _ in T["hits"] if a == 0)} / '
             f'{sum(1 for _, a, _, _, _ in T["hits"] if a == 1)} | disables {len(T["dis"])}</text>')
    s.append('</svg>')
    pathlib.Path(path).write_text("\n".join(s))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/kit1/ppo.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "watch.svg"
    ticks = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    T, env = run_duel(ckpt, ticks, seed)
    svg(T, env, out)
    print(f"wrote {out}: {T['outcome']}, {len(T['px'])} ticks, "
          f"{len(T['hits'])} hits, {len(T['dis'])} disables")


if __name__ == "__main__":
    main()
