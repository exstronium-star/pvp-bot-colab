"""Self-play PPO trainer for the sword-duel sim (v2).

Opponents (assigned PER ENV every iteration, so every batch has the same mix):
    bot     scripted sparring bot (stable anchor: "walk at them and hit them")
    mirror  the current policy (both seats feed the learner -> 2x data)
    league  frozen snapshots from earlier in the run (stops cycling)
    champ   the best-by-eval checkpoint so far (stable strong opponent)

Why v2: the old version picked ONE opponent type per iteration and logged the
mean of the last 40 episodes (~1 iteration). The "avg reward" chart therefore
just jumped between "vs bot" / "vs mirror" / "vs league" iterations, and PPO
was also being updated on a batch that was entirely one opponent type.

    python -u train.py --iters 500 --envs 32 --ticks 256 --outdir runs/kit1
    python -u train.py --iters 500 --envs 32 --ticks 256 --outdir runs/kit1 --resume

When EXTENDING a run (resume with a bigger --iters) also pass
--anneal-iters <original --iters>, otherwise lr / entropy jump back up.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time

import numpy as np
import torch

from mc_pvp.bots import ScriptedBot, AggroBot, TurtleBot, RunnerBot
from mc_pvp.ppo import PPO, ActorCritic
from mc_pvp.sim import COOLDOWN_TICKS, EYE_HEIGHT, DuelEnv  # noqa: F401

RUNS = pathlib.Path("runs")
MAX_LOOK = 20.0

BOT, MIRROR, LEAGUE, CHAMP = 0, 1, 2, 3
KIND_NAMES = ("bot", "mirror", "league", "champ")
DRILL_NAMES = ("melee", "heal", "web", "repot")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def pack_action(cont: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """(n,4),(n,10) -> (n,14) in the env's action layout."""
    n = cont.shape[0]
    act = np.zeros((n, DuelEnv.ACTION_DIM), np.float32)
    act[:, DuelEnv.A_YAW] = cont[:, 0] * MAX_LOOK
    act[:, DuelEnv.A_PITCH] = cont[:, 1] * MAX_LOOK
    act[:, DuelEnv.A_FWD] = cont[:, 2]
    act[:, DuelEnv.A_STRAFE] = cont[:, 3]
    act[:, DuelEnv.A_JUMP] = disc[:, 0]
    act[:, DuelEnv.A_SPRINT] = disc[:, 1]
    act[:, DuelEnv.A_ATTACK] = disc[:, 2]
    act[:, DuelEnv.A_BLOCK] = disc[:, 3]
    act[:, DuelEnv.A_SWAP] = disc[:, 4]
    act[:, DuelEnv.A_EAT] = disc[:, 5]
    act[:, DuelEnv.A_SPL_HEAL] = disc[:, 6]
    act[:, DuelEnv.A_SPL_STR] = disc[:, 7]
    act[:, DuelEnv.A_SPL_SPD] = disc[:, 8]
    act[:, DuelEnv.A_SPL_FIR] = disc[:, 9]
    if disc.shape[1] > 10:
        act[:, DuelEnv.A_WEAVE] = disc[:, 10]
    else:
        # old checkpoints (10-disc): weave head missing -> never weave.
        # Backwards compatible: old weights still load, new head trains fresh.
        act[:, DuelEnv.A_WEAVE] = 0.0
    return act


def aim_delta(env: DuelEnv, env_idx: np.ndarray,
              seats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw/pitch delta (deg) that points seat `seats[k]` of env `env_idx[k]`
    at its opponent's chest."""
    idx = env_idx
    o = 1 - seats
    dx = env.px[idx, o] - env.px[idx, seats]
    dz = env.pz[idx, o] - env.pz[idx, seats]
    dy = (env.py[idx, o] + 1.2) - (env.py[idx, seats] + EYE_HEIGHT)
    flat = np.maximum(np.hypot(dx, dz), 1e-6)
    want_yaw = np.degrees(np.arctan2(-dx, dz))
    err_yaw = ((want_yaw - env.yaw[idx, seats] + 180.0) % 360.0) - 180.0
    want_pitch = np.degrees(-np.arctan2(dy, flat))
    err_pitch = want_pitch - env.pitch[idx, seats]
    return err_yaw.astype(np.float32), err_pitch.astype(np.float32)


def net_device(net: ActorCritic) -> torch.device:
    return next(net.parameters()).device


def to_t(x: np.ndarray, dev=None) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)


@torch.no_grad()
def act_batch(net: ActorCritic, obs: np.ndarray, deterministic=False):
    dev = net_device(net)
    cont, disc, logp, value = net.act(to_t(obs, dev), deterministic=deterministic)
    return (cont.detach().cpu().numpy(), disc.detach().cpu().numpy(),
            logp.detach().cpu().numpy(), value.detach().cpu().numpy())


def build_net(state: dict, obs_dim: int, hidden: int,
              device: torch.device, verbose: bool = False) -> ActorCritic:
    ref = ActorCritic(obs_dim, hidden)
    ref_sd = ref.state_dict()
    # Backwards compat: checkpoints trained on 46-obs / 10-disc load into
    # the 49-obs / 11-disc brain. Matching tensors copy; the new weave-logit
    # row and the 3 new obs columns init fresh (small rand = explore fast).
    own = {}
    for k, v in state.items():
        if k in ref_sd and ref_sd[k].shape == v.shape:
            own[k] = v
    missing = [k for k in ref_sd if k not in own]
    if missing and any(k in state for k in missing):
        # shape-changed but present: widen by zero-pad (obs trunk) / row-append
        for k in missing:
            if k not in state:
                continue
            old = state[k]
            new_shape = ref_sd[k].shape
            if old.shape == new_shape:
                own[k] = old
                continue
            if k == "trunk.0.weight" and len(new_shape) == 2:
                w = torch.zeros(new_shape, dtype=old.dtype)
                w[:, :old.shape[1]] = old
                # fresh columns for the 3 web feats: small random, not dead zero
                with torch.no_grad():
                    torch.nn.init.orthogonal_(w[:, old.shape[1]:], gain=0.1)
                own[k] = w
            elif k == "logits.weight" and len(new_shape) == 2:
                w = torch.zeros(new_shape, dtype=old.dtype)
                w[:old.shape[0]] = old
                with torch.no_grad():
                    torch.nn.init.orthogonal_(w[old.shape[0]:], gain=0.01)
                own[k] = w
            elif k == "logits.bias" and len(new_shape) == 1:
                b = torch.zeros(new_shape, dtype=old.dtype)
                b[:old.shape[0]] = old
                own[k] = b
            else:
                own[k] = ref_sd[k]
    net = ActorCritic(obs_dim, hidden)
    net.load_state_dict({**ref_sd, **own}, strict=True)
    if verbose and missing:
        print(f"build_net: carried {len(own)} tensors, "
              f"fresh {missing} (old brain upgraded)", flush=True)
    net.to(device)
    net.eval()
    return net


def _widen_snapshot(state: dict, ref_sd: dict) -> dict:
    """Widen one frozen league snapshot to the current brain shape.

    Same rule as build_net: matched tensors copy, the 3 web obs columns and
    the weave logit row init fresh. Runs once at resume, not every iter."""
    own = {}
    for k, v in state.items():
        if k in ref_sd and ref_sd[k].shape == v.shape:
            own[k] = v
    for k in ("trunk.0.weight", "logits.weight", "logits.bias"):
        if k in own or k not in state:
            continue
        old, new_shape = state[k], ref_sd[k].shape
        if k == "trunk.0.weight" and len(new_shape) == 2:
            w = torch.zeros(new_shape, dtype=old.dtype)
            w[:, :old.shape[1]] = old
            with torch.no_grad():
                torch.nn.init.orthogonal_(w[:, old.shape[1]:], gain=0.1)
            own[k] = w
        elif k == "logits.weight" and len(new_shape) == 2:
            w = torch.zeros(new_shape, dtype=old.dtype)
            w[:old.shape[0]] = old
            with torch.no_grad():
                torch.nn.init.orthogonal_(w[old.shape[0]:], gain=0.01)
            own[k] = w
        elif k == "logits.bias" and len(new_shape) == 1:
            b = torch.zeros(new_shape, dtype=old.dtype)
            b[:old.shape[0]] = old
            own[k] = b
    return {**{k: v.detach().clone() for k, v in ref_sd.items()}, **own}


def outcome_of(health, seat: int) -> int:
    """+1 win, -1 loss, 0 draw (timeout) from `seat`'s point of view."""
    me, en = health[seat], health[1 - seat]
    if en <= 0 and me > 0:
        return 1
    if me <= 0:
        return -1
    return 0


# --------------------------------------------------------------------------
# evaluation vs the scripted bot (fixed seed, deterministic policy)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net: ActorCritic, episodes: int = 128, seed: int = 999) -> dict:
    """Learner (seat 0) vs the scripted sparring bot, deterministic actions.

    Per-drill split (melee / heal-or-die). A drill only reports when it has
    >=4 finished episodes, otherwise null so a 1-sample split can't fake a
    trend. Only the first episode of every env is counted.
    """
    env = DuelEnv(episodes, seed=seed, max_ticks=20 * 45)
    env.drill_p = 0.25  # fixed eval mix: 3:1 melee:heal
    bot = ScriptedBot(episodes, np.random.default_rng(seed))
    # per drill: wins, losses, draws, dmg dealt, dmg taken
    per = {0: [0, 0, 0, 0.0, 0.0], 1: [0, 0, 0, 0.0, 0.0],
           2: [0, 0, 0, 0.0, 0.0], 3: [0, 0, 0, 0.0, 0.0]}
    done_flags = np.zeros(episodes, bool)
    finished = 0

    obs = env.reset()
    drills = np.array(env.drill, copy=True)
    guard = 0
    while finished < episodes and guard < 5000:
        guard += 1
        bot_act = bot.act(env)
        cont, disc, _, _ = act_batch(net, obs[:, 0], deterministic=True)
        bot_act[:, 0] = pack_action(cont, disc)
        obs, rew, done, info = env.step(bot_act)
        for f in info.get("finished", []):
            i = f["env"]
            if done_flags[i]:
                continue
            done_flags[i] = True
            finished += 1
            # BUGFIX: `drills` still holds the drill the episode STARTED with
            # (it is only refreshed after this loop). The old code refreshed it
            # inside the loop, so a 2nd finish in the same step got the wrong drill.
            p = per[int(drills[i])]
            p[3] += f["dmg"][0]
            p[4] += f["dmg"][1]
            oc = outcome_of(f["health"], 0)
            p[0 if oc == 1 else 1 if oc == -1 else 2] += 1
        drills = np.array(env.drill, copy=True)

    n_all = max(finished, 1)
    tot = lambda j: sum(per[d][j] for d in per)  # noqa: E731
    out = {
        "win_rate": tot(0) / n_all,
        "loss_rate": tot(1) / n_all,
        "draw_rate": tot(2) / n_all,
        "dmg_dealt": round(tot(3) / n_all, 2),
        "dmg_taken": round(tot(4) / n_all, 2),
        "n": finished,
    }
    for drill_id, key in enumerate(DRILL_NAMES):
        w, l, d, dd, dt = per[drill_id]
        n = w + l + d
        out[f"{key}_n"] = n
        if n >= 4:
            out[f"{key}_win"] = w / n
            out[f"{key}_draw"] = d / n
            out[f"{key}_dmg"] = round(dd / n, 2)
        else:
            out[f"{key}_win"] = None
            out[f"{key}_draw"] = None
            out[f"{key}_dmg"] = None
    # escape + repot rates: the two new drills report what they teach
    for key in ("web", "repot"):
        n = out.get(f"{key}_n", 0) or 0
        out[f"{key}_rate"] = out.get(f"{key}_win")  # alias: win == lesson learned
    return out


# --------------------------------------------------------------------------
# training-episode statistics (rolling window, split by opponent + drill)
# --------------------------------------------------------------------------
def summarize(buf: collections.deque) -> dict:
    """buf rows: (opp_kind, drill, raw_return, length, outcome)."""
    out: dict = {}
    if not buf:
        return out
    a = np.asarray(buf, dtype=np.float64).reshape(-1, 5)

    def m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    out["ret"] = m(a[:, 2])
    out["ep_len"] = m(a[:, 3])
    out["draw"] = m(a[:, 4] == 0)
    for k, name in enumerate(KIND_NAMES):
        s = a[:, 0] == k
        out[f"ret_{name}"] = m(a[s, 2])
        out[f"win_{name}"] = m(a[s, 4] == 1)
        out[f"draw_{name}"] = m(a[s, 4] == 0)
    for d, name in enumerate(DRILL_NAMES):
        s = a[:, 1] == d
        out[f"ret_{name}"] = m(a[s, 2])
        out[f"draw_{name}"] = m(a[s, 4] == 0)
        out[f"len_{name}"] = m(a[s, 3])
    return out


def fmt(x, spec="+6.2f"):
    return format(x, spec) if x is not None and np.isfinite(x) else "   -  "


# --------------------------------------------------------------------------
# opponent assignment
# --------------------------------------------------------------------------
def draw_kinds(rng, n: int, args, have_pool: bool, have_champ: bool) -> np.ndarray:
    p_bot = args.bot_p
    p_mir = args.mirror_p
    p_champ = args.champ_p if have_champ else 0.0
    p_league = max(0.0, 1.0 - p_bot - p_mir - p_champ)  # remainder = league
    if not have_pool:  # nothing to put in the league yet -> mirror instead
        p_mir += p_league
        p_league = 0.0
    p = np.array([p_bot, p_mir, p_league, p_champ], dtype=np.float64)
    p /= p.sum()
    return rng.choice(4, size=n, p=p)


def pick_league_ids(rng, pool_len: int, k: int) -> list[int]:
    """Half the picks come from the newest 3 snapshots, half uniform."""
    ids: list[int] = []
    for _ in range(k):
        if rng.random() < 0.5:
            i = pool_len - 1 - int(rng.integers(0, min(3, pool_len)))
        else:
            i = int(rng.integers(0, pool_len))
        if i not in ids:
            ids.append(i)
    return ids


def save_ckpt(ppo: PPO, it: int, history: list, pool: list, args) -> None:
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": ppo.policy.state_dict(),
        "opt": ppo.opt.state_dict(),
        "it": it,
        "history": history,
        "pool": pool,
        "args": vars(args),
    }, outdir / "ckpt.pt")
    torch.save(ppo.policy.state_dict(), outdir / "ppo.pt")
    (outdir / "history.json").write_text(json.dumps(history, indent=2))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--anneal-iters", type=int, default=0,
                    help="length of the lr/entropy anneal. 0 = use --iters. "
                         "Keep it at the ORIGINAL --iters when extending a run.")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=256)
    ap.add_argument("--max-ticks", type=int, default=1200,
                    help="episode timeout in ticks (old default ~4000 made every "
                         "stall cost ~16 rollouts of useless data)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.002,
                    help="entropy bonus. Must stay well below the policy-loss "
                         "scale or the gradient just maximises randomness")
    # opponent mix (league = whatever is left over)
    ap.add_argument("--bot-p", type=float, default=0.25,
                    help="share of envs vs the scripted sparring bot. NOTE: the "
                         "eval is also vs the scripted bot, so the higher this "
                         "is the less 'held-out' the eval becomes")
    ap.add_argument("--mirror-p", type=float, default=0.30)
    ap.add_argument("--champ-p", type=float, default=0.15,
                    help="share of envs vs the best-by-eval checkpoint")
    ap.add_argument("--gate-iters", type=int, default=100,
                    help="curriculum gate: before this iter, ALL bot_p envs "
                         "face the dummy ScriptedBot and mirror/league/champ "
                         "are off. A random brain loses 83%% of fights (kit1 "
                         "iter 25: dealt 2, took 36); throwing Aggro at it on "
                         "iter 1 teaches turtling, not fighting. After the "
                         "gate the full style rotation + self-play unlock.")
    ap.add_argument("--league-k", type=int, default=3,
                    help="distinct league snapshots used per iteration")
    ap.add_argument("--single-seat", action="store_true",
                    help="mirror envs only train ONE seat (old behaviour)")
    ap.add_argument("--draw-penalty", type=float, default=1.0,
                    help="extra reward subtracted from the learner when an "
                         "episode times out with both alive. Tune it to your "
                         "reward scale: a draw must be clearly worse than "
                         "an average fight")
    ap.add_argument("--drill-p", type=float, default=0.25,
                    help="share of envs spawned into the heal-or-die drill "
                         "(ramps 0 -> this over --drill-ramp iters). 0 = all melee.")
    ap.add_argument("--drill-ramp", type=int, default=150)
    ap.add_argument("--aux-iters", type=int, default=150,
                    help="iterations over which the aim regression loss anneals "
                         "linearly to 0")
    ap.add_argument("--aux-coef", type=float, default=0.6)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-eps", type=int, default=128,
                    help="episodes per eval (48 was too noisy: 20%%<->100%% swings)")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--stat-window", type=int, default=300,
                    help="episodes in the rolling training-stat window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-pool", action="store_true",
                    help="drop league snapshots loaded from the checkpoint")
    ap.add_argument("--reset-std", action="store_true",
                    help="on resume, force log_std back to -1.5. By default it "
                         "is only CLAMPED to <= -1.0 (the old code overwrote it "
                         "on every resume, which kicked a healthy policy)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--outdir", default="runs",
                    help="checkpoint dir (use runs/kit1 for the kit retrain)")
    args = ap.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RUNS.mkdir(exist_ok=True)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    anneal_total = args.anneal_iters if args.anneal_iters > 0 else args.iters

    E = args.envs
    env = DuelEnv(E, seed=args.seed, max_ticks=args.max_ticks)
    obs = env.reset()
    obs_dim = obs.shape[-1]

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
          flush=True)

    ppo = PPO(obs_dim, lr=args.lr, hidden=args.hidden, batch=args.batch)
    ppo.policy.to(device)
    seat = np.random.default_rng(args.seed).integers(0, 2, size=E)
    # sparring rotation: the scripted dummy plus the three style bots.
    # BOT envs roll one style per env per iter, so the learner faces axe
    # openers (Aggro), live shields to crack (Turtle) and whiff-baits
    # (Runner) - not just the dummy's rhythm. Eval stays ScriptedBot-only
    # so the baseline number keeps meaning the same thing.
    style_bots = [ScriptedBot(E, np.random.default_rng(args.seed + 1)),
                  AggroBot(E, np.random.default_rng(args.seed + 2)),
                  TurtleBot(E, np.random.default_rng(args.seed + 3)),
                  RunnerBot(E, np.random.default_rng(args.seed + 4))]
    pool: list[dict] = []
    history: list[dict] = []
    start_it = 1

    champ = {"state": None, "net": None}

    resume_ckpt = outdir / "ckpt.pt"
    if args.resume and resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        # shape-upgrade path: old 46x10 brains widen into 49x11 (see build_net).
        _upgraded = False
        try:
            ppo.policy.load_state_dict(ck["model"])
        except RuntimeError as e:
            print(f"resume: {e}\nresume: upgrading old brain -> widening", flush=True)
            up = build_net(ck["model"], obs_dim, args.hidden, torch.device("cpu"))
            ppo.policy.load_state_dict(up.state_dict())
            # fresh head explores: reopen std so the new buttons get tried
            with torch.no_grad():
                ppo.policy.log_std.fill_(-1.2)
            _upgraded = True
        if _upgraded:
            # Adam's exp_avg / exp_avg_sq are param-shaped (128x46, 10x128).
            # Loading them into a 128x49 / 11x128 net crashes on the first
            # step with "size 46 must match 49". Old momentum points the
            # wrong way anyway (it learned the old shape), so restart it:
            # params + lr schedule carry, optimizer memory does not.
            print("resume: optimizer state dropped (shape upgrade 46->49), "
                  "fresh Adam, lr schedule kept", flush=True)
            ppo.opt = torch.optim.Adam(ppo.policy.parameters(),
                                       lr=ppo.opt.param_groups[0]["lr"], eps=1e-5)
        else:
            ppo.opt.load_state_dict(ck["opt"])
        for s in ppo.opt.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
        history = ck.get("history", [])
        pool = ck.get("pool", [])
        # widen frozen league snapshots ONCE at resume: they are old-shape
        # (46x10) brains, and re-widening them every iter spams the log.
        _narrow = [s for s in pool
                   if s.get("trunk.0.weight", torch.zeros(0)).shape[1] != obs_dim
                   or s.get("logits.weight", torch.zeros(0)).shape[0] != 11]
        if _narrow:
            ref = ActorCritic(obs_dim, args.hidden)
            ref_sd = ref.state_dict()
            pool = [_widen_snapshot(s, ref_sd) for s in pool]
            print(f"resume: widened {len(_narrow)} league snapshots "
                  f"to {obs_dim}x11 (once, quiet from here)", flush=True)
        start_it = int(ck.get("it", 0)) + 1
        with torch.no_grad():
            if args.reset_std:
                ppo.policy.log_std.fill_(-1.5)
            else:
                # only undo an inflated std; never overwrite a learned one
                ppo.policy.log_std.clamp_(max=-1.0)
        if args.reset_pool and pool:
            pool = []
            print("pool flushed (--reset-pool)", flush=True)
        best_file = outdir / "ppo_best.pt"
        if best_file.exists():
            champ["state"] = torch.load(best_file, map_location="cpu")
        print(f"resumed from iter {start_it - 1} ({len(pool)} league snapshots, "
              f"champion {'yes' if champ['state'] is not None else 'no'})",
              flush=True)

    # best-keep: by eval win_rate vs scripted (tie-break: damage diff).
    def _best_key(h):
        return (h.get("win_rate", -1.0),
                h.get("dmg_dealt", 0.0) - h.get("dmg_taken", 0.0))
    best_key = (-1.0, float("-inf"))
    best_iter = 0
    for _h in history:
        if _best_key(_h) > best_key:
            best_key = _best_key(_h)
            best_iter = _h.get("iter", 0)
    if best_iter:
        print(f"best-keep loaded: iter {best_iter} win {best_key[0]*100:.1f}%",
              flush=True)

    if start_it > args.iters:
        print(f"checkpoint already at iter {start_it - 1} >= --iters {args.iters}; "
              "nothing to train (raise --iters to continue)", flush=True)
        return

    it = start_it - 1
    n_eps = 0
    ep_buf: collections.deque = collections.deque(maxlen=args.stat_window)
    metrics_path = outdir / "metrics.jsonl"

    def log_metrics(d: dict) -> None:
        def clean(v):
            if isinstance(v, (float, np.floating)):
                v = float(v)
                return v if np.isfinite(v) else None
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            return v
        d = {"t": round(time.time(), 1), **clean(d)}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    rng = np.random.default_rng(args.seed + 7 + start_it)
    cur_drill = np.array(env.drill, copy=True)
    all_envs = np.arange(E)

    t_start = time.time()
    log_metrics({"event": "started" if start_it <= 1 else "resumed",
                 "iter": start_it})

    try:
        for it in range(start_it, args.iters + 1):
            # ---- schedules ---------------------------------------------------
            frac = min(1.0, it / anneal_total)
            for g in ppo.opt.param_groups:
                g["lr"] = args.lr * (1 - 0.85 * frac)
            ppo.ent_coef = args.ent_coef * (1 - 0.8 * frac)

            if args.drill_ramp > 0:
                env.drill_p = args.drill_p * min(1.0, it / args.drill_ramp)
            else:
                env.drill_p = args.drill_p

            # aim tutor: smooth linear decay to exactly 0 (no cliff)
            aux_coef = args.aux_coef * max(0.0, 1.0 - it / args.aux_iters) \
                if args.aux_iters > 0 else 0.0

            # ---- opponents for THIS iteration, per env -----------------------
            # curriculum gate: pre-gate, everything is the killable dummy.
            # Post-gate the full mix (styles + mirror + league + champ).
            gated = it <= args.gate_iters and args.gate_iters > 0
            if gated:
                kinds = np.full(E, BOT, dtype=np.int64)
            else:
                kinds = draw_kinds(rng, E, args, have_pool=len(pool) > 0,
                                   have_champ=champ["state"] is not None)
            bot_envs = np.flatnonzero(kinds == BOT)
            mirror_envs = np.flatnonzero(kinds == MIRROR)
            league_envs = np.flatnonzero(kinds == LEAGUE)
            champ_envs = np.flatnonzero(kinds == CHAMP)

            frozen: list[tuple[ActorCritic, np.ndarray]] = []
            if league_envs.size:
                ids = pick_league_ids(rng, len(pool), args.league_k)
                which = rng.integers(0, len(ids), size=league_envs.size)
                for j, pid in enumerate(ids):
                    sel = league_envs[which == j]
                    if sel.size:
                        frozen.append((build_net(pool[pid], obs_dim,
                                                 args.hidden, device), sel))
            if champ_envs.size:
                if champ["net"] is None:
                    champ["net"] = build_net(champ["state"], obs_dim,
                                             args.hidden, device)
                frozen.append((champ["net"], champ_envs))

            # learner "slots": one per env (its own seat) + the other seat too
            # in mirror envs (same policy plays both sides -> double the data)
            if args.single_seat:
                extra = np.zeros(0, dtype=np.int64)
            else:
                extra = mirror_envs
            slot_env = np.concatenate([all_envs, extra]).astype(np.int64)
            slot_seat = np.concatenate([seat, 1 - seat[extra]]).astype(np.int64)
            S = slot_env.size
            slots_by_env: list[list[int]] = [[e] for e in range(E)]
            for j, e in enumerate(extra):
                slots_by_env[int(e)].append(E + j)
            slot_kind = kinds[slot_env]
            other = 1 - seat

            buf_obs, buf_cont, buf_disc, buf_aux = [], [], [], []
            buf_logp, buf_val, buf_rew, buf_done = [], [], [], []
            n_draw_pen = 0

            # ---- rollout -----------------------------------------------------
            dev = net_device(ppo.policy)
            for _ in range(args.ticks):
                learner_obs = obs[slot_env, slot_seat]
                cont, disc, _, _ = act_batch(ppo.policy, learner_obs)

                err_yaw, err_pitch = aim_delta(env, slot_env, slot_seat)
                aux = np.stack([
                    np.clip(err_yaw / MAX_LOOK, -1.0, 1.0),
                    np.clip(err_pitch / MAX_LOOK, -1.0, 1.0),
                ], axis=1).astype(np.float32)

                # logp/value recomputed through evaluate() so the stored
                # logp matches exactly what ppo.update() will recompute
                with torch.no_grad():
                    logp_t, _, value_t, _ = ppo.policy.evaluate(
                        to_t(learner_obs, dev), to_t(cont, dev), to_t(disc, dev))
                logp = logp_t.detach().cpu().numpy()
                value = value_t.detach().cpu().numpy()

                act = np.zeros((E, 2, DuelEnv.ACTION_DIM), np.float32)
                act[slot_env, slot_seat] = pack_action(cont, disc)
                if bot_envs.size:
                    # post-gate: role one style per env per iter. Pre-gate
                    # everything is the dummy (gated kinds are all BOT).
                    style = (style_bots[0] if gated else
                             style_bots[int(rng.integers(0, len(style_bots)))])
                    bot_act = style.act(env)
                    act[bot_envs, other[bot_envs]] = bot_act[bot_envs,
                                                             other[bot_envs]]
                for net, sel in frozen:
                    o_cont, o_disc, _, _ = act_batch(net, obs[sel, other[sel]])
                    act[sel, other[sel]] = pack_action(o_cont, o_disc)

                drill_before = cur_drill
                obs, rew, done, info = env.step(act, record_events=False)
                learner_rew = np.array(rew[slot_env, slot_seat], dtype=np.float32)

                for f in info.get("finished", []):
                    e = int(f["env"])
                    for k in slots_by_env[e]:
                        s = int(slot_seat[k])
                        oc = outcome_of(f["health"], s)
                        if oc == 0 and args.draw_penalty > 0:
                            learner_rew[k] -= args.draw_penalty
                            n_draw_pen += 1
                        ep_buf.append((int(slot_kind[k]), int(drill_before[e]),
                                       float(f["ret"][s]), float(f["len"]), oc))
                        n_eps += 1
                cur_drill = np.array(env.drill, copy=True)

                buf_obs.append(learner_obs)
                buf_cont.append(cont)
                buf_disc.append(disc)
                buf_aux.append(aux)
                buf_logp.append(logp)
                buf_val.append(value)
                buf_rew.append(learner_rew)
                buf_done.append(np.asarray(done, dtype=np.float32)[slot_env])

            # ---- GAE ---------------------------------------------------------
            last_obs = obs[slot_env, slot_seat]
            _, _, _, last_val = act_batch(ppo.policy, last_obs)

            T = len(buf_rew)
            adv = np.zeros((T, S), np.float32)
            last = np.zeros(S, np.float32)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else buf_val[t + 1]
                not_done = 1.0 - buf_done[t]
                delta = buf_rew[t] + ppo.gamma * next_val * not_done - buf_val[t]
                last = delta + ppo.gamma * ppo.lam * not_done * last
                adv[t] = last

            tt = lambda x: torch.as_tensor(  # noqa: E731
                np.concatenate(x), dtype=torch.float32, device=device)
            flat_obs = tt(buf_obs)
            flat_cont = tt(buf_cont)
            flat_disc = tt(buf_disc)
            flat_aux = tt(buf_aux)
            flat_logp = tt(buf_logp)
            flat_val = tt(buf_val)
            flat_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32,
                                       device=device)
            flat_ret = flat_adv + flat_val
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            if not (torch.isfinite(flat_adv).all() and torch.isfinite(flat_ret).all()):
                print(f"it {it}: non-finite advantages/returns, skipping update",
                      flush=True)
                continue

            stats = ppo.update(flat_obs, flat_cont, flat_disc, flat_logp,
                               flat_adv, flat_ret, aux=flat_aux, aux_coef=aux_coef)

            # ---- league ------------------------------------------------------
            if it % args.snapshot_every == 0:
                pool.append(ppo.snapshot())
                if len(pool) > args.pool_size:
                    pool.pop(0)

            # ---- logging -----------------------------------------------------
            ts = summarize(ep_buf)
            steps = (it - start_it + 1) * args.ticks * E
            sps = steps / max(time.time() - t_start, 1e-6)
            mode = (f"b{bot_envs.size}/m{mirror_envs.size}/"
                    f"l{league_envs.size}/c{champ_envs.size}")
            print(f"it {it:5d} [{mode:11s}] "
                  f"| ret {fmt(ts.get('ret'))} "
                  f"(bot {fmt(ts.get('ret_bot'))} mir {fmt(ts.get('ret_mirror'))} "
                  f"lge {fmt(ts.get('ret_league'))} chp {fmt(ts.get('ret_champ'))}) "
                  f"| len {fmt(ts.get('ep_len'), '5.0f')} "
                  f"| draw {fmt(ts.get('draw'), '4.2f')} "
                  f"(heal {fmt(ts.get('draw_heal'), '4.2f')}) "
                  f"| ent {stats['entropy']:.3f} "
                  f"| aux {stats.get('aux_loss', 0.0):.3f} "
                  f"| pi {stats['policy_loss']:+.3f} "
                  f"| v {stats['value_loss']:.3f} | {sps:6.0f} sps",
                  flush=True)
            log_metrics({"iter": it, "mode": mode, "eps": n_eps,
                         "ent": stats["entropy"],
                         "aux": stats.get("aux_loss", 0.0),
                         "pi": stats["policy_loss"],
                         "v": stats["value_loss"], "sps": sps,
                         "lr": ppo.opt.param_groups[0]["lr"],
                         "n_bot": int(bot_envs.size),
                         "n_mirror": int(mirror_envs.size),
                         "n_league": int(league_envs.size),
                         "n_champ": int(champ_envs.size),
                         "samples": int(S * T),
                         **ts})

            # ---- eval --------------------------------------------------------
            if it % args.eval_every == 0:
                ev = evaluate(ppo.policy, episodes=args.eval_eps)
                history.append({"iter": it, **ev})
                log_metrics({"iter": it, "event": "eval", **ev})
                print(f"   eval vs scripted ({ev['n']} eps): "
                      f"win {ev['win_rate']*100:5.1f}% "
                      f"loss {ev['loss_rate']*100:5.1f}% "
                      f"draw {ev['draw_rate']*100:4.1f}% "
                      f"dmg {ev['dmg_dealt']:.1f}/{ev['dmg_taken']:.1f} | "
                      f"melee win {fmt(ev['melee_win'], '.2f')} "
                      f"heal win {fmt(ev['heal_win'], '.2f')} "
                      f"web win {fmt(ev['web_win'], '.2f')} "
                      f"repot win {fmt(ev['repot_win'], '.2f')}",
                      flush=True)
                if _best_key(history[-1]) > best_key:
                    best_key = _best_key(history[-1])
                    best_iter = it
                    torch.save(ppo.policy.state_dict(), outdir / "ppo_best.pt")
                    torch.save({
                        "model": ppo.policy.state_dict(),
                        "opt": ppo.opt.state_dict(),
                        "it": it,
                        "history": history,
                        "pool": pool,
                        "args": vars(args),
                    }, outdir / "ckpt_best.pt")
                    champ["state"] = ppo.snapshot()  # becomes the champion
                    champ["net"] = None
                    log_metrics({"iter": it, "event": "best",
                                 "win_rate": ev["win_rate"]})
                    print(f"   *** new best: iter {it} win "
                          f"{ev['win_rate']*100:.1f}% -> ppo_best.pt "
                          f"(now the champion opponent)", flush=True)

            if it % args.save_every == 0 or it == args.iters:
                save_ckpt(ppo, it, history, pool, args)
                log_metrics({"iter": it, "event": "saved"})

    except KeyboardInterrupt:
        print("\ninterrupted - saving", flush=True)
        save_ckpt(ppo, it, history, pool, args)
        log_metrics({"iter": it, "event": "stopped"})

    save_ckpt(ppo, it, history, pool, args)
    print(f"saved {outdir / 'ppo.pt'} and {outdir / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()