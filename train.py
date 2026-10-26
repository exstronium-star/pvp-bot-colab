"""Self-play PPO trainer for the sword-duel sim.

The learner controls one seat per duel; the other seat is either the current
policy (mirror), a frozen snapshot from the league, or the scripted sparring
bot. Frozen snapshots are what stop self-play from cycling; the scripted bot is
a stable anchor that teaches "walk at the enemy and hit them".

Aim: the policy's own yaw/pitch output is always what executes. An auxiliary
regression loss on the action mean teaches aiming early and is annealed away,
instead of overwriting the action (the old curriculum did that, so the policy
never received a gradient about aiming and collapsed into spinning).

    python -u train.py --iters 500 --envs 32 --ticks 256
    python -u train.py --iters 500 --envs 32 --ticks 256 --resume
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from mc_pvp.bots import ScriptedBot
from mc_pvp.ppo import PPO, ActorCritic
from mc_pvp.sim import COOLDOWN_TICKS, EYE_HEIGHT, DuelEnv

RUNS = pathlib.Path("runs")
CKPT = RUNS / "ckpt.pt"
# Kit retrain (46-obs/10-disc) must not clobber the sword-only checkpoint the
# bridge currently serves. Point --outdir at runs/kit1 for the real run.
OUTDIR = RUNS
MAX_LOOK = 20.0


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
    return act


def aim_delta(env: DuelEnv, seats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw/pitch delta (degrees) that points each seat at its opponent's chest."""
    n = len(seats)
    idx = np.arange(n)
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


@torch.no_grad()
def evaluate(net: ActorCritic, episodes: int = 48, seed: int = 999) -> dict:
    """Learner (seat 0) vs the scripted sparring bot, deterministic actions."""
    env = DuelEnv(episodes, seed=seed, max_ticks=20 * 45)
    bot = ScriptedBot(episodes, np.random.default_rng(seed))
    obs = env.reset()
    wins = losses = draws = 0
    dmg_dealt = 0.0
    dmg_taken = 0.0
    done_flags = np.zeros(episodes, bool)
    finished = 0

    while finished < episodes:
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
            dmg_dealt += f["dmg"][0]
            dmg_taken += f["dmg"][1]
            if f["health"][1] <= 0 and f["health"][0] > 0:
                wins += 1
            elif f["health"][0] <= 0:
                losses += 1
            else:
                draws += 1

    return {
        "win_rate": wins / max(finished, 1),
        "loss_rate": losses / max(finished, 1),
        "draw_rate": draws / max(finished, 1),
        "dmg_dealt": dmg_dealt / max(finished, 1),
        "dmg_taken": dmg_taken / max(finished, 1),
    }


def save_ckpt(ppo: PPO, it: int, history: list, pool: list, args) -> None:
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / "ckpt.pt"
    torch.save({
        "model": ppo.policy.state_dict(),
        "opt": ppo.opt.state_dict(),
        "it": it,
        "history": history,
        "pool": pool,
        "args": vars(args),
    }, ckpt)
    torch.save(ppo.policy.state_dict(), outdir / "ppo.pt")
    (outdir / "history.json").write_text(json.dumps(history, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.002,
                    help="entropy bonus. Must stay well below the policy-loss "
                         "scale or the gradient just maximises randomness")
    ap.add_argument("--mirror-p", type=float, default=0.35)
    ap.add_argument("--bot-p", type=float, default=0.35,
                    help="chance the opponent is the scripted sparring bot")
    ap.add_argument("--aux-iters", type=int, default=150,
                    help="iterations over which the aim regression loss is annealed")
    ap.add_argument("--aux-coef", type=float, default=0.6)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-pool", action="store_true",
                    help="drop league snapshots loaded from the checkpoint "
                         "(use when the old pool teaches bad habits)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = cuda on colab GPU runtimes, else cpu")
    ap.add_argument("--outdir", default="runs",
                    help="checkpoint dir (use runs/kit1 so the kit retrain "
                         "cannot overwrite the sword-only checkpoint)")
    args = ap.parse_args()

    torch.set_num_threads(2)          # 4 threads on tiny matmuls is slower
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RUNS.mkdir(exist_ok=True)

    env = DuelEnv(args.envs, seed=args.seed)
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
    seat = np.random.default_rng(args.seed).integers(0, 2, size=args.envs)
    bot = ScriptedBot(args.envs, np.random.default_rng(args.seed + 1))
    pool: list[dict] = []
    history: list[dict] = []
    start_it = 1

    resume_ckpt = pathlib.Path(args.outdir) / "ckpt.pt"
    if args.resume and resume_ckpt.exists():
        ck = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        ppo.policy.load_state_dict(ck["model"])
        ppo.opt.load_state_dict(ck["opt"])
        for s in ppo.opt.state.values():
            for k, v in s.items():
                if torch.is_tensor(v):
                    s[k] = v.to(device)
        history = ck.get("history", [])
        pool = ck.get("pool", [])
        start_it = int(ck.get("it", 0)) + 1
        with torch.no_grad():
            # earlier runs let the entropy bonus inflate the exploration std;
            # pull it back to a sane 0.22 so PPO samples are meaningful
            ppo.policy.log_std.fill_(-1.5)
        if args.reset_pool and pool:
            pool = []
            print("pool flushed (--reset-pool): league restarts empty", flush=True)
        print(f"resumed from iter {start_it - 1} ({len(pool)} league snapshots)",
              flush=True)

    buf_obs, buf_cont, buf_disc, buf_aux, buf_logp, buf_val, buf_rew, buf_done = (
        [] for _ in range(8))

    if start_it > args.iters:
        print(f"checkpoint already at iter {start_it - 1} >= --iters {args.iters}; "
              "nothing to train (raise --iters to continue)", flush=True)
        return
    it = start_it - 1
    ep_returns = []
    ep_lens = []
    metrics_path = pathlib.Path(args.outdir) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def log_metrics(d: dict) -> None:
        def clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            return v
        d = {"t": round(time.time(), 1), **clean(d)}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    t_start = time.time()
    log_metrics({"event": "started" if start_it <= 1 else "resumed",
                 "iter": start_it})

    try:
        for it in range(start_it, args.iters + 1):
            # anneal
            frac = it / args.iters
            for g in ppo.opt.param_groups:
                g["lr"] = args.lr * (1 - 0.85 * frac)
            ppo.ent_coef = args.ent_coef * (1 - 0.8 * frac)

            # ---- pick opponent for this iteration ---------------------------
            r = np.random.random()
            if r < args.bot_p and len(pool) > 0:
                mode = "bot"
            elif r < args.bot_p + args.mirror_p or len(pool) == 0:
                mode = "mirror"
            else:
                mode = "league"

            if mode == "mirror":
                opp = ppo.policy
            elif mode == "league":
                opp = ActorCritic(obs_dim, args.hidden)
                opp.load_state_dict(pool[np.random.randint(len(pool))])
                opp.eval()
            else:
                opp = None

            buf_obs.clear(); buf_cont.clear(); buf_disc.clear(); buf_aux.clear()
            buf_logp.clear(); buf_val.clear(); buf_rew.clear(); buf_done.clear()

            aux_coef = 0.0
            if it <= args.aux_iters:
                aux_coef = args.aux_coef * (1.0 - 0.9 * it / args.aux_iters)

            for _ in range(args.ticks):
                learner_obs = obs[np.arange(args.envs), seat]
                cont, disc, _, _ = act_batch(ppo.policy, learner_obs)

                # supervised aim target for this tick (normalised action units)
                err_yaw, err_pitch = aim_delta(env, seat)
                aux = np.stack([
                    np.clip(err_yaw / MAX_LOOK, -1.0, 1.0),
                    np.clip(err_pitch / MAX_LOOK, -1.0, 1.0),
                ], axis=1).astype(np.float32)

                with torch.no_grad():
                    logp_t, _, value_t, _ = ppo.policy.evaluate(
                        to_t(learner_obs), to_t(cont), to_t(disc))
                logp = logp_t.numpy()
                value = value_t.numpy()

                act = np.zeros((args.envs, 2, DuelEnv.ACTION_DIM), np.float32)
                act[np.arange(args.envs), seat] = pack_action(cont, disc)
                other = 1 - seat
                if opp is None:
                    bot_act = bot.act(env)
                    act[np.arange(args.envs), other] = bot_act[
                        np.arange(args.envs), other]
                else:
                    o_cont, o_disc, _, _ = act_batch(
                        opp, obs[np.arange(args.envs), other])
                    act[np.arange(args.envs), other] = pack_action(o_cont, o_disc)

                obs, rew, done, info = env.step(act, record_events=False)
                learner_rew = rew[np.arange(args.envs), seat]

                buf_obs.append(learner_obs)
                buf_cont.append(cont)
                buf_disc.append(disc)
                buf_aux.append(aux)
                buf_logp.append(logp)
                buf_val.append(value)
                buf_rew.append(learner_rew)
                buf_done.append(done.astype(np.float32))

                for f in info.get("finished", []):
                    ep_returns.append(f["ret"][seat[f["env"]]])
                    ep_lens.append(f["len"])

            # ---- GAE ---------------------------------------------------------
            last_obs = obs[np.arange(args.envs), seat]
            _, _, _, last_val = act_batch(ppo.policy, last_obs)

            T = len(buf_rew)
            adv = np.zeros((T, args.envs), np.float32)
            last = np.zeros(args.envs, np.float32)
            for t in reversed(range(T)):
                next_val = last_val if t == T - 1 else buf_val[t + 1]
                not_done = 1.0 - buf_done[t]
                delta = buf_rew[t] + ppo.gamma * next_val * not_done - buf_val[t]
                last = delta + ppo.gamma * ppo.lam * not_done * last
                adv[t] = last

            flat_obs = torch.as_tensor(np.concatenate(buf_obs), dtype=torch.float32, device=device)
            flat_cont = torch.as_tensor(np.concatenate(buf_cont), dtype=torch.float32, device=device)
            flat_disc = torch.as_tensor(np.concatenate(buf_disc), dtype=torch.float32, device=device)
            flat_aux = torch.as_tensor(np.concatenate(buf_aux), dtype=torch.float32, device=device)
            flat_logp = torch.as_tensor(np.concatenate(buf_logp), dtype=torch.float32, device=device)
            flat_val = torch.as_tensor(np.concatenate(buf_val), dtype=torch.float32, device=device)
            flat_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=device)
            flat_ret = flat_adv + flat_val
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            stats = ppo.update(flat_obs, flat_cont, flat_disc, flat_logp,
                               flat_adv, flat_ret, aux=flat_aux, aux_coef=aux_coef)

            # ---- league ------------------------------------------------------
            if it % args.snapshot_every == 0:
                pool.append(ppo.snapshot())
                if len(pool) > args.pool_size:
                    pool.pop(0)

            if it % 10 == 0:
                recent = np.mean(ep_returns[-40:]) if ep_returns else float("nan")
                recent_len = np.mean(ep_lens[-40:]) if ep_lens else float("nan")
                sps = it * args.ticks / (time.time() - t_start)
                print(f"it {it:5d} [{mode:6s}] | ret {recent:7.3f} "
                      f"| eps {len(ep_returns):5d} | ent {stats['entropy']:.3f} "
                      f"| aux {stats.get('aux_loss', 0.0):.3f} "
                      f"| pi {stats['policy_loss']:+.3f} "
                      f"| v {stats['value_loss']:.3f} | {sps:5.1f} it/s",
                      flush=True)
                log_metrics({"iter": it, "mode": mode, "ret": recent,
                             "ep_len": recent_len, "eps": len(ep_returns),
                             "ent": stats["entropy"],
                             "aux": stats.get("aux_loss", 0.0),
                             "pi": stats["policy_loss"],
                             "v": stats["value_loss"], "sps": sps})

            if it % args.eval_every == 0:
                ev = evaluate(ppo.policy)
                history.append({"iter": it, **ev})
                log_metrics({"iter": it, "event": "eval", **ev})
                print(f"   eval vs scripted: win {ev['win_rate']*100:5.1f}% "
                      f"loss {ev['loss_rate']*100:5.1f}% "
                      f"draw {ev['draw_rate']*100:4.1f}% "
                      f"dmg {ev['dmg_dealt']:.1f}/{ev['dmg_taken']:.1f}",
                      flush=True)

            if it % args.save_every == 0 or it == args.iters:
                save_ckpt(ppo, it, history, pool, args)
                log_metrics({"iter": it, "event": "saved"})

    except KeyboardInterrupt:
        print("\ninterrupted - saving", flush=True)
        save_ckpt(ppo, it, history, pool, args)
        log_metrics({"iter": it, "event": "stopped"})

    save_ckpt(ppo, it, history, pool, args)
    print(f"saved {pathlib.Path(args.outdir) / 'ppo.pt'} and "
          f"{pathlib.Path(args.outdir) / 'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()
