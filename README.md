# Minecraft 1.21.11 Sword PvP — self-play agent

A tick-accurate sword-duel simulation, calibrated against a real 1.21.11 server,
with PPO self-play training.

## Why a sim at all

Training against a live server means ~20 decisions per second of wall clock and
network jitter in the reward signal. This sim runs **83,640 env-ticks/s**
(167,281 agent-steps/s, ~4,180x real time) at 512 envs on an i5-4460, so 2–3
hours of training buys hundreds of millions of steps. The sim exists to be
*accurate*, not convenient.

## Physics validation

Offline checks (`python -m mc_pvp.validate`) compare against known vanilla values:

| check | sim | vanilla |
|---|---|---|
| jump apex | 1.252 | 1.2522 |
| walk speed | 4.317 b/s | 4.317 b/s |
| sprint speed | 5.612 b/s | 5.612 b/s |
| terminal fall | -3.920 b/t | -3.92 b/t |
| full-charge sword damage | 7.000 | 7.000 |
| half-charge damage | 2.800 | 2.800 |
| hurt immunity | one hit / 10 ticks | same |
| reach | hits at 3.25, misses at 3.6 | same |

Live checks against a real 1.21.11 server (`tools/probe*.mjs`, mineflayer):

| measurement | real 1.21.11 | notes |
|---|---|---|
| walk / sprint / jump | 4.317 / 5.612 / 1.252 | matches sim exactly |
| diamond sword full charge | 7.0 damage | matches |
| crit (falling hit) | 10.5 damage | matches 1.5x |
| knockback, standing target | 1.989 blocks, 0.965 apex | sim is calibrated to this |
| sprint-hit knockback | **unmeasured** | mineflayer can't sprint reliably (rubber-bands to 3.06 b/s) |
| server hit-accept range | ~6 blocks | vanilla leniency; the *client* limits to 3.0, which the sim enforces |

Knockback needed calibration: vanilla source reads strength 0.4, but the measured
arc (1.989 blocks) only reproduces with a horizontal strength of 0.30. Vertical
stays at vanilla 0.4 (`min(0.4, vy/2 + 0.4)`, with the resting vy of -0.0784 -
which is what produces the 0.965 apex).

## Layout

```
mc_pvp/sim.py        tick-accurate duel sim, vectorised over N envs
mc_pvp/bots.py       scripted sparring partner
mc_pvp/ppo.py        PPO, hybrid action space
mc_pvp/validate.py   offline physics checks
train.py             self-play training loop (resumable)
tools/bench_sim.py       throughput benchmark
tools/diag_policy.py     what a checkpoint actually does in a duel
tools/probe_duel.py      sim sanity probe (scripted vs scripted)
tools/fetch_server.py    download vanilla server jar
tools/probe*.mjs         mineflayer ground-truth probes
tools/calibrate_kb.py    fit knockback to measured arc
server/              local 1.21.11 test server (offline mode, superflat)
```

## Observation space

23 → **26 floats per agent**, all things a player can actually see:

- own velocity (world + body frame), height, on-ground, sprint
- own attack cooldown, health, hurt timer
- opponent position in body frame, distance, **signed distance outside reach**
- **aim error**: yaw bearing + pitch error to the opponent's chest
- opponent velocity, on-ground, sprint, cooldown, health
- time since own last hit / last damage taken, arena margin

The two aim-error features are the ones that made training work: they turn
"learn to aim" from a needle-in-a-haystack into a regression problem.

## Action space

Per tick, per agent:

- continuous: yaw delta, pitch delta, forward, strafe (tanh-Gaussian)
- discrete: jump, sprint, attack (Bernoulli)

Yaw/pitch are capped at **20°/tick** (400°/s) — fast enough to flick, slow
enough that the policy cannot spin like a turret.

## Training

```
python -u train.py --iters 1500 --envs 48 --ticks 256        # fresh
python -u train.py --iters 1500 --envs 48 --ticks 256 --resume   # continue
```

- PPO, 128x128 MLP, CPU (this GPU is a GTX 750 / sm_50, useless for this)
- opponents: scripted bot (35%), mirror self-play (35%), league snapshots (30%)
- **aux aim loss**: an extra MSE term regresses the yaw/pitch action mean onto
  perfect aim, annealed out by iteration 150. Only the mean is supervised, so
  the PPO ratio and exploration stay intact.
- approach shaping: potential-based on distance outside reach, so it sums to
  zero over an episode and cannot be farmed
- eval: 48 deterministic duels against the scripted bot
- `runs/ckpt.pt` holds model + optimizer + history + league, saved every 25
  iterations, so shutting the PC off costs at most ~15 seconds of training

### Why `python -u`

When stdout is piped, Python block-buffers it and a training run looks dead for
minutes. Always run with `-u` (or `PYTHONUNBUFFERED=1`).

### Training on Colab (GPU)

The repo carries `runs/kit1/ckpt.pt` (model + optimizer + history), so a fresh
clone can resume immediately. `--device auto` picks cuda on a GPU runtime.

```
# colab cells (Runtime -> Change runtime type -> T4 GPU)
!git clone <your-repo-url> dojo
!pip install -q numpy          # torch is preinstalled with CUDA; do NOT pip install torch
!nvidia-smi -L                # confirm the GPU is visible
!cd dojo && python -u train.py --iters 1000 --envs 32 --ticks 256 \
    --outdir runs/kit1 --resume --reset-pool
```

- `--reset-pool` drops the league snapshots in the checkpoint when they teach
  bad habits (e.g. mutual cowardice); weights and optimizer state are kept.
- checkpoints save every 25 iters; if colab kills the runtime, resume again.
- when done: download `runs/kit1/ppo.pt` (Files panel, ~100KB), copy it over
  local `runs/kit1/ppo.pt`, then `python tools/export_policy.py` for the server.

### Live dashboard on colab

Run the trainer in one cell, then in another:

```
!cd dojo && (pgrep -f "http.server 8000" || nohup python -m http.server 8000 >/dev/null 2>&1 &)
```

```
from google.colab import output
output.serve_kernel_port_as_window(8000, path='/tools/dashboard.html?run=runs/kit1')
```

This opens the dashboard in a new tab (same ember theme, auto-refreshes every
2s) reading `runs/kit1/metrics.jsonl` as training appends to it.
- anti-coward rule lives in `mc_pvp/sim.py`: a timeout with no kill scores
  `R_TIMEOUT = -1.5` for BOTH sides, so fleeing until the clock runs out is
  worse than a fair fight (EV 0). Do not re-add an HP-lead timeout prize.

## Bugs fixed (do not reintroduce these)

1. **Whiff penalty was charged per tick.** Holding attack cost `-0.002 * 1200`
   per episode, which made attacking strictly worse than never attacking — the
   policy converged to 0% attack rate. Attack is now edge-triggered: a tap
   swings, holding auto-swings only when the cooldown is ready.
2. **Seat 0 won every simultaneous exchange.** Health was read inside the
   per-player combat loop, so seat 0's killing blow cancelled seat 1's swing in
   the same tick. Scripted-vs-scripted was 24–0 for seat 0. Alive-ness is now
   snapshotted before either player swings, so mutual kills are draws.
3. **Eval damage was counted after auto-reset**, inflating "damage taken" past
   20 HP (it read every step instead of per-episode totals).

## Open questions (need a human who actually plays)

- [x] sprint-hit knockback — HT3 ground truth: **1.9x** a standing hit
  (`SPRINT_KNOCKBACK_MULT = 1.9` in the sim, was an unvalidated +0.5)
- [x] sprint reset on hit — HT3: **no reset** (`SPRINT_CANCELS_ON_HIT = False`),
  which is why W-tap / S-tap spacing works
3. **Crits** — in 1.9+ sword, is jump-critting actually worth it, or does the
   airtime lose you the trade?
4. **Kit** — currently 20 HP, diamond sword, no armour, no offhand, flat arena.
5. **Arena size** — currently 24x24.

## Roadmap

- [x] tick-accurate sim, calibrated (9/9 checks)
- [x] PPO + self-play league, resumable
- [x] policy that aims and lands hits (was 0 damage before the aim-error obs)
- [ ] policy that beats the scripted bot consistently (>60% wins)
- [ ] full Money SMP kit: armour damage reduction, gapple, splash pots, shield,
      axe disable, webs/deepslate/water, bow
- [ ] richer gamesense rewards (spacing, whiff punish, centre control)
- [ ] terrain (heightmap arena, then the real map)
- [ ] spectator render
- [ ] deploy to HeroBot fake player on the live server

## Current status

| metric | value |
|---|---|
| damage dealt / taken vs scripted | 18.8 / 21.0 per duel |
| time inside reach | 9.8% of ticks |
| aim within 15° | 79% of ticks |
| win rate vs scripted (iter 40) | ~29%, noisy |

It trades evenly but loses on attrition: it aims well, closes, hits, then spends
too long walking back in after knockback. That is the next training target.
