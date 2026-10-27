"""Physics validation against known vanilla Minecraft numbers.

If these drift, PPO will happily learn a game that nobody plays.
Run:  python -m mc_pvp.validate
"""

from __future__ import annotations

import numpy as np

from . import sim


def _env(n=1, seed=1, arena=sim.DEFAULT_ARENA) -> sim.DuelEnv:
    e = sim.DuelEnv(n, seed=seed, arena=arena)
    e.reset()
    e.health[:] = 20.0
    e.hurt_timer[:] = 0
    # raw-physics checks run unbuffed; kit checks set buffs explicitly
    e.str_t[:] = 0
    e.spd_t[:] = 0
    e.fir_t[:] = 0
    return e


def _idle(n=1) -> np.ndarray:
    return np.zeros((n, 2, sim.DuelEnv.ACTION_DIM), np.float32)


def test_jump_apex():
    e = _env()
    act = _idle()
    e.vy[:] = 0.0
    e.on_ground[:] = True
    act[0, 0, sim.DuelEnv.A_JUMP] = 1.0
    e.step(act)
    apex = e.py[0, 0]
    for _ in range(80):
        e.step(_idle())
        apex = max(apex, float(e.py[0, 0]))
    return apex, 1.2522, 0.02


def _ground_speed(sprint: bool):
    e = _env(arena=1e4)          # no walls, so nothing clips terminal velocity
    act = _idle()
    act[:, :, sim.DuelEnv.A_FWD] = 1.0
    if sprint:
        act[:, :, sim.DuelEnv.A_SPRINT] = 1.0
    for _ in range(120):          # spin up to terminal speed
        e.step(act)
    return abs(float(e.vx[0, 0])) * sim.TICK_RATE


def test_walk_speed():
    return _ground_speed(False), 4.317, 0.08


def test_sprint_speed():
    return _ground_speed(True), 5.612, 0.12


def test_terminal_fall():
    """Terminal velocity is the fixed point of v <- (v - g) * drag = -3.92 b/t."""
    e = _env(arena=1e4)
    e.py[:] = 1000.0
    e.vy[:] = -3.92
    e.on_ground[:] = False
    for _ in range(20):
        e.step(_idle())
    return float(e.vy[0, 0]), -3.92, 0.02


def _setup_duel(dist: float, sprint: bool = False, charge_ticks: float | None = None):
    e = _env()
    e.px[0, 0], e.pz[0, 0] = -dist / 2, 0.0
    e.px[0, 1], e.pz[0, 1] = dist / 2, 0.0
    e.yaw[0, 0] = 270.0        # p0 faces +X
    e.yaw[0, 1] = 90.0         # p1 faces -X
    e.pitch[0, 0] = np.degrees(np.arctan2(sim.EYE_HEIGHT - 0.9, dist))
    e.cooldown[:] = sim.COOLDOWN_TICKS if charge_ticks is None else charge_ticks
    e.on_ground[:] = True
    e.vy[:] = 0.0
    act = _idle()
    if sprint:
        act[0, 0, sim.DuelEnv.A_SPRINT] = 1.0
        act[0, 0, sim.DuelEnv.A_FWD] = 1.0
    return e, act


def test_full_charge_damage():
    e, act = _setup_duel(2.0)
    act[0, 0, sim.DuelEnv.A_ATTACK] = 1.0
    e.step(act)
    expected = float(sim.damage_after_armor(
        np.float32(sim.SWORD_DAMAGE + sim.SHARPNESS_BONUS)))
    return 20.0 - float(e.health[0, 1]), expected, 0.01


def test_weak_hit_damage():
    # step() ticks the cooldown once before combat resolves
    e, act = _setup_duel(2.0, charge_ticks=sim.COOLDOWN_TICKS * 0.5 - 1)
    act[0, 0, sim.DuelEnv.A_ATTACK] = 1.0
    e.step(act)
    raw = (sim.SWORD_DAMAGE + sim.SHARPNESS_BONUS) * (0.2 + 0.8 * 0.25)
    expected = float(sim.damage_after_armor(np.float32(raw)))
    return 20.0 - float(e.health[0, 1]), expected, 0.01


def test_hurt_immunity():
    e, act = _setup_duel(2.0)
    act[0, 0, sim.DuelEnv.A_ATTACK] = 1.0
    e.step(act)
    first = 20.0 - float(e.health[0, 1])
    e.cooldown[:] = sim.COOLDOWN_TICKS
    e.fire_t[:] = 0                   # Fire Aspect burn ticks through i-frames; not under test
    e.tick = 1                        # avoid the %20 burn tick landing on the probe step
    e.step(act)                       # immediate second swing, still immune
    total = 20.0 - float(e.health[0, 1])
    return total, first, 0.01


def test_reach_edge():
    """Vanilla reach is measured to the hitbox surface (0.3 half-width)."""
    out = {}
    for d in (3.25, 3.6):
        e, act = _setup_duel(d)
        act[0, 0, sim.DuelEnv.A_ATTACK] = 1.0
        e.step(act)
        out[d] = 20.0 - float(e.health[0, 1]) > 0
    return out


def _kit_duel(dist: float = 2.0) -> "sim.DuelEnv":
    """Two fighters facing each other, unbuffed, weapons ready."""
    e = _env()
    e.px[0, 0], e.pz[0, 0] = -dist / 2, 0.0
    e.px[0, 1], e.pz[0, 1] = dist / 2, 0.0
    e.yaw[0, 0] = 270.0
    e.yaw[0, 1] = 90.0
    e.pitch[:] = 0.0
    e.cooldown[:] = sim.COOLDOWN_TICKS
    e.axe_cd[:] = sim.AXE_COOLDOWN_TICKS
    return e


def _act0(**kw) -> np.ndarray:
    a = np.zeros((1, 2, sim.DuelEnv.ACTION_DIM), np.float32)
    for k, v in kw.items():
        a[0, 0, getattr(sim.DuelEnv, k)] = v
    return a


def _act1(**kw) -> np.ndarray:
    a = np.zeros((1, 2, sim.DuelEnv.ACTION_DIM), np.float32)
    for k, v in kw.items():
        a[0, 1, getattr(sim.DuelEnv, k)] = v
    return a


def test_shield_blocks():
    e = _kit_duel()
    for _ in range(6):
        e.step(_act1(A_BLOCK=1))
    a = _act0(A_ATTACK=1)
    a[0, 1, sim.DuelEnv.A_BLOCK] = 1.0
    e.step(a)
    return 20.0 - float(e.health[0, 1]), 0.0, 0.01


def test_shield_disable_stun():
    e = _kit_duel()
    for _ in range(6):
        e.step(_act1(A_BLOCK=1))
    # sword sprint-hit must NOT disable (vanilla: axe only)
    a = _act0(A_ATTACK=1, A_SPRINT=1, A_FWD=1)
    a[0, 1, sim.DuelEnv.A_BLOCK] = 1.0
    e.step(a)
    sword_off = int(e.shield_off[0, 1])
    # axe sprint-hit MUST disable with the stun (no KB)
    e2 = _kit_duel()
    e2.has_axe[:] = True
    for _ in range(6):
        e2.step(_act1(A_BLOCK=1))
    a2 = _act0(A_ATTACK=1, A_SPRINT=1, A_FWD=1)
    a2[0, 0, sim.DuelEnv.A_SWAP] = 0.0  # already holding axe
    a2[0, 1, sim.DuelEnv.A_BLOCK] = 1.0
    vx_before = float(e2.vx[0, 1])
    e2.step(a2)
    dmg = 20.0 - float(e2.health[0, 1])
    kb = abs(float(e2.vx[0, 1]) - vx_before * 0.5)
    off = int(e2.shield_off[0, 1])
    ok = (sword_off == 0) and (dmg > 0.5 and off == sim.SHIELD_DISABLE_TICKS and kb < 0.01)
    return ok


def test_strength_bonus():
    e = _kit_duel()
    e.str_t[0, 0] = 100
    e.step(_act0(A_ATTACK=1))
    raw = (sim.SWORD_DAMAGE + sim.SHARPNESS_BONUS + sim.STR_BONUS)
    import numpy as _np
    expected = float(sim.damage_after_armor(_np.float32(raw)))
    return 20.0 - float(e.health[0, 1]), expected, 0.02


def test_splash_heal():
    e = _kit_duel()
    e.health[0, 0] = 10.0
    e.step(_act0(A_SPL_HEAL=1))
    got = (float(e.health[0, 0]), int(e.n_heal[0, 0]))
    return got == (18.0, sim.N_HEAL_POTS - 1)


def test_gapple_eat():
    e = _kit_duel()
    e.saturation[0, 0] = 5.0
    e.step(_act0(A_EAT=1))
    started = int(e.eating[0, 0]) == sim.EAT_TICKS - 1
    for _ in range(sim.EAT_TICKS - 1):
        e.step(np.zeros((1, 2, sim.DuelEnv.ACTION_DIM), np.float32))
    full = abs(float(e.saturation[0, 0]) - 20.0) < 0.01
    return started and full


def test_weave_web_speed():
    """Weaving doubles web speed: 50% vs vanilla 25%."""
    def trial(weave: bool):
        e = _env()
        e.px[0, 0] = -5.0
        e.px[0, 1] = 5.0
        e.pz[:] = 0.0
        e.vx[:] = 0.0
        e.vz[:] = 0.0
        e.web_x[0, 0] = -5.0
        e.web_z[0, 0] = 0.0
        e.web_x[0, 1] = 5.0
        e.web_z[0, 1] = 0.0
        e.weave_t[:] = sim.WEAVE_DURATION if weave else 0
        a = np.zeros((1, 2, sim.DuelEnv.ACTION_DIM), np.float32)
        a[:, :, sim.DuelEnv.A_FWD] = 1.0
        a[:, :, sim.DuelEnv.A_SPRINT] = 1.0
        e.step(a)
        return abs(float(e.vx[0, 0])), bool(e.in_web[0, 0])
    v_w, in_w = trial(True)
    v_b, in_b = trial(False)
    ratio = v_w / max(v_b, 1e-9)
    return in_w and in_b and abs(ratio - 2.0) < 0.15


def test_shared_cooldown():
    """Any swing resets BOTH cooldowns: sword->axe flapping buys nothing."""
    e = _kit_duel()
    e.cooldown[:] = sim.COOLDOWN_TICKS
    e.axe_cd[:] = sim.AXE_COOLDOWN_TICKS
    e.step(_act0(A_ATTACK=1))
    return float(e.cooldown[0, 0]) == 0.0 and float(e.axe_cd[0, 0]) == 0.0


def _kb_distance(sprint: bool) -> float:
    e, act = _setup_duel(2.0, sprint=sprint)
    act[0, 0, sim.DuelEnv.A_ATTACK] = 1.0
    e.step(act)
    # let go and coast
    coast = _idle()
    x0 = float(e.px[0, 1])
    for _ in range(120):
        e.step(coast)
        if abs(float(e.vx[0, 1])) < 0.005 and e.on_ground[0, 1]:
            break
    return float(e.px[0, 1]) - x0


def test_knockback():
    walk = _kb_distance(False)
    sprint = _kb_distance(True)
    return walk, sprint


def main():
    rows = []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        rows.append((ok, name, f"{got:8.3f}", f"{want:8.3f}"))

    check("jump apex (blocks)", *test_jump_apex())
    check("walk speed (b/s)", *test_walk_speed())
    check("sprint speed (b/s)", *test_sprint_speed())
    check("terminal fall (b/t)", *test_terminal_fall())
    check("full charge damage", *test_full_charge_damage())
    check("half charge damage", *test_weak_hit_damage())
    check("hurt immunity (10t)", *test_hurt_immunity())

    reach = test_reach_edge()
    ok = reach[3.25] and not reach[3.6]
    rows.append((ok, "reach edge 3.25 hit / 3.6 miss", str(reach), ""))

    walk, sprint = test_knockback()
    rows.append((sprint > walk, "sprint KB > walk KB", f"{walk:5.2f} / {sprint:5.2f}", ""))

    check("shield blocks frontal", *test_shield_blocks())
    rows.append((test_shield_disable_stun(), "axe disables + stun (sword does NOT)", "", ""))
    rows.append((test_weave_web_speed(), "weave 50% vs base 25% in web", "", ""))
    rows.append((test_shared_cooldown(), "shared attack lockout (no swap spam)", "", ""))
    check("strength II +6 raw", *test_strength_bonus())
    rows.append((test_splash_heal(), "splash Healing II +8", "", ""))
    rows.append((test_gapple_eat(), "gapple 32t -> sat 20", "", ""))

    print(f"{'':3} {'check':34} {'sim':>10} {'vanilla':>10}")
    print("-" * 62)
    for ok, name, got, want in rows:
        print(f"{'ok' if ok else 'XX':3} {name:34} {got:>10} {want:>10}")
    failed = sum(1 for r in rows if not r[0])
    print("-" * 62)
    print(f"{len(rows) - failed}/{len(rows)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
