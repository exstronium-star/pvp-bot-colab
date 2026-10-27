"""Scripted sparring partner.

Not meant to be strong - meant to be *legible*. It holds max reach, strafes on
a rhythm, attacks on cooldown, and occasionally jumps for a crit. That gives
PPO something with a real punish window to exploit, and gives us a baseline
number for "is the agent improving".
"""

from __future__ import annotations

import numpy as np

from . import sim


class ScriptedBot:
    """Drives both seats of a DuelEnv. Stateless w.r.t. learning."""

    def __init__(self, n_envs: int, rng: np.random.Generator):
        self.n = n_envs
        self.rng = rng
        self.strafe = rng.choice([-1.0, 1.0], size=(n_envs, 2))
        self.timer = rng.integers(8, 26, size=(n_envs, 2))

    def act(self, env: sim.DuelEnv) -> np.ndarray:
        n = self.n
        act = np.zeros((n, 2, sim.DuelEnv.ACTION_DIM), np.float32)

        self.timer -= 1
        flip = self.timer <= 0
        if flip.any():
            self.strafe[flip] *= -1.0
            self.timer[flip] = self.rng.integers(8, 26, size=int(flip.sum()))

        for p in (0, 1):
            o = 1 - p
            dx = env.px[:, o] - env.px[:, p]
            dz = env.pz[:, o] - env.pz[:, p]
            dist = np.maximum(np.hypot(dx, dz), 1e-6)
            dy = (env.py[:, o] + 1.2) - (env.py[:, p] + sim.EYE_HEIGHT)

            # --- aim -------------------------------------------------------
            want_yaw = np.degrees(np.arctan2(-dx, dz))
            flat = np.hypot(dx, dz)
            want_pitch = np.degrees(-np.arctan2(dy, np.maximum(flat, 1e-6)))

            yaw_err = ((want_yaw - env.yaw[:, p] + 180.0) % 360.0) - 180.0
            pitch_err = want_pitch - env.pitch[:, p]
            act[:, p, sim.DuelEnv.A_YAW] = np.clip(yaw_err, -28.0, 28.0)
            act[:, p, sim.DuelEnv.A_PITCH] = np.clip(pitch_err, -28.0, 28.0)

            # --- footwork --------------------------------------------------
            too_far = dist > 3.05
            too_close = dist < 2.4
            act[:, p, sim.DuelEnv.A_FWD] = np.where(too_far, 1.0, np.where(too_close, -1.0, 0.0))
            act[:, p, sim.DuelEnv.A_STRAFE] = np.where(too_far | too_close, 0.0, self.strafe[:, p])
            # always sprint when closing or close (user callout): walk is for
            # taps only. Sim crits need no-sprint, but the dummy hops to crit.
            act[:, p, sim.DuelEnv.A_SPRINT] = (dist > 2.2).astype(np.float32)

            # --- crit hops ------------------------------------------------
            hop = (dist < 3.0) & env.on_ground[:, p] & (env.cooldown[:, p] > 6.0) & (self.rng.random(n) < 0.06)
            act[:, p, sim.DuelEnv.A_JUMP] = hop.astype(np.float32)

            # --- weapon: predict shield, 1-tick swap sword->axe, disable, back.
            # HT3: sword is home, axe comes out ONLY for the disable attempt.
            # No flapping: edge-triggered, so hold-one-tick then back one tick.
            # NOTE: sim _can_swing needs 95% charge, so the dummy only swings
            # full-charge — weak-chip is never demonstrated as correct.
            opp_turtle = env.blocking[:, o] & (dist < 4.0)
            want_axe = opp_turtle & ~env.has_axe[:, p] & (env.shield_off[:, o] == 0)
            act[:, p, sim.DuelEnv.A_SWAP] = (want_axe & (self.rng.random(n) < 0.5)).astype(np.float32)

            # --- shield: cover while the sword recharges, swing while ready ---
            # (Legible rhythm: the dummy alternates open/covered, so both
            # turtling and whiff-punishing are answered. It never attacks
            # behind its own shield: the sim suppresses that, like the bridge.)
            recharging = env.cooldown[:, p] < sim.COOLDOWN_TICKS * 0.6
            shield_up = (dist < 3.5) & (env.shield_off[:, p] == 0) & recharging
            swing_ok = _can_swing(env, p, dist, yaw_err)
            attack = swing_ok & ~shield_up
            act[:, p, sim.DuelEnv.A_ATTACK] = attack.astype(np.float32)
            act[:, p, sim.DuelEnv.A_BLOCK] = (shield_up & ~attack).astype(np.float32)
            # swap back to sword after the axe lands its disable
            back_to_sword = env.has_axe[:, p] & (env.shield_off[:, o] > 0)
            act[:, p, sim.DuelEnv.A_SWAP] = np.where(
                back_to_sword, 1.0, act[:, p, sim.DuelEnv.A_SWAP])

            # --- eat: distance first (HT3 rule), gapple < saturation, ench ~10hp
            far = dist > 6.0
            hungry = env.saturation[:, p] < 14.0
            low = env.health[:, p] < 11.0
            act[:, p, sim.DuelEnv.A_EAT] = (
                far & (hungry | low) & (env.eating[:, p] == 0)).astype(np.float32)

            # --- splash heal when low and far, Str repot when expired ---------
            act[:, p, sim.DuelEnv.A_SPL_HEAL] = (
                far & (env.health[:, p] < 12.0) & (env.n_heal[:, p] > 0)).astype(np.float32)
            act[:, p, sim.DuelEnv.A_SPL_STR] = (
                (env.str_t[:, p] == 0) & (env.n_str[:, p] > 0)).astype(np.float32)

        return act


# ---------------------------------------------------------------------------
# HT3 style bots (agents.md Q&A, encoded as opponents).
#
# House rules every style obeys (so skills learned against them transfer to
# the server bridge, which enforces the same): never assert ATTACK and BLOCK
# on the same tick (the sim suppresses swings behind a raised shield, exactly
# like BotController's `!shieldUp` attack gate), and pulse SWAP for exactly
# one tick per decision (the sim edge-triggers the toggle, like a keypress).
# ---------------------------------------------------------------------------

def _aim(env: sim.DuelEnv, p: int, act: np.ndarray):
    """Chest aim for seat p. Returns (dist, yaw_err)."""
    o = 1 - p
    dx = env.px[:, o] - env.px[:, p]
    dz = env.pz[:, o] - env.pz[:, p]
    dist = np.maximum(np.hypot(dx, dz), 1e-6)
    dy = (env.py[:, o] + 1.2) - (env.py[:, p] + sim.EYE_HEIGHT)
    want_yaw = np.degrees(np.arctan2(-dx, dz))
    want_pitch = np.degrees(-np.arctan2(dy, np.maximum(np.hypot(dx, dz), 1e-6)))
    yaw_err = ((want_yaw - env.yaw[:, p] + 180.0) % 360.0) - 180.0
    pitch_err = want_pitch - env.pitch[:, p]
    act[:, p, sim.DuelEnv.A_YAW] = np.clip(yaw_err, -28.0, 28.0)
    act[:, p, sim.DuelEnv.A_PITCH] = np.clip(pitch_err, -28.0, 28.0)
    return dist, yaw_err


def _can_swing(env: sim.DuelEnv, p: int, dist: np.ndarray, yaw_err: np.ndarray):
    """Full-charge, aimed, in-reach swing with free hands."""
    o = 1 - p
    axe_out = env.has_axe[:, p]
    ready = np.where(axe_out,
                     env.axe_cd[:, p] >= sim.AXE_COOLDOWN_TICKS * 0.95,
                     env.cooldown[:, p] >= sim.COOLDOWN_TICKS * 0.95)
    return ((dist <= sim.REACH - 0.05) & ready & (np.abs(yaw_err) < 10.0)
            & (env.hurt_timer[:, o] == 0) & (env.eating[:, p] == 0))


def _eat_drink_repot(env: sim.DuelEnv, p: int, act: np.ndarray, far: np.ndarray):
    """Shared survival layer: gapple when far, splash when low, repot instantly."""
    hungry = env.saturation[:, p] < 14.0
    low = env.health[:, p] < 11.0
    act[:, p, sim.DuelEnv.A_EAT] = (
        far & (hungry | low) & (env.eating[:, p] == 0)
        & ((env.n_gap[:, p] > 0) | (env.n_egap[:, p] > 0))).astype(np.float32)
    act[:, p, sim.DuelEnv.A_SPL_HEAL] = (
        (env.health[:, p] < 12.0) & (env.n_heal[:, p] > 0)).astype(np.float32)
    # repot INSTANTLY on expiry (Q11): window is ticks, not seconds
    act[:, p, sim.DuelEnv.A_SPL_STR] = (
        (env.str_t[:, p] <= 2) & (env.n_str[:, p] > 0)).astype(np.float32)
    # weaving: throw at own feet while stuck in a web (HT3: 50% speed)
    in_web = env.in_web[:, p] if hasattr(env, 'in_web') else np.zeros_like(low)
    n_weave = env.n_weave[:, p] if hasattr(env, 'n_weave') else np.zeros_like(low, dtype=np.int32)
    act[:, p, sim.DuelEnv.A_WEAVE] = (
        in_web & (n_weave > 0)).astype(np.float32)


class TurtleBot:
    """Shield-spammer (agents.md Q12/Q13).

    Holds the shield whenever the enemy is in range, only dropping it to
    punish (enemy just swung) or to sprint-tap a fellow turtle's shield with
    the AXE (only the axe disables now). Goes passive and backs off while its
    own shield is disabled. Never attacks behind its own shield.
    """

    PUNISH_T = 10  # ticks the punish window stays open after an enemy swing

    def __init__(self, n_envs: int, rng: np.random.Generator):
        self.n = n_envs
        self.rng = rng
        self.strafe = rng.choice([-1.0, 1.0], size=(n_envs, 2))
        self.timer = rng.integers(8, 26, size=(n_envs, 2))
        self.punish = np.zeros((n_envs, 2), np.int32)
        self.prev_ecd = np.full((n_envs, 2), sim.COOLDOWN_TICKS, np.float32)
        self.prev_n_heal = np.full((n_envs, 2), sim.N_HEAL_POTS, np.float32)
        self.spl_wait = np.zeros((n_envs, 2), np.int32)

    def act(self, env: sim.DuelEnv) -> np.ndarray:
        n = self.n
        act = np.zeros((n, 2, sim.DuelEnv.ACTION_DIM), np.float32)
        self.timer -= 1
        flip = self.timer <= 0
        if flip.any():
            self.strafe[flip] *= -1.0
            self.timer[flip] = self.rng.integers(8, 26, size=int(flip.sum()))

        for p in (0, 1):
            o = 1 - p
            dist, yaw_err = _aim(env, p, act)
            aimed = np.abs(yaw_err) < 10.0
            passive = env.shield_off[:, p] > 0  # Q13: disabled -> play passive
            self.punish[:, p] = np.maximum(self.punish[:, p] - 1, 0)
            swung = (self.prev_ecd[:, p] > 4.0) & (env.cooldown[:, o] < 2.0)
            self.punish[:, p] = np.where(swung & (dist < 4.0),
                                          self.PUNISH_T, self.punish[:, p])
            self.prev_ecd[:, p] = env.cooldown[:, o]
            swing_ok = _can_swing(env, p, dist, yaw_err)

            # fellow turtle in range: 1-tick swap to axe, sprint-tap the shield
            # (ONLY the axe disables). Otherwise hold and punish their swing.
            # An open, passive enemy is a free hit: take it with the sword.
            dis_op = (env.blocking[:, o] & (dist < 3.0) & aimed
                      & (env.eating[:, p] == 0) & ~passive)
            free_hit = (~env.blocking[:, o] & (dist < 3.0) & aimed
                        & (env.eating[:, p] == 0) & ~passive
                        & (self.punish[:, p] == 0))
            attack = ((self.punish[:, p] > 0) | dis_op | free_hit) & swing_ok & ~passive
            # axe out ONLY for the disable tap, sword for everything else.
            # 1-tick pulses (edge-triggered): predict, don't flap.
            want_axe = dis_op & ~env.has_axe[:, p]
            want_sword = (env.has_axe[:, p]
                          & ((env.shield_off[:, o] > 0) | ~dis_op))
            swap = (((want_axe | want_sword)) & (self.rng.random(n) < 0.5)).astype(np.float32)
            act[:, p, sim.DuelEnv.A_SWAP] = swap
            block = ((dist < 3.2) & (env.shield_off[:, p] == 0)
                     & (self.punish[:, p] == 0) & (env.eating[:, p] == 0)
                     & ~passive & ~attack)
            # footwork: hold ~2.8 blocks, back right off while passive.
            too_far = dist > 3.05
            too_close = dist < 2.4
            fwd = np.where(too_far, 1.0, np.where(too_close, -1.0, 0.0))
            strafe = np.where(too_far | too_close, 0.0, self.strafe[:, p])
            fwd = np.where(passive & (dist < 4.5), -1.0,
                           np.where(passive, 0.0, fwd))
            act[:, p, sim.DuelEnv.A_FWD] = fwd
            act[:, p, sim.DuelEnv.A_STRAFE] = np.where(passive, strafe, strafe)
            sprint = dis_op & (fwd > 0.1)  # sprint only for the disable tap
            act[:, p, sim.DuelEnv.A_SPRINT] = sprint.astype(np.float32)
            act[:, p, sim.DuelEnv.A_ATTACK] = attack.astype(np.float32)
            act[:, p, sim.DuelEnv.A_BLOCK] = block.astype(np.float32)

            _eat_drink_repot(env, p, act, dist > 6.0)
            # one splash per 50 ticks max: the sim has no throw cooldown and a
            # held bit would chug the whole stack in one second.
            self.spl_wait[:, p] = np.maximum(self.spl_wait[:, p] - 1, 0)
            used = env.n_heal[:, p] < self.prev_n_heal[:, p]
            self.spl_wait[:, p] = np.where(used, 50, self.spl_wait[:, p])
            self.prev_n_heal[:, p] = env.n_heal[:, p]
            act[:, p, sim.DuelEnv.A_SPL_HEAL] *= (self.spl_wait[:, p] == 0)

        return act


class RunnerBot:
    """Kiter (agents.md Q6/Q7/Q9).

    Holds 4.5-6.5 blocks, baits whiffs, and answers them with a sprint-in
    jump crit (sprint cut the moment it leaves the ground: sim crits need
    falling + airborne + NOT sprinting). Never turtles; eats on the move
    when far, splashes when too close to eat safely.
    """

    PUNISH_T = 22  # whiff punish window: enough to close 3 blocks + jump

    def __init__(self, n_envs: int, rng: np.random.Generator):
        self.n = n_envs
        self.rng = rng
        self.strafe = rng.choice([-1.0, 1.0], size=(n_envs, 2))
        self.timer = rng.integers(8, 26, size=(n_envs, 2))
        self.punish = np.zeros((n_envs, 2), np.int32)
        self.prev_ecd = np.full((n_envs, 2), sim.COOLDOWN_TICKS, np.float32)
        self.prev_n_heal = np.full((n_envs, 2), sim.N_HEAL_POTS, np.float32)
        self.spl_wait = np.zeros((n_envs, 2), np.int32)

    def act(self, env: sim.DuelEnv) -> np.ndarray:
        n = self.n
        act = np.zeros((n, 2, sim.DuelEnv.ACTION_DIM), np.float32)
        self.timer -= 1
        flip = self.timer <= 0
        if flip.any():
            self.strafe[flip] *= -1.0
            self.timer[flip] = self.rng.integers(8, 26, size=int(flip.sum()))

        for p in (0, 1):
            o = 1 - p
            dist, yaw_err = _aim(env, p, act)
            self.punish[:, p] = np.maximum(self.punish[:, p] - 1, 0)
            # enemy swung from out of range -> whiff -> punish window (Q6).
            swung = (self.prev_ecd[:, p] > 4.0) & (env.cooldown[:, o] < 2.0)
            self.punish[:, p] = np.where(swung & (dist > sim.REACH + 0.2),
                                          self.PUNISH_T, self.punish[:, p])
            self.prev_ecd[:, p] = env.cooldown[:, o]
            punishing = self.punish[:, p] > 0
            grounded = env.on_ground[:, p]
            falling = (env.vy[:, p] < 0) & ~grounded
            readyish = env.cooldown[:, p] > 6.0

            fwd = np.where(dist > 6.5, 1.0,
                           np.where(dist < 4.5, -1.0, 0.0))
            fwd = np.where(punishing, 1.0, fwd)
            strafe = np.where((dist >= 4.5) & (dist <= 6.5) & ~punishing,
                              self.strafe[:, p], 0.0)
            # sprint to close, never in the air (kills crit eligibility).
            sprint = (fwd > 0.1) & grounded & (punishing | (dist > 3.0))
            jump = punishing & grounded & (dist < 3.6) & readyish
            swing_ok = _can_swing(env, p, dist, yaw_err)
            # crack a turtle: 1-tick swap to axe, sprint-tap the live shield
            # (ONLY the axe disables now). Sword taps never break a turtle.
            crack = (env.blocking[:, o] & (dist < 3.4) & grounded
                     & (np.abs(yaw_err) < 10.0))
            sprint = sprint | (crack & ~punishing)
            fwd = np.where(crack & ~punishing, 1.0, fwd)
            want_axe_c = crack & ~punishing & ~env.has_axe[:, p]
            want_sword_c = env.has_axe[:, p] & (~crack | punishing)
            crack_swap = ((want_axe_c | want_sword_c)
                          & (self.rng.random(n) < 0.5)).astype(np.float32)
            act[:, p, sim.DuelEnv.A_SWAP] = crack_swap
            attack = (punishing & swing_ok & (falling | (env.shield_off[:, o] > 0))
                      | crack & swing_ok & ~punishing)
            act[:, p, sim.DuelEnv.A_FWD] = fwd
            act[:, p, sim.DuelEnv.A_STRAFE] = strafe
            act[:, p, sim.DuelEnv.A_SPRINT] = sprint.astype(np.float32)
            act[:, p, sim.DuelEnv.A_JUMP] = jump.astype(np.float32)
            act[:, p, sim.DuelEnv.A_ATTACK] = attack.astype(np.float32)
            # no shield: spacing is the defense.
            _eat_drink_repot(env, p, act, dist > 6.0)
            self.spl_wait[:, p] = np.maximum(self.spl_wait[:, p] - 1, 0)
            used = env.n_heal[:, p] < self.prev_n_heal[:, p]
            self.spl_wait[:, p] = np.where(used, 50, self.spl_wait[:, p])
            self.prev_n_heal[:, p] = env.n_heal[:, p]
            act[:, p, sim.DuelEnv.A_SPL_HEAL] *= (self.spl_wait[:, p] == 0)

        return act


class AggroBot:
    """Aggressive W-tapper (agents.md Q1-Q5, Q7).

    Axe opener with a sprint-in jump crit, W-tap by default after landing
    (release forward 4 ticks), S-tap when still in their reach (back 5
    ticks), combos only past a disabled shield, sword into a live shield is
    skipped (axe disable attempts excepted). Never shields right after
    taking a sprint hit (Q12); backs off to eat at ~10 HP unless winning,
    in which case it pushes the quickdrop instead of eating (Q-C).
    """

    def __init__(self, n_envs: int, rng: np.random.Generator):
        self.n = n_envs
        self.rng = rng
        self.strafe = rng.choice([-1.0, 1.0], size=(n_envs, 2))
        self.timer = rng.integers(8, 26, size=(n_envs, 2))
        self.wtap = np.zeros((n_envs, 2), np.int32)
        self.stap = np.zeros((n_envs, 2), np.int32)
        self.reset = np.zeros((n_envs, 2), np.int32)
        self.hurt_t = np.full((n_envs, 2), 999, np.int32)
        self.prev_ehp = np.full((n_envs, 2), 20.0, np.float32)
        self.prev_hp = np.full((n_envs, 2), 20.0, np.float32)
        self.prev_want_axe = np.zeros((n_envs, 2), bool)
        self.prev_want_sword = np.zeros((n_envs, 2), bool)
        self.prev_n_heal = np.full((n_envs, 2), sim.N_HEAL_POTS, np.float32)
        self.spl_wait = np.zeros((n_envs, 2), np.int32)

    def act(self, env: sim.DuelEnv) -> np.ndarray:
        n = self.n
        act = np.zeros((n, 2, sim.DuelEnv.ACTION_DIM), np.float32)
        self.timer -= 1
        flip = self.timer <= 0
        if flip.any():
            self.strafe[flip] *= -1.0
            self.timer[flip] = self.rng.integers(8, 26, size=int(flip.sum()))

        for p in (0, 1):
            o = 1 - p
            dist, yaw_err = _aim(env, p, act)
            aimed = np.abs(yaw_err) < 10.0
            my_hp = env.health[:, p]
            en_hp = env.health[:, o]
            # --- hit bookkeeping -------------------------------------
            landed = en_hp < self.prev_ehp[:, p] - 0.01
            self.prev_ehp[:, p] = en_hp.copy()
            drop = self.prev_hp[:, p] - my_hp
            took = drop > 0.01
            took_crit = took & (drop >= 4.0) & ~env.on_ground[:, o]
            self.prev_hp[:, p] = my_hp.copy()
            self.hurt_t[:, p] = np.where(took, 0,
                                          np.minimum(self.hurt_t[:, p] + 1, 999))
            # W-tap default, S-tap when still in their reach (Q3/F).
            self.wtap[:, p] = np.maximum(self.wtap[:, p] - 1, 0)
            self.stap[:, p] = np.maximum(self.stap[:, p] - 1, 0)
            self.reset[:, p] = np.maximum(self.reset[:, p] - 1, 0)
            fresh = landed & (self.wtap[:, p] == 0) & (self.stap[:, p] == 0)
            self.stap[:, p] = np.where(fresh & (dist < 2.2), 5, self.stap[:, p])
            self.wtap[:, p] = np.where(fresh & (dist >= 2.2), 4, self.wtap[:, p])
            # Q4: never answer damage with a sprint hit; reset first.
            self.reset[:, p] = np.where(took_crit, 8, self.reset[:, p])
            self.reset[:, p] = np.where(took & (self.reset[:, p] == 0),
                                        np.maximum(self.reset[:, p], 4),
                                        self.reset[:, p])
            tapping = (self.wtap[:, p] > 0) | (self.stap[:, p] > 0)
            resetting = self.reset[:, p] > 0
            # --- weapon: axe while their shield is up, sword once it pops -
            has_axe = env.has_axe[:, p]
            want_axe = ~has_axe & (env.shield_off[:, o] == 0) & (dist < 12.0)
            want_sword = has_axe & ((env.shield_off[:, o] > 0) | (dist > 12.0))
            swap = ((want_axe & ~self.prev_want_axe[:, p])
                    | (want_sword & ~self.prev_want_sword[:, p]))
            self.prev_want_axe[:, p] = want_axe
            self.prev_want_sword[:, p] = want_sword
            # --- footwork --------------------------------------------
            low_hp = my_hp <= 10.0
            winning = en_hp + 6.0 <= my_hp  # Q-C: push the quickdrop
            disengage = low_hp & (dist < 6.0) & ~winning  # Q7: go eat
            fwd = np.where(disengage | resetting, -1.0, 1.0)
            fwd = np.where(self.stap[:, p] > 0, -1.0, fwd)
            fwd = np.where(self.wtap[:, p] > 0, 0.0, fwd)
            grounded = env.on_ground[:, p]
            # sprint to close, never airborne (crit needs ~sprinting), never
            # while tapping/resetting (that IS the W/S-tap).
            sprint = (dist > 2.6) & grounded & ~tapping & ~resetting
            # grounded axe slam vs a live shield (sprint = disable); jump
            # crit vs everything else.
            charge = has_axe & env.blocking[:, o] & (dist < 3.5) & grounded
            sprint = sprint | (charge & ~tapping & ~resetting)
            fwd = np.where(charge, 1.0, fwd)
            jump = (~charge & grounded & (dist < 3.6) & (env.cooldown[:, p] > 6.0)
                    & ~tapping & ~resetting & ~disengage)
            # --- hands: combo only past a disabled shield (Q3) -------------
            swing_ok = _can_swing(env, p, dist, yaw_err)
            trade_ok = ((env.saturation[:, p] >= 18.0) | winning
                        | (env.shield_off[:, o] > 0))  # Q2 golden rule
            sword_worth = trade_ok & ~(env.blocking[:, o] & ~has_axe)
            attack = (swing_ok & (has_axe | sword_worth) & aimed
                      & ~tapping & ~resetting & ~disengage
                      & (env.eating[:, p] == 0))
            # Q12: shield only inside 3 blocks, never right after taking a
            # sprint hit, never on the same tick as a swing.
            just_sprinted = (self.hurt_t[:, p] < 8) & env.sprinting[:, o]
            block = ((dist < 3.0) & (env.shield_off[:, p] == 0)
                     & ~just_sprinted & (env.eating[:, p] == 0)
                     & ~tapping & ~resetting & ~attack)
            act[:, p, sim.DuelEnv.A_FWD] = fwd
            act[:, p, sim.DuelEnv.A_STRAFE] = 0.0
            act[:, p, sim.DuelEnv.A_SPRINT] = sprint.astype(np.float32)
            act[:, p, sim.DuelEnv.A_JUMP] = jump.astype(np.float32)
            act[:, p, sim.DuelEnv.A_SWAP] = swap.astype(np.float32)
            act[:, p, sim.DuelEnv.A_ATTACK] = attack.astype(np.float32)
            act[:, p, sim.DuelEnv.A_BLOCK] = block.astype(np.float32)

            _eat_drink_repot(env, p, act, dist > 6.0)
            self.spl_wait[:, p] = np.maximum(self.spl_wait[:, p] - 1, 0)
            used = env.n_heal[:, p] < self.prev_n_heal[:, p]
            self.spl_wait[:, p] = np.where(used, 50, self.spl_wait[:, p])
            self.prev_n_heal[:, p] = env.n_heal[:, p]
            act[:, p, sim.DuelEnv.A_SPL_HEAL] *= (self.spl_wait[:, p] == 0)

        return act
