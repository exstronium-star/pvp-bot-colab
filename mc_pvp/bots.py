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
            act[:, p, sim.DuelEnv.A_SPRINT] = (dist > 3.4).astype(np.float32)

            # --- crit hops ------------------------------------------------
            hop = (dist < 3.0) & env.on_ground[:, p] & (env.cooldown[:, p] > 6.0) & (self.rng.random(n) < 0.06)
            act[:, p, sim.DuelEnv.A_JUMP] = hop.astype(np.float32)

            # --- weapon: axe opener for disable/crit, sword otherwise --------
            # HT3: lead with the axe (disable or crit), then sword.
            want_axe = (dist < 4.0) & (env.shield_off[:, o] == 0) & ~env.has_axe[:, p]
            act[:, p, sim.DuelEnv.A_SWAP] = (want_axe & (self.rng.random(n) < 0.3)).astype(np.float32)

            # --- shield: raise at close range, never while eating ------------
            shield_up = (dist < 3.5) & (env.shield_off[:, p] == 0)
            act[:, p, sim.DuelEnv.A_BLOCK] = shield_up.astype(np.float32)

            # --- swing ----------------------------------------------------
            aimed = np.abs(yaw_err) < 10.0
            axe_out = env.has_axe[:, p]
            ready = np.where(axe_out,
                             env.axe_cd[:, p] >= sim.AXE_COOLDOWN_TICKS * 0.95,
                             env.cooldown[:, p] >= sim.COOLDOWN_TICKS * 0.95)
            can_swing = (dist <= sim.REACH - 0.05) & ready & aimed & (env.hurt_timer[:, o] == 0)
            can_swing = can_swing & (env.eating[:, p] == 0)
            act[:, p, sim.DuelEnv.A_ATTACK] = can_swing.astype(np.float32)
            # swap back to sword after the axe lands its disable
            back_to_sword = axe_out & (env.shield_off[:, o] > 0)
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
