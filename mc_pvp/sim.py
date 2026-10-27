"""Tick-accurate Minecraft 1.21 sword-duel simulation.

Scope: vanilla 1.9+ melee combat, flat arena, two players.
Everything runs at the vanilla tick rate (20 TPS) because the skills we want
the agent to learn - cooldown discipline, reach trading, crit timing, spacing -
are all tick-timed.

Units: 1 block. Angles: degrees in the public API.

Constants marked CERTAIN are taken from vanilla decompiled behaviour.
Constants marked TUNING are approximations we should verify in-game.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Tick rate
# --------------------------------------------------------------------------
TICK_RATE = 20
DT = 1.0 / TICK_RATE

# --------------------------------------------------------------------------
# Movement - vanilla LivingEntity.travel()
#   horizontal: v = (v + input*accel) * drag
#   vertical:   v = (v - gravity) * drag
# --------------------------------------------------------------------------
GRAVITY = 0.08              # CERTAIN
VERTICAL_DRAG = 0.98        # CERTAIN
AIR_DRAG = 0.91             # CERTAIN
BLOCK_SLIPPERINESS = 0.6    # CERTAIN (normal block)
GROUND_DRAG = BLOCK_SLIPPERINESS * AIR_DRAG              # 0.546
# Vanilla scales ground accel by 0.16277136 / drag^3 (~1.0 on default ground),
# and the empirical 4.317 / 5.612 b/s speeds work out to accel = speed * 0.98.
INPUT_ACCEL_SCALE = 0.98
USE_SLOW = 0.2           # CERTAIN (decompiled LocalPlayer.aiStep: input *= 0.2F
                            # while using an item; wiki: blocking slows to sneaking pace).
                            # Sprint does NOT cancel - the flag persists and
                            # sprint-then-block stays faster than walk-block (MC-152728,
                            # confirmed live). Only the legs are slow.
WALK_SPEED = 0.1            # CERTAIN -> 4.317 blocks/s
SPRINT_SPEED = 0.13         # CERTAIN -> 5.612 blocks/s
AIR_ACCEL = 0.02            # CERTAIN
JUMP_VELOCITY = 0.42        # CERTAIN -> 1.2522 block apex
SPRINT_JUMP_BOOST = 0.2     # CERTAIN

# --------------------------------------------------------------------------
# Body
# --------------------------------------------------------------------------
PLAYER_WIDTH = 0.6
PLAYER_HEIGHT = 1.8
EYE_HEIGHT = 1.62
PLAYER_PUSH = 0.05          # TUNING (vanilla entity push is ~0.05/tick)

# --------------------------------------------------------------------------
# Combat - 1.9+ cooldown model
# --------------------------------------------------------------------------
REACH = 3.0                 # CERTAIN (survival attack range)
HITBOX_SLOP = 0.1           # TUNING (vanilla expands the target box slightly)
ATTACK_SPEED = 1.6          # CERTAIN (diamond sword)
COOLDOWN_TICKS = TICK_RATE / ATTACK_SPEED                # 12.5 ticks
HURT_IMMUNITY = 10          # CERTAIN
SWORD_DAMAGE = 8.0          # CERTAIN (netherite sword, Diamond SMP kit)
AXE_DAMAGE = 10.0           # CERTAIN (netherite axe base)
SHARPNESS_BONUS = 3.0       # CERTAIN (Sharp V: 1 + 0.5*(5-1) = +3.0, sword and axe)
CRIT_MULT = 1.5             # CERTAIN
# Axe cooldown is 1.0s (slower than the sword's 1.6/s). Tracked per weapon.
AXE_COOLDOWN_TICKS = TICK_RATE / 1.0                  # 20.0 ticks
# Sprint-axe vs a raised shield disables it for 5s (100 ticks).
SHIELD_DISABLE_TICKS = 100  # CERTAIN
# Shield raise has a 250ms (~5 tick) delay before it starts blocking.
SHIELD_RAISE_TICKS = 5       # HT3 callout
# Weapon swap (sword<->axe attribute swap) is instant - same tick.
# Healing: gapple restores 4 hunger + 9.6 saturation; the bot must keep
# saturation maxed (HT3: eat until saturated, always make distance first).
GAPPLE_SATURATION = 9.6    # CERTAIN (4 hunger, 9.6 saturation)
EAT_TICKS = 32              # CERTAIN (1.6s to eat a gapple)
ENCH_GAPPLE_ABSORB = 8.0    # CERTAIN (Absorption IV: 16 half-hearts = 8 HP)
# Splash counts/durations from the Money SMP kit datapack.
SPLASH_HEAL = 8.0           # CERTAIN (Instant Health II splash = 8 HP)
N_GAPPLES = 128             # kit count (64 hotbar + 64 inventory)
N_EGAPPLES = 2              # kit count
N_HEAL_POTS = 4             # kit count
N_STR_POTS = 9              # kit count (vanilla strong_strength is 1:30)
N_SPD_POTS = 3              # kit count
N_FIR_POTS = 3              # kit count
N_WEAVE_POTS = 3            # kit count
STR_DURATION = 20 * 90      # CERTAIN (strong_strength splash = 1:30)
SPD_DURATION = 20 * 480     # server fix: long_swiftness = 8:00 (datapack 3-min bug)
FIR_DURATION = 20 * 480     # server fix: long_fire_resistance = 8:00
STR_BONUS = 6.0             # CERTAIN (Strength II: +3 per level, Java)
SPD_MULT = 1.2              # CERTAIN (Speed I: +20% move)
FIRE_TICKS = 8 * 20         # Fire Aspect II: ~8s burn (1 dmg/s through armour)
# Weaving (movement half only, phase 1): 50% speed in web vs 25% without.
WEAVE_WEB_MULT = 0.5        # HT3: 50% of original speed in web
BASE_WEB_MULT = 0.25        # vanilla web slow
WEAVE_DURATION = 20 * 60    # weaving effect lasts 60s once splashed
# Webs: 2 patches per duel, radius in blocks. Phase 2 models movement +
# escape only (no headweb damage, no fullbox). Web positions random per reset.
N_WEBS = 2
WEB_RADIUS = 0.9
# Full maxed diamond armour (Prot IV, no Thorns): 20 armour, 8 toughness
# (2 per piece), EPF 5 per Prot IV piece -> 20 total. Deterministic 1.21
# CombatRules: absorb then magic-absorb, no random roll in modern versions.
ARMOR_POINTS = 20.0         # CERTAIN (diamond helm 3 + chest 8 + legs 6 + boots 3)
ARMOR_TOUGHNESS = 8.0       # CERTAIN (2.0 per diamond piece)
PROT_EPF = 20.0             # CERTAIN (Prot IV EPF 5 x4, capped at 20)
COOLDOWN_DAMAGE_FLOOR = 0.2 # CERTAIN (damage = base * (0.2 + 0.8 * c^2))

BASE_KNOCKBACK_H = 0.30     # CALIBRATED against real 1.21.11 (tools/calibrate_kb.py)
                            # vanilla source reads 0.4; the measured arc needs 0.30
BASE_KNOCKBACK_V = 0.4      # vanilla LivingEntity.knockback strength
RESTING_VY = -0.0784        # a grounded entity's leftover vy (g * 0.98)
SPRINT_KNOCKBACK_MULT = 1.9  # HT3 ground truth: sprint-hit deals 190% knockback
SPRINT_CANCELS_ON_HIT = False   # HT3 ground truth: landing a hit does NOT reset sprint


def damage_after_armor(raw: np.ndarray) -> np.ndarray:
    """Vanilla 1.21 CombatRules for melee vs Prot IV diamond, deterministic."""
    f = 2.0 + ARMOR_TOUGHNESS / 4.0
    f1 = np.clip(ARMOR_POINTS - raw / f, ARMOR_POINTS * 0.2, 20.0)
    after = raw * (1.0 - f1 / 25.0)
    epf = float(np.clip(PROT_EPF, 0.0, 20.0))
    return np.maximum(after * (1.0 - epf / 25.0), 0.0)

# --------------------------------------------------------------------------
# Episode
# --------------------------------------------------------------------------
DEFAULT_MAX_TICKS = 20 * 180     # 3 minute duels (real kit fights run 5+ min)
DEFAULT_ARENA = 32.0         # half-extent (64x64, Money SMP plains size)
SPAWN_DISTANCE = 16.0        # HT3: fights start buffed, 16 blocks apart
MAX_LOOK_PER_TICK = 20.0         # degrees. 20/tick = 400 deg/s, a fast but
                                 # human-plausible flick, and a much smoother
                                 # action space than 40 (which let PPO spin).

# Reward weights (kit v1: damage signal rescaled for ~1.0/hit TTK).
# A full kill is ~20-24 armored HP through healing; R_DAMAGE * that should be
# on the order of R_WIN so trading damage is worth slightly less than winning.
R_DAMAGE = 0.15
R_WHIFF = -0.002   # per whiffed swing; spamming attack should not be free
                   # but at 1.6 swings/s this must stay small vs the win bonus
R_WIN = 1.0
# Anti-coward rule: a timeout (draw) must score strictly worse than
# LOSING. A loss teaches the value function what dying costs; a draw used
# to pay -1.5 = free damage-lessons, so the policy turtled into timeouts
# (block 65% + eat 28%, ~6 dmg/duel at iter 131) instead of fighting.
# Losing now pays -R_WIN + damage dealt (you fought); drawing pays
# R_TIMEOUT plus the stall bills (you refused). Engaging and dying beats
# turtling out the clock.
R_TIMEOUT = -3.0
# Cowardice clock: a timeout also bills per-tick stall penalties, capped.
# Bloodless stalling costs R_COWARD_TICK/tick (cap R_COWARD_CAP); blocking
# out of range costs R_TURTLE_TICK/tick (cap R_TURTLE_CAP); eating while
# nobody bleeds costs R_STALL_EAT_TICK/tick (cap R_STALL_EAT_CAP). A long
# honest fight resets the bloodless clock on every hit, so only stalling is
# billed. Episodes that end in a kill ignore all three.
R_COWARD_TICK = -0.002   # per tick with no damage from either side
R_COWARD_CAP = -1.0      # floor on the accumulated bloodless bill
R_TURTLE_TICK = -0.004   # per tick blocking while the enemy is out of reach
R_TURTLE_CAP = -1.5      # floor on the accumulated turtle bill
R_STALL_EAT_TICK = -0.01  # per tick eating while nobody bleeds
R_STALL_EAT_CAP = -1.5    # floor on the accumulated stall-eat bill
# Potential-based shaping on "distance outside reach". Sums to ~0 over an
# episode, so it cannot be farmed, but it gives a gradient for closing in -
# without it a random policy never lands a hit and gets no learning signal.
R_APPROACH = 0.012
# --- HT3 gamesense rewards (agents.md) ------------------------------------
# All event-based (fire on hits/uses), so they cannot be farmed by stalling.
# Per-tick terms are potential-shaped or tiny by design.
R_SAT_HIT = 0.03      # hit landed while saturated (>=18): the golden rule (Q2)
R_OPENER = 0.15       # episode's first damage is an axe crit: the HT3 opener (Q1)
R_WHIFFPUNISH = 0.15  # crit landed within 1s of the opponent's whiff (Q6)
R_CRITBACK = 0.10     # own crit within 1s of taking a crit: answer back (follow-up B)
R_DISABLE = 0.20      # landing a shield disable: the stun that starts combos (Q5)
R_BLOCK = 0.02        # shield negates an incoming hit cleanly (Q12)
R_EAT_SAFE = 0.10     # finished eating at <=12 HP out of reach (Q7)
R_POT_HEAL = 0.05     # splash heal thrown at <=12 HP (Q9)
R_REPOT = 0.10        # str re-thrown near/after expiry: repot instantly (Q11)
R_EGAP_WASTE = -0.05  # egap burned while safe (enemy out of reach) (Q8)
R_TURTLE = -0.001     # shield held up while enemy is out of range (Q12)
R_EDGE = -0.002       # backed within 2 blocks of the wall (Q15)
R_SHELTER = 0.024     # 2x approach pull, outward, while own shield disabled (Q13)
# ---- combo / discipline rewards (weak-hit + swap-spam fixes) ---------------
R_FULL = 0.05         # hit landed at >=90% charge: full-charge discipline
R_WEAK = -0.04        # hit landed at <60% charge: weak-hit spam tax
R_CLEAN = 0.06        # hit landed at >=60% charge while untouched for 20+ ticks:
                        # clean contact. Symmetric R_DAMAGE nets ~0 on even trades,
                        # and a baby brain only makes even trades - so early fighting
                        # has no gradient and it learns not to fight at all. This pays
                        # hit-and-don't-get-hit-back IMMEDIATELY (on the hit tick),
                        # and graduates into whiff-punish/crit-back (which outpay it)
                        # as timing improves. Skill ladder stays monotone:
                        # clean (0.06) < full (0.05+0.06) < crit-back (0.10) < punish
                        # (0.15) < disable (0.20). Anti-farm: needs real applied damage
                        # (blocked hits pay nothing), a 20-tick untouch window the
                        # opponent must grant (styles never cooperate), and >=0.6 charge
                        # (weak spam still eats R_WEAK). Stalling earns zero (no hits).
R_SWAP = -0.03        # per attribute-swap edge: swaps must earn (disable), not flap.
                        # A disable (+0.20) outpays ~7 swaps; flapping every swing bleeds.
R_SPRINT_CLOSE = 0.001  # per tick sprinting with forward input, close up (<6 blocks)
                        # small: crits (which need no-sprint) still outpay it when timed
# ---- web / weaving rewards (phase 2) ------------------------------------
R_IN_WEB = -0.002     # per tick stuck in a web: learn to get out
R_WEB_ESCAPE = 0.15   # exited a web: the escape itself
R_WEAVE_USE = 0.08    # weaving splash thrown while in a web: correct answer

# ---- wasteful-action penalties (diag showed block 0.98, eat 0.71, splash 0.92) ---
R_EAT_WASTE = -0.05   # eat while saturated + full HP: not worth the downtime
R_POT_WASTE = -0.05   # splash heal when HP is already high
R_BLOCK_OUT_OF_RANGE = -0.004  # blocking inside the turtle zone is now doubly discouraged


def forward_right(yaw_deg: np.ndarray):
    """Vanilla yaw basis. yaw 0 faces +Z, increasing yaw turns toward -X."""
    yr = np.radians(yaw_deg)
    return -np.sin(yr), np.cos(yr), -np.cos(yr), -np.sin(yr)


def ray_box_hit(origin, direction, box_min, box_max, max_dist):
    """Slab test: does the ray hit the AABB within max_dist?"""
    inv = 1.0 / np.where(np.abs(direction) < 1e-9, 1e-9, direction)
    t1 = (box_min - origin) * inv
    t2 = (box_max - origin) * inv
    tmin = np.max(np.minimum(t1, t2), axis=-1)
    tmax = np.min(np.maximum(t1, t2), axis=-1)
    return (tmax >= np.maximum(tmin, 0.0)) & (tmin <= max_dist)


class DuelEnv:
    """Vectorised 1v1 sword duel. N independent games stepped together."""

    # action layout per agent. Slots map 1:1 to the kit hotbar the bridge
    # drives: sword sl.1, gapple sl.8, axe sl.9, shield offhand, pots any slot.
    A_YAW, A_PITCH, A_FWD, A_STRAFE = range(4)                          # continuous
    A_JUMP, A_SPRINT, A_ATTACK = range(4, 7)                            # discrete
    A_BLOCK, A_SWAP = range(7, 9)                                       # discrete
    A_EAT, A_SPL_HEAL, A_SPL_STR, A_SPL_SPD, A_SPL_FIR = range(9, 14)    # discrete
    A_WEAVE = 14                                                        # discrete: weaving splash at own feet
    N_CONT, N_DISC = 4, 11
    ACTION_DIM = N_CONT + N_DISC

    def __init__(self, n_envs: int = 16, seed: int = 0, arena: float = DEFAULT_ARENA,
                 max_ticks: int = DEFAULT_MAX_TICKS, latency: int = 0):
        self.n = n_envs
        self.arena = arena
        self.max_ticks = max_ticks
        self.latency = latency
        self.rng = np.random.default_rng(seed)
        self.obs_dim = None   # set on first reset

        n2 = (n_envs, 2)
        self.px = np.zeros(n2, np.float32)
        self.py = np.zeros(n2, np.float32)
        self.pz = np.zeros(n2, np.float32)
        self.vx = np.zeros(n2, np.float32)
        self.vy = np.zeros(n2, np.float32)
        self.vz = np.zeros(n2, np.float32)
        self.yaw = np.zeros(n2, np.float32)
        self.pitch = np.zeros(n2, np.float32)
        self.on_ground = np.ones(n2, bool)
        self.sprinting = np.zeros(n2, bool)
        self.health = np.full(n2, 20.0, np.float32)
        self.saturation = np.full(n2, 20.0, np.float32)  # HT3: keep maxed, eat until saturated
        self.absorb = np.zeros(n2, np.float32)          # ench-gapple absorption buffer
        self.axe_cd = np.zeros(n2, np.float32)          # axe cooldown (20 ticks)
        self.has_axe = np.ones(n2, bool)                # True = axe in hand, False = sword
        self.blocking = np.zeros(n2, bool)              # shield actively blocking (after raise delay)
        self.raise_t = np.zeros(n2, np.int32)           # ticks the block button has been held
        self.shield_off = np.zeros(n2, np.int32)        # ticks of disable left
        self.fire_t = np.zeros(n2, np.int32)            # ticks of Fire Aspect burning left
        self.str_t = np.zeros(n2, np.int32)             # Strength II ticks left
        self.spd_t = np.zeros(n2, np.int32)             # Speed I ticks left
        self.fir_t = np.zeros(n2, np.int32)             # Fire Res ticks left
        self.weave = np.ones(n2, bool)                  # legacy flag (kept for compat, use weave_t)
        self.weave_t = np.zeros(n2, np.int32)           # weaving effect ticks left (0 = off)
        self.n_weave = np.zeros(n2, np.int32)            # weaving splash pots left
        self.in_web = np.zeros(n2, bool)                 # inside a cobweb this tick
        self.was_in_web = np.zeros(n2, bool)             # inside a cobweb last tick (escape detect)
        self.web_x = np.zeros((n_envs, N_WEBS), np.float32)  # web patch centres
        self.web_z = np.zeros((n_envs, N_WEBS), np.float32)
        self.n_gap = np.zeros(n2, np.int32)             # gapples left (64+64 carry)
        self.n_egap = np.zeros(n2, np.int32)            # ench gapples left (2)
        self.n_heal = np.zeros(n2, np.int32)            # splash Healing II left
        self.n_str = np.zeros(n2, np.int32)             # splash Str II left
        self.n_spd = np.zeros(n2, np.int32)             # splash Speed I left
        self.n_fir = np.zeros(n2, np.int32)             # splash Fire Res left
        self.eating = np.zeros(n2, np.int32)            # gapple eat ticks left (1.6s = 32t)
        self.ate_ench = np.zeros(n2, bool)              # eating an ench gapple (not plain)
        self.cooldown = np.zeros(n2, np.float32)      # ticks, capped at COOLDOWN_TICKS
        self.hurt_timer = np.zeros(n2, np.int32)      # ticks of i-frames left
        self.since_hit = np.zeros(n2, np.int32)       # ticks since this player landed a hit
        self.since_hurt = np.zeros(n2, np.int32)      # ticks since this player took damage
        self.since_opp_whiff = np.zeros(n2, np.int32)  # ticks since the opponent whiffed (punish window)
        self.since_took_crit = np.zeros(n2, np.int32)  # ticks since this player took a crit (crit-back window)
        self.since_str_out = np.full(n2, 999, np.int32)  # ticks since Str II expired (repot window)
        self.prev_attack = np.zeros(n2, bool)         # attack button last tick (edge trigger)
        self.prev_swap = np.zeros(n2, bool)           # swap key last tick (edge trigger)
        # --- bridge governors (mirrored in BotController.java) ----------------
        # The bridge suppresses toddler exploits: at most one splash per 30
        # ticks (shared across all four pot types), eat only when needy with
        # a 40-tick gap between gapples, and no re-turtle for 10 ticks after
        # potting. The sim must enforce the same, or the policy learns a
        # chug-and-turtle exploit that the bridge then neuters on the server.
        self.spl_cd = np.zeros(n2, np.int32)          # ticks until next splash allowed
        self.eat_cd = np.zeros(n2, np.int32)          # ticks until next eat may start
        self.blk_sup = np.zeros(n2, np.int32)         # ticks of forced shield-down after a pot
        self.coward = np.zeros(n_envs, np.float32)    # accumulated bloodless-stall penalty
        self.turtle_bill = np.zeros(n2, np.float32)  # per-seat out-of-range block bill
        self.eat_bill = np.zeros(n2, np.float32)     # per-seat bloodless-eat bill
        self.prev_pen = np.zeros(n_envs, np.float32)  # distance outside reach, last tick
        self.ep_ret = np.zeros(n2, np.float32)
        self.ep_dmg = np.zeros(n2, np.float32)
        self.ep_len = np.zeros(n_envs, np.int32)
        # Drill id active in each env (0 = melee, 1 = heal-or-die). The policy
        # cannot see it: HP / distance / timers are all ordinary obs features,
        # so there is no fingerprint to exploit, only the situation itself.
        self.drill = np.zeros(n_envs, np.int32)
        self.drill_p = 0.0  # set per-iter by the trainer; 0 = all melee
        self.tick = 0
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        n = self.n
        self.tick = 0
        self.ep_ret[:] = 0
        self.ep_dmg[:] = 0
        self.ep_len[:] = 0
        self.px[:] = 0.0
        self.py[:] = 0.0
        self.pz[:] = 0.0
        self.vx[:] = 0.0
        self.vy[:] = 0.0
        self.vz[:] = 0.0
        self.sprinting[:] = False
        self.health[:] = 20.0
        self.saturation[:] = 20.0       # pre-fight: full HP + saturation, gapple pre-eaten
        self.absorb[:] = 0.0
        self.axe_cd[:] = AXE_COOLDOWN_TICKS
        self.has_axe[:] = False         # sword out at start
        self.blocking[:] = False
        self.raise_t[:] = 0
        self.shield_off[:] = 0
        self.fire_t[:] = 0
        self.str_t[:] = STR_DURATION    # pre-potted: Str II + Speed I + Fire Res
        self.spd_t[:] = SPD_DURATION
        self.fir_t[:] = FIR_DURATION
        self.weave[:] = False
        self.weave_t[:] = 0             # no weaving until splashed (throw it in webs)
        self.n_weave[:] = N_WEAVE_POTS
        self.in_web[:] = False
        self.was_in_web[:] = False
        # 2 web patches per duel, off to the sides so spawn lane stays clean
        self.web_x[:] = self.rng.uniform(-8.0, 8.0, size=(n, N_WEBS)).astype(np.float32)
        self.web_z[:] = self.rng.uniform(4.0, 10.0, size=(n, N_WEBS)).astype(np.float32) * np.where(self.rng.random((n, N_WEBS)) < 0.5, -1.0, 1.0).astype(np.float32)
        self.n_gap[:] = 128             # 64 hotbar + 64 inventory
        self.n_egap[:] = 2
        self.n_heal[:] = N_HEAL_POTS
        self.n_str[:] = N_STR_POTS
        self.n_spd[:] = N_SPD_POTS
        self.n_fir[:] = N_FIR_POTS
        self.eating[:] = 0
        self.ate_ench[:] = False
        self.cooldown[:] = COOLDOWN_TICKS
        self.hurt_timer[:] = 0
        self.since_hit[:] = 999
        self.since_hurt[:] = 999
        self.since_opp_whiff[:] = 999
        self.since_took_crit[:] = 999
        self.since_str_out[:] = 999
        self.prev_attack[:] = False
        self.prev_swap[:] = False
        self.spl_cd[:] = 0
        self.eat_cd[:] = 0
        self.blk_sup[:] = 0
        self.coward[:] = 0.0
        self.turtle_bill[:] = 0.0
        self.eat_bill[:] = 0.0
        self.on_ground[:] = True
        self.pitch[:] = 0.0
        # spawn: symmetric, 16 blocks apart (Money SMP start), facing each other
        side = np.where(self.rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        self.px[:, 0] = -SPAWN_DISTANCE / 2 * side
        self.px[:, 1] = SPAWN_DISTANCE / 2 * side
        self.yaw[:, 0] = 90.0     # -X
        self.yaw[:, 1] = 270.0    # +X
        self.yaw[:, 0] = np.where(side > 0, 90.0, 270.0)
        self.yaw[:, 1] = np.where(side > 0, 270.0, 90.0)
        self.prev_pen[:] = np.maximum(self._dist() - REACH, 0.0)
        # drills reroll a share of envs AFTER the standard spawn above.
        self.drill[:] = 0
        self._apply_drill(np.arange(n))
        return self.obs()

    def _dist(self) -> np.ndarray:
        return np.hypot(self.px[:, 1] - self.px[:, 0], self.pz[:, 1] - self.pz[:, 0])

    # -------------------------------------------------------------- observation
    def obs(self) -> np.ndarray:
        """Self-centric observation for each player, shape (n, 2, F).

        Both seats see the same kind of vector, so one policy can play either.
        Player 0's view is exactly player 1's mirrored, so the policy cannot
        learn a seat-specific quirk.
        """
        dx = np.stack([self.px[:, 1] - self.px[:, 0],
                       self.px[:, 0] - self.px[:, 1]], axis=1)
        dz = np.stack([self.pz[:, 1] - self.pz[:, 0],
                       self.pz[:, 0] - self.pz[:, 1]], axis=1)
        f_x, f_z, r_x, r_z = forward_right(self.yaw)

        rel_f = dx * f_x + dz * f_z
        rel_r = dx * r_x + dz * r_z
        dist = np.hypot(dx, dz)
        flat = np.maximum(dist, 1e-6)

        # Aim error to the opponent's chest. This is observable (it is literally
        # where the enemy is on screen) and it is the single feature that turns
        # "learn to aim" from a needle-in-a-haystack into a regression.
        dy = np.stack([self.py[:, 1] - self.py[:, 0],
                       self.py[:, 0] - self.py[:, 1]], axis=1)
        aim_yaw = np.degrees(np.arctan2(rel_r, rel_f)) / 180.0
        want_pitch = np.degrees(-np.arctan2(dy, flat))
        aim_pitch = np.clip((want_pitch - self.pitch) / 45.0, -1.0, 1.0)

        opp = 1 - np.arange(2)[None, :].repeat(self.n, 0)
        # per-player velocity in own frame
        self_f = self.vx * f_x + self.vz * f_z
        self_r = self.vx * r_x + self.vz * r_z
        opp_vx = self.vx[np.arange(self.n)[:, None], opp]
        opp_vz = self.vz[np.arange(self.n)[:, None], opp]
        rel_vf = (opp_vx - self.vx) * f_x + (opp_vz - self.vz) * f_z
        rel_vr = (opp_vx - self.vx) * r_x + (opp_vz - self.vz) * r_z

        feats = [
            self.vx * 5.0, self.vz * 5.0, self.vy * 5.0,
            self.py / 2.0,
            self.on_ground.astype(np.float32),
            self.cooldown / COOLDOWN_TICKS,
            self.health / 20.0,
            self.hurt_timer / HURT_IMMUNITY,
            self.sprinting.astype(np.float32),
            rel_f / 5.0,
            rel_r / 5.0,
            dist / 5.0,
            aim_yaw,
            aim_pitch,
            np.clip((dist - REACH) / 5.0, -1.0, 1.0),
            self.cooldown[np.arange(self.n)[:, None], opp] / COOLDOWN_TICKS,
            self.health[np.arange(self.n)[:, None], opp] / 20.0,
            self.on_ground[np.arange(self.n)[:, None], opp].astype(np.float32),
            self.sprinting[np.arange(self.n)[:, None], opp].astype(np.float32),
            self_f * 5.0, self_r * 5.0,
            rel_vf * 5.0, rel_vr * 5.0,
            np.clip(self.since_hit / 40.0, 0, 1),
            np.clip(self.since_hurt / 40.0, 0, 1),
            (np.minimum(self.arena - np.abs(self.px), self.arena - np.abs(self.pz)) / self.arena),
            # --- kit state: everything a player can see on their own HUD ---
            self.saturation / 20.0,
            np.clip(self.absorb / 8.0, 0, 1),
            self.has_axe.astype(np.float32),
            self.axe_cd / AXE_COOLDOWN_TICKS,
            self.blocking.astype(np.float32),
            np.clip(self.shield_off / SHIELD_DISABLE_TICKS, 0, 1),
            np.clip(self.str_t / STR_DURATION, 0, 1),
            np.clip(self.spd_t / SPD_DURATION, 0, 1),
            np.clip(self.fir_t / FIR_DURATION, 0, 1),
            np.clip(self.weave_t / WEAVE_DURATION, 0, 1),
            np.clip(self.n_gap / 128.0, 0, 1),
            np.clip(self.n_egap / 2.0, 0, 1),
            np.clip(self.n_heal / 4.0, 0, 1),
            np.clip(self.n_str / 9.0, 0, 1),
            np.clip(self.eating / 32.0, 0, 1),
            # --- opponent kit reads: shield state, buffs, axe out ---
            # (opp health / cooldown / sprint already in feats 16/17/19)
            self.blocking[np.arange(self.n)[:, None], opp].astype(np.float32),
            np.clip(self.shield_off[np.arange(self.n)[:, None], opp] / SHIELD_DISABLE_TICKS, 0, 1),
            self.has_axe[np.arange(self.n)[:, None], opp].astype(np.float32),
            np.clip(self.str_t[np.arange(self.n)[:, None], opp] / STR_DURATION, 0, 1),
            np.clip(self.absorb[np.arange(self.n)[:, None], opp] / 8.0, 0, 1),
            # --- web / weaving reads (phase 2) ---
            self.in_web.astype(np.float32),
            self.weave_t[np.arange(self.n)[:, None], opp] / WEAVE_DURATION,
            np.clip(self.n_weave / 3.0, 0, 1),
        ]
        out = np.stack(feats, axis=-1).astype(np.float32)
        self.obs_dim = out.shape[-1]
        return out

    # ---------------------------------------------------------------- stepping
    def step(self, act: np.ndarray, record_events: bool = True):
        n = self.n
        act = np.asarray(act, np.float32)
        idx = np.arange(n)[:, None]
        events = []

        # ---- look -----------------------------------------------------------
        self.yaw = (self.yaw + np.clip(act[..., self.A_YAW], -MAX_LOOK_PER_TICK, MAX_LOOK_PER_TICK)) % 360.0
        self.pitch = np.clip(self.pitch + np.clip(act[..., self.A_PITCH], -MAX_LOOK_PER_TICK, MAX_LOOK_PER_TICK), -90.0, 90.0)

        fwd_in = np.clip(act[..., self.A_FWD], -1, 1)
        str_in = np.clip(act[..., self.A_STRAFE], -1, 1)
        jump = act[..., self.A_JUMP] > 0.5
        want_sprint = act[..., self.A_SPRINT] > 0.5
        want_attack = act[..., self.A_ATTACK] > 0.5
        want_block = act[..., self.A_BLOCK] > 0.5
        want_swap = act[..., self.A_SWAP] > 0.5
        want_eat = act[..., self.A_EAT] > 0.5
        want_heal = act[..., self.A_SPL_HEAL] > 0.5
        want_str = act[..., self.A_SPL_STR] > 0.5
        want_spd = act[..., self.A_SPL_SPD] > 0.5
        want_fir = act[..., self.A_SPL_FIR] > 0.5
        want_weave = act[..., self.A_WEAVE] > 0.5

        # vanilla: sprinting requires forward input
        self.sprinting = want_sprint & (fwd_in > 0.1)

        # weapon swap: edge-triggered toggle, same tick (HT3). Holding the
        # key does not flap: matches vanilla (and the bridge, which pulses
        # the hotbar select), and kills the zero-cost swap spam the policy
        # learned when holding swap toggled every tick.
        swap_edge = want_swap & ~self.prev_swap
        self.has_axe = np.where(swap_edge, ~self.has_axe, self.has_axe)

        # ---- shield raise: 250ms delay, only while not disabled ------------
        # Bridge mirror: no re-turtle for 10 ticks after potting (BotController
        # sets blockSuppress=10 on every splash so the bot opens up).
        self.blk_sup = np.maximum(self.blk_sup - 1, 0)
        can_raise = want_block & (self.shield_off == 0) & (self.blk_sup == 0)
        self.raise_t = np.where(can_raise, self.raise_t + 1, 0)
        self.blocking = can_raise & (self.raise_t >= SHIELD_RAISE_TICKS)
        self.shield_off = np.maximum(self.shield_off - 1, 0)

        # ---- eating: 32 ticks locked in, cannot block while eating ----------
        # Bridge mirror (BotController.eatAllowed): only when needy (HP<20
        # or saturation<18) and 40+ ticks since the last gapple finished.
        # Without this the policy chain-eats behind its shield - free on the
        # sim, suppressed on the server, so the transfer dies.
        self.eat_cd = np.maximum(self.eat_cd - 1, 0)
        needy = (self.health < 20.0) | (self.saturation < 18.0)
        start_eat = want_eat & (self.eating == 0) & needy & (self.eat_cd == 0)
        has_plain = self.n_gap > 0
        has_ench = self.n_egap > 0
        # ench gapple preferred when health is low (bot learns exact timing;
        # HT3 eats around 10 hearts with gamesense variance)
        use_ench = start_eat & has_ench & (self.health <= 12.0)
        use_plain = start_eat & ~use_ench & has_plain
        self.eating = np.where(use_ench | use_plain, EAT_TICKS, self.eating)
        self.ate_ench = np.where(use_ench | use_plain, use_ench, self.ate_ench)
        self.n_egap = np.where(use_ench, self.n_egap - 1, self.n_egap)
        self.n_gap = np.where(use_plain, self.n_gap - 1, self.n_gap)
        self.blocking = np.where(self.eating > 0, False, self.blocking)
        # eat finishes when the counter hits 1 -> apply, then decrement to 0
        finishing = self.eating == 1
        self.saturation = np.where(finishing, 20.0, self.saturation)
        self.absorb = np.where(finishing & self.ate_ench,
                                ENCH_GAPPLE_ABSORB, self.absorb)
        self.eat_cd = np.where(finishing, 40, self.eat_cd)
        self.eating = np.maximum(self.eating - 1, 0)

        # ---- splash pots: instant, at own feet, no downtime (HT3) ----------
        # Bridge mirror (BotController.governedSplash): one shared 30-tick
        # throw cooldown across all four pot types; heal only below full HP,
        # buffs only when mostly expired (<90%). Level-triggered + cooldown
        # (not edge): holding re-throws every 30 ticks, so the scripted bots
        # keep working unchanged, but the 4-tick chug-the-stack exploit that
        # diag showed (splash button 1.00) is gone.
        pre_health = self.health.copy()  # for low-HP use rewards (Q9)
        pre_str = self.str_t.copy()      # for repot rewards (Q11)
        self.spl_cd = np.maximum(self.spl_cd - 1, 0)
        spl_ready = self.spl_cd == 0
        do_heal = want_heal & (self.n_heal > 0) & spl_ready & (self.health < 20.0)
        do_str = (want_str & (self.n_str > 0) & spl_ready
                  & (self.str_t < 0.9 * STR_DURATION))
        do_spd = (want_spd & (self.n_spd > 0) & spl_ready
                  & (self.spd_t < 0.9 * SPD_DURATION))
        do_fir = (want_fir & (self.n_fir > 0) & spl_ready
                  & (self.fir_t < 0.9 * FIR_DURATION))
        # weaving: self-throw at own feet. Allowed anytime we hold one and the
        # effect is mostly expired. Shares the 30t splash governor like all pots.
        do_weave = (want_weave & (self.n_weave > 0) & spl_ready
                     & (self.weave_t < 0.9 * WEAVE_DURATION))
        any_spl = do_heal | do_str | do_spd | do_fir | do_weave
        self.spl_cd = np.where(any_spl, 30, self.spl_cd)
        self.blk_sup = np.where(any_spl, 10, self.blk_sup)
        self.n_heal = np.where(do_heal, self.n_heal - 1, self.n_heal)
        self.n_str = np.where(do_str, self.n_str - 1, self.n_str)
        self.n_spd = np.where(do_spd, self.n_spd - 1, self.n_spd)
        self.n_fir = np.where(do_fir, self.n_fir - 1, self.n_fir)
        self.n_weave = np.where(do_weave, self.n_weave - 1, self.n_weave)
        self.weave_t = np.where(do_weave, WEAVE_DURATION, self.weave_t)
        self.weave = self.weave_t > 0
        self.health = np.minimum(self.health + np.where(do_heal, SPLASH_HEAL, 0.0), 20.0)
        self.str_t = np.where(do_str, STR_DURATION, self.str_t)
        self.spd_t = np.where(do_spd, SPD_DURATION, self.spd_t)
        self.fir_t = np.where(do_fir, FIR_DURATION, self.fir_t)

        # ---- buff/fire/weave timers ---------------------------------------
        self.str_t = np.maximum(self.str_t - 1, 0)
        self.spd_t = np.maximum(self.spd_t - 1, 0)
        self.fir_t = np.maximum(self.fir_t - 1, 0)
        self.weave_t = np.maximum(self.weave_t - 1, 0)
        self.weave = self.weave_t > 0

        # ---- webs: inside check BEFORE movement so slow applies this tick ---
        # 2 patches per env; dist in XZ to patch centre < WEB_RADIUS = stuck.
        self.was_in_web = self.in_web.copy()
        d2 = np.zeros((n, 2), bool)
        for w in range(N_WEBS):
            dxw = self.px - self.web_x[:, w][:, None]
            dzw = self.pz - self.web_z[:, w][:, None]
            d2 |= (dxw * dxw + dzw * dzw) < (WEB_RADIUS * WEB_RADIUS)
        self.in_web = d2
        web_mult = np.where(self.in_web,
                            np.where(self.weave_t > 0, WEAVE_WEB_MULT, BASE_WEB_MULT),
                            1.0)

        # ---- saturation drain (sprint drains; eating restores, resolved in combat section)
        self.saturation = np.maximum(
            self.saturation - np.where(self.sprinting, 0.01, 0.002), 0.0)

        f_x, f_z, r_x, r_z = forward_right(self.yaw)

        # ---- horizontal acceleration + drag ---------------------------------
        mx = f_x * fwd_in + r_x * str_in
        mz = f_z * fwd_in + r_z * str_in
        mag = np.hypot(mx, mz)
        norm = np.where(mag > 1.0, 1.0 / np.maximum(mag, 1e-6), 1.0)
        mx *= norm
        mz *= norm

        spd_mult = np.where(self.spd_t > 0, SPD_MULT, 1.0) * web_mult
        speed = np.where(self.sprinting, SPRINT_SPEED, WALK_SPEED) * spd_mult
        # Using an item (shield button held + available, or gapple going
        # down): vanilla input x0.2. Sprint persists (see above) - a
        # sprint-blocker still moves faster than a walk-blocker, but both
        # crawl. This is what makes far-away shielding terrible: the enemy
        # closes for free while you sneak. can_raise is the button-held +
        # available gate (raise delay only gates the block EFFECT, not the slow).
        using_item = can_raise | (self.eating > 0)
        use_mult = np.where(using_item, USE_SLOW, 1.0)
        mx = mx * use_mult
        mz = mz * use_mult
        # Drag is applied to the existing velocity, then input accel is added.
        # With GROUND_DRAG=0.546 this lands on the real 4.317 / 5.612 b/s, which
        # the "accel then drag" ordering does not.
        accel = np.where(self.on_ground, speed * INPUT_ACCEL_SCALE, AIR_ACCEL)
        drag = np.where(self.on_ground, GROUND_DRAG, AIR_DRAG)
        self.vx = self.vx * drag + mx * accel
        self.vz = self.vz * drag + mz * accel

        # ---- jump -----------------------------------------------------------
        jump_now = jump & self.on_ground
        self.vy = np.where(jump_now, JUMP_VELOCITY, self.vy)
        boost = jump_now & self.sprinting
        self.vx = self.vx + np.where(boost, f_x * SPRINT_JUMP_BOOST, 0.0)
        self.vz = self.vz + np.where(boost, f_z * SPRINT_JUMP_BOOST, 0.0)

        # ---- integrate, then apply gravity + vertical drag (vanilla order) ---
        self.px = self.px + self.vx
        self.py = self.py + self.vy
        self.pz = self.pz + self.vz
        self.vy = (self.vy - GRAVITY) * VERTICAL_DRAG

        landed = self.py <= 0.0
        self.py = np.where(landed, 0.0, self.py)
        self.vy = np.where(landed, 0.0, self.vy)
        self.on_ground = landed

        # arena walls
        clipped = np.abs(self.px) > self.arena
        self.px = np.clip(self.px, -self.arena, self.arena)
        self.vx = np.where(clipped, 0.0, self.vx)
        clipped = np.abs(self.pz) > self.arena
        self.pz = np.clip(self.pz, -self.arena, self.arena)
        self.vz = np.where(clipped, 0.0, self.vz)

        # ---- player-vs-player push -----------------------------------------
        pdx = self.px[:, 1] - self.px[:, 0]
        pdz = self.pz[:, 1] - self.pz[:, 0]
        pdist = np.maximum(np.hypot(pdx, pdz), 1e-6)
        overlap = pdist < PLAYER_WIDTH
        push_x = np.where(overlap, pdx / pdist * PLAYER_PUSH, 0.0)
        push_z = np.where(overlap, pdz / pdist * PLAYER_PUSH, 0.0)
        self.vx[:, 0] -= push_x
        self.vz[:, 0] -= push_z
        self.vx[:, 1] += push_x
        self.vz[:, 1] += push_z

        # ---- cooldown / timers ---------------------------------------------
        self.cooldown = np.minimum(self.cooldown + 1.0, COOLDOWN_TICKS)
        self.axe_cd = np.minimum(self.axe_cd + 1.0, AXE_COOLDOWN_TICKS)
        self.hurt_timer = np.maximum(self.hurt_timer - 1, 0)
        self.since_hit += 1
        self.since_hurt += 1
        self.since_opp_whiff += 1
        self.since_took_crit += 1
        self.since_str_out += 1
        # fire burn: 1 HP/s through armour unless Fire Res is up
        burning = (self.fire_t > 0) & (self.fir_t == 0)
        self.health = np.where(burning & (self.tick % 20 == 0), self.health - 1.0, self.health)
        self.fire_t = np.maximum(self.fire_t - 1, 0)

        # Str expiry: became 0 this tick without re-potting (Q11 repot window).
        # (do_str would have refilled it above, so a 1 -> 0 edge is natural expiry.)
        self.since_str_out = np.where((pre_str == 1) & (self.str_t == 0), 0, self.since_str_out)

        # ---- combat ---------------------------------------------------------
        # Snapshot who is alive *before* anyone swings. Vanilla lets both
        # players kill each other on the same tick; if we read health inside the
        # loop, seat 0's hit would cancel seat 1's and seat 0 would win every
        # simultaneous exchange (that bug alone made scripted-vs-scripted 24-0).
        alive = self.health > 0
        damage_dealt = np.zeros((n, 2), np.float32)
        whiffed = np.zeros((n, 2), bool)
        bonus = np.zeros((n, 2), np.float32)  # HT3 gamesense rewards for this tick

        for p in (0, 1):
            o = 1 - p
            ox, oy, oz = self.px[:, o], self.py[:, o], self.pz[:, o]
            ex = self.px[:, p]
            ey = self.py[:, p] + EYE_HEIGHT
            ez = self.pz[:, p]
            pitch_r = np.radians(self.pitch[:, p])
            dirx = f_x[:, p] * np.cos(pitch_r)
            diry = -np.sin(pitch_r)
            dirz = f_z[:, p] * np.cos(pitch_r)
            origin = np.stack([ex, ey, ez], axis=-1)
            direction = np.stack([dirx, diry, dirz], axis=-1)
            half = PLAYER_WIDTH / 2 + HITBOX_SLOP
            box_min = np.stack([ox - half, oy - HITBOX_SLOP, oz - half], axis=-1)
            box_max = np.stack([ox + half, oy + PLAYER_HEIGHT + HITBOX_SLOP, oz + half], axis=-1)
            in_reach = ray_box_hit(origin, direction, box_min, box_max, REACH)

            # Vanilla-ish attack trigger: tapping swings immediately (any charge),
            # holding auto-swings once the cooldown is ready. Without this, holding
            # attack produced one "whiff" per tick and made attacking strictly bad.
            auto = self.cooldown[:, p] >= COOLDOWN_TICKS * 0.9
            attacking = (want_attack[:, p] & (~self.prev_attack[:, p] | auto)
                         & alive[:, p] & ~self.blocking[:, p])
            hit = (attacking & in_reach & (self.hurt_timer[:, o] == 0)
                   & alive[:, o])
            whiffed[:, p] = attacking & ~hit

            # weapon-aware cooldown: axe is 20t, sword is 12.5t
            axe_out = self.has_axe[:, p]
            cd_cap = np.where(axe_out, AXE_COOLDOWN_TICKS, COOLDOWN_TICKS)
            cd_now = np.where(axe_out, self.axe_cd[:, p], self.cooldown[:, p])
            charge = np.clip(cd_now / cd_cap, 0.0, 1.0)
            base = np.where(axe_out, AXE_DAMAGE, SWORD_DAMAGE) + SHARPNESS_BONUS
            raw = base * (COOLDOWN_DAMAGE_FLOOR + (1 - COOLDOWN_DAMAGE_FLOOR) * charge ** 2)
            # crit: falling, airborne, not sprinting (axe openers crit: HT3)
            crit = (self.vy[:, p] < 0) & ~self.on_ground[:, p] & ~self.sprinting[:, p]
            raw = raw * np.where(crit, CRIT_MULT, 1.0)
            # Strength II: +6 raw, pre-armor (vanilla Java)
            raw = raw + np.where(self.str_t[:, p] > 0, STR_BONUS, 0.0)
            dmg = damage_after_armor(raw)

            # ---- shield: blocks melee from the front while raised ------------
            # vanilla: shield blocks when the attacker is within ~100 deg of
            # the defender's facing. Approximate: dot(def_fwd, to_attacker).
            to_atk_x = self.px[:, p] - self.px[:, o]
            to_atk_z = self.pz[:, p] - self.pz[:, o]
            to_atk_n = np.maximum(np.hypot(to_atk_x, to_atk_z), 1e-6)
            def_yaw_r = np.radians(self.yaw[:, o])
            def_fx, def_fz = -np.sin(def_yaw_r), np.cos(def_yaw_r)
            facing = (to_atk_x * def_fx + to_atk_z * def_fz) / to_atk_n
            blocked = self.blocking[:, o] & (facing > -0.17)  # ~100 deg half-angle

            # ---- shield disable: sprint-AXE vs raised shield ONLY ----------
            # Vanilla: only the axe disables. Sword sprint-hits never disable,
            # so brute-forcing a turtle with the sword is just weak-chip + KB.
            # HT3 predicts the shield, 1-tick attribute-swaps sword->axe, lands
            # the sprint-axe disable, then swaps back. Disabling hit deals
            # damage but NO knockback (the stun: victim holds S).
            disabling = (hit & blocked & self.sprinting[:, p] & axe_out
                         & (self.shield_off[:, o] == 0))
            self.shield_off[:, o] = np.where(disabling, SHIELD_DISABLE_TICKS,
                                              self.shield_off[:, o])
            self.blocking[:, o] = np.where(disabling, False, self.blocking[:, o])
            dmg = np.where(hit & blocked & ~disabling, 0.0, dmg)

            # absorption (ench gapple) soaks first, vanilla order
            soak = np.minimum(self.absorb[:, o], np.where(hit, dmg, 0.0))
            self.absorb[:, o] -= soak
            applied = np.where(hit, dmg - soak, 0.0)
            # eating locks the hands: cannot attack while a gapple is going down
            applied = np.where(self.eating[:, p] > 0, 0.0, applied)
            hit = hit & (self.eating[:, p] == 0)
            self.health[:, o] -= applied
            damage_dealt[:, p] = applied
            # ---- HT3 gamesense rewards (agents.md) ------------------------
            first = (self.ep_dmg[:, p] == 0) & (applied > 0)
            bonus[:, p] += np.where(first & crit & axe_out, R_OPENER, 0.0)
            bonus[:, p] += np.where((applied > 0) & (self.saturation[:, p] >= 18.0), R_SAT_HIT, 0.0)
            bonus[:, p] += np.where((applied > 0) & crit & (self.since_opp_whiff[:, p] <= 20), R_WHIFFPUNISH, 0.0)
            bonus[:, p] += np.where((applied > 0) & crit & (self.since_took_crit[:, p] <= 20), R_CRITBACK, 0.0)
            bonus[:, p] += np.where(disabling, R_DISABLE, 0.0)
            bonus[:, o] += np.where(hit & blocked & ~disabling, R_BLOCK, 0.0)
            # charge discipline: full-charge hits pay, weak-chip hits get taxed.
            # This is what stops the 40%-charge poke spam (diag: weak hits).
            bonus[:, p] += np.where((applied > 0) & (charge >= 0.9), R_FULL, 0.0)
            bonus[:, p] += np.where((applied > 0) & (charge < 0.6), R_WEAK, 0.0)
            # Clean contact: charged hit while untouched for 20+ ticks.
            # Gate is pre-hit (since_hurt counts up AFTER the loop ticks it),
            # so landing this hit resets the clock at >=20 with no drip-feed:
            # each qualifying hit is one discrete 0.06 event on the hit tick.
            clean = ((applied > 0) & (charge >= 0.6)
                     & (self.since_hurt[:, p] >= 20))
            bonus[:, p] += np.where(clean, R_CLEAN, 0.0)
            self.since_took_crit[:, o] = np.where(crit & hit & (applied > 0), 0, self.since_took_crit[:, o])
            self.hurt_timer[:, o] = np.where(hit & (applied > 0), HURT_IMMUNITY, self.hurt_timer[:, o])
            self.since_hit[:, p] = np.where(hit, 0, self.since_hit[:, p])
            self.since_hurt[:, o] = np.where(applied > 0, 0, self.since_hurt[:, o])
            # Shared attack lockout: ANY swing resets BOTH weapon cooldowns.
            # Vanilla has one attackStrength ticker, so sword->axe->sword flapping
            # never bought free hits. The old separate timers let the policy
            # alternate weapons for ~2x DPS, which is why it swapped every swing.
            # Now swapping only earns when it buys a disable (R_DISABLE).
            self.cooldown[:, p] = np.where(attacking, 0.0, self.cooldown[:, p])
            self.axe_cd[:, p] = np.where(attacking, 0.0, self.axe_cd[:, p])
            # Fire Aspect II lights the target (no-op through Fire Res)
            self.fire_t[:, o] = np.where(hit & (applied > 0), FIRE_TICKS, self.fire_t[:, o])

            # ---- knockback (vanilla: halve current horizontal, add strength) --
            # Disabling hits deal damage but no KB (HT3 stun: victim holds S).
            kb_mult = np.where(disabling, 0.0, 1.0)
            strength_h = (BASE_KNOCKBACK_H * np.where(self.sprinting[:, p], SPRINT_KNOCKBACK_MULT, 1.0)) * kb_mult
            strength_v = BASE_KNOCKBACK_V
            kx = f_x[:, p] * strength_h
            kz = f_z[:, p] * strength_h
            self.vx[:, o] = np.where(hit, self.vx[:, o] * 0.5 + kx, self.vx[:, o])
            self.vz[:, o] = np.where(hit, self.vz[:, o] * 0.5 + kz, self.vz[:, o])
            rest_vy = np.where(self.on_ground[:, o], RESTING_VY, self.vy[:, o])
            self.vy[:, o] = np.where(
                hit,
                np.minimum(0.4, rest_vy * 0.5 + strength_v),
                self.vy[:, o],
            )
            self.on_ground[:, o] = np.where(hit & (self.vy[:, o] > 0), False, self.on_ground[:, o])

            if SPRINT_CANCELS_ON_HIT:
                self.sprinting[:, p] = np.where(hit, False, self.sprinting[:, p])

            for i in (np.nonzero(hit)[0] if record_events else ()):
                if applied[i] > 0:
                    events.append({
                        "tick": int(self.tick),
                        "attacker": int(p),
                        "target": int(o),
                        "damage": float(round(applied[i], 2)),
                        "crit": bool(crit[i]),
                        "x": float(self.px[i, o]),
                        "z": float(self.pz[i, o]),
                    })

        # punish windows arm after both seats have swung (whiffed is per-seat)
        self.since_opp_whiff[:, 0] = np.where(whiffed[:, 1], 0, self.since_opp_whiff[:, 0])
        self.since_opp_whiff[:, 1] = np.where(whiffed[:, 0], 0, self.since_opp_whiff[:, 1])

        # ---- saturation regen (vanilla ~1 HP / 4s while saturated) -----------
        # This is what makes the HT3 golden rule real: fighting saturated
        # backfills damage, fighting dry does not (Q2).
        regen_tick = (self.ep_len % 80 == 79)
        regen = regen_tick[:, None] & (self.saturation > 0) & (self.health < 20.0) & (self.health > 0)
        self.health = np.minimum(self.health + np.where(regen, 1.0, 0.0), 20.0)
        self.saturation = np.maximum(self.saturation - np.where(regen, 1.0, 0.0), 0.0)

        # ---- rewards + termination -----------------------------------------
        self.prev_attack = want_attack.copy()
        self.prev_swap = want_swap.copy()
        rew = R_DAMAGE * damage_dealt - R_DAMAGE * damage_dealt[:, ::-1] + R_WHIFF * whiffed

        # potential-based approach shaping: closing the gap outside reach pays,
        # opening it costs the same. Bounded, cannot be farmed by orbiting.
        pen = np.maximum(self._dist() - REACH, 0.0)
        gap_delta = pen - self.prev_pen
        rew += (-R_APPROACH * gap_delta)[:, None]
        self.prev_pen = pen

        # ---- cowardice clock: bleed while nobody deals damage -------------
        # Bloodless ticks accumulate a capped penalty, added ONCE at
        # timeout (folding it per-tick would double-count through the
        # value function: V learns the drip AND eats the timeout too).
        # A long honest fight resets the clock on every hit, so only
        # stalling is punished. Episodes that end in a kill ignore it.
        bloodless = (damage_dealt.sum(axis=1) <= 0.0)
        self.coward = np.where(bloodless,
                               np.maximum(self.coward + R_COWARD_TICK, R_COWARD_CAP),
                               0.0)
        # per-seat stall bills, same bloodless gate, own caps.
        d_now = self._dist()
        far_now = (d_now > REACH + 1.0)[:, None]
        turtling = self.blocking & far_now
        self.turtle_bill = np.where(bloodless[:, None],
                                    np.maximum(self.turtle_bill + np.where(
                                        turtling, R_TURTLE_TICK, 0.0), R_TURTLE_CAP),
                                    0.0)
        stalling = (self.eating > 0)
        self.eat_bill = np.where(bloodless[:, None],
                                 np.maximum(self.eat_bill + np.where(
                                     stalling, R_STALL_EAT_TICK, 0.0), R_STALL_EAT_CAP),
                                 0.0)
        # ---- HT3 event / posture rewards (agents.md) ----------------------
        d = self._dist()
        far = d[:, None] > REACH + 1.0
        inrange = d[:, None] < REACH
        # eat / pot discipline (Q7, Q8, Q9)
        rew += np.where(finishing & (self.health <= 12.0) & (d[:, None] > REACH), R_EAT_SAFE, 0.0)
        rew += np.where(do_heal & (pre_health <= 12.0), R_POT_HEAL, 0.0)
        rew += np.where(use_ench & far, R_EGAP_WASTE, 0.0)
        # repot instantly on expiry (Q11): str re-thrown while the window is fresh
        repotted = do_str & (self.since_str_out <= 60)
        rew += np.where(repotted, R_REPOT, 0.0)
        # attribute-swap discipline: every 1-tick swap costs unless it buys a disable.
        # R_DISABLE (+0.20) still outpays ~20 swaps, so predicted shield-breaks win.
        rew += np.where(swap_edge, R_SWAP, 0.0)
        # always sprint close (user callout): per-tick pay for sprinting
        # with forward input inside 6 blocks. Grounded only (no air-sprint
        # subsidy: jumps are for crits, which need NO sprint) and never
        # while slowed by shield/gapple (that IS the turtle, don't pay it).
        # Crit rewards still outpay it when a cut is the right call.
        close = d[:, None] < 6.0
        rew += np.where(self.sprinting & (fwd_in > 0.1) & close & self.on_ground
                        & ~using_item, R_SPRINT_CLOSE, 0.0)
        # webs: bleed while stuck, pay the escape, pay weaving-in-web (HT3)
        rew += np.where(self.in_web, R_IN_WEB, 0.0)
        escaped = self.was_in_web & ~self.in_web
        rew += np.where(escaped, R_WEB_ESCAPE, 0.0)
        rew += np.where(do_weave & self.in_web, R_WEAVE_USE, 0.0)
        # look-down-to-throw is instant in sim (no pitch cost), so no extra shaping
        # eating / splashing while full HP + saturated is wasteful — not free (diag: block 0.98, eat 0.71)
        rew += np.where(start_eat & (self.saturation >= 18.0) & (self.health >= 18.0), R_EAT_WASTE, 0.0)
        rew += np.where(do_heal & (pre_health >= 18.0), R_POT_WASTE, 0.0)
        # shield held up when no one is in range — raise the turtle penalty (Q12, diag: block 0.98)
        rew += np.where(self.blocking & far, R_TURTLE, 0.0)
        # block is still too sticky: punish pure blocking while out of reach even before the turtle window
        rew += np.where(self.blocking & far, R_BLOCK_OUT_OF_RANGE, 0.0)
        # backed to the wall is death (Q15)
        margin = np.minimum(self.arena - np.abs(self.px), self.arena - np.abs(self.pz))
        rew += np.where(margin < 2.0, R_EDGE, 0.0)
        # shield disabled: play passive, hold distance (Q13). Potential-based
        # like approach, so it cannot be farmed by hiding.
        rew += np.where(self.shield_off > 0, (R_SHELTER * gap_delta)[:, None], 0.0)
        rew += bonus
        rew = rew.astype(np.float32)

        dead = self.health <= 0
        any_dead = dead.any(axis=1)
        self.tick += 1
        self.ep_len += 1
        timeout = self.ep_len >= self.max_ticks
        done = any_dead | timeout

        if done.any():
            win = np.where(dead[:, 0] & ~dead[:, 1], -1.0,
                           np.where(dead[:, 1] & ~dead[:, 0], 1.0, 0.0)).astype(np.float32)
            # timeout with no kill: both sides lose, worse than losing a
            # fight. R_TIMEOUT plus the stall bills accrued while bloodless:
            # cowardice (shared), out-of-range turtling (per-seat), and
            # bloodless eating (per-seat). A loss pays -R_WIN + damage dealt
            # (you fought); a turtle-draw pays the full timeout + stall
            # bill (you refused).
            win = np.where(timeout & ~any_dead, 0.0, win)
            rew = np.where((timeout & ~any_dead)[:, None],
                           rew + R_TIMEOUT + self.coward[:, None]
                           + self.turtle_bill + self.eat_bill, rew)
            rew[:, 0] += np.where(done, -win, 0.0)
            rew[:, 1] += np.where(done, win, 0.0)

        self.ep_ret += rew
        self.ep_dmg += damage_dealt

        info = {"events": events, "tick": self.tick}
        if done.any():
            info["finished"] = [
                {"env": int(i), "ret": [float(self.ep_ret[i, 0]), float(self.ep_ret[i, 1])],
                 "health": [float(self.health[i, 0]), float(self.health[i, 1])],
                 "dmg": [float(self.ep_dmg[i, 0]), float(self.ep_dmg[i, 1])],
                 "len": int(self.ep_len[i])}
                for i in np.nonzero(done)[0]
            ]

        # reset finished duels *before* observing, so the policy never trains
        # on a terminal frame it cannot act in
        if done.any():
            self._reset_mask(done)

        return self.obs(), rew, done, info

    def _apply_drill(self, idx: np.ndarray):
        """Overwrite spawn state for drill envs (by env index).

        Split of drill_p: 40% heal-or-die, 35% web-escape, 25% repot.
        Types: 0 melee, 1 heal, 2 web, 3 repot. The policy cannot see the id:
        HP / webs / timers are all ordinary obs features.
        """
        if self.drill_p <= 0.0 or len(idx) == 0:
            return
        roll = self.rng.random(len(idx))
        # thresholds within the drilled share
        h_heal = roll < self.drill_p * 0.40
        h_web = (roll >= self.drill_p * 0.40) & (roll < self.drill_p * 0.75)
        h_repot = (roll >= self.drill_p * 0.75) & (roll < self.drill_p)
        if h_heal.any():
            h = idx[h_heal]
            self.drill[h] = 1
            hp = self.rng.uniform(8.0, 10.0, size=len(h)).astype(np.float32)
            self.health[h, 0] = hp
            self.health[h, 1] = hp
            self.saturation[h] = self.rng.uniform(6.0, 12.0,
                                                   size=(len(h), 2)).astype(np.float32)
            d = self.rng.uniform(5.0, 7.0, size=len(h)).astype(np.float32)
            side = np.where(self.rng.random(len(h)) < 0.5, -1.0, 1.0).astype(np.float32)
            self.px[h, 0] = -d / 2 * side
            self.px[h, 1] = d / 2 * side
            self.pz[h] = 0.0
            self.yaw[h, 0] = np.where(side > 0, 90.0, 270.0)
            self.yaw[h, 1] = np.where(side > 0, 270.0, 90.0)
            self.prev_pen[h] = np.maximum(d - REACH, 0.0)
        if h_web.any():
            h = idx[h_web]
            self.drill[h] = 2
            # both spawn IN a web: drop a patch on each fighter, mid range.
            # The lesson is escape + weave (50% speed) vs 25% crawl.
            d = self.rng.uniform(4.0, 6.0, size=len(h)).astype(np.float32)
            side = np.where(self.rng.random(len(h)) < 0.5, -1.0, 1.0).astype(np.float32)
            self.px[h, 0] = -d / 2 * side
            self.px[h, 1] = d / 2 * side
            self.pz[h] = 0.0
            self.web_x[h, 0] = self.px[h, 0]
            self.web_z[h, 0] = self.pz[h, 0]
            self.web_x[h, 1] = self.px[h, 1]
            self.web_z[h, 1] = self.pz[h, 1]
            self.in_web[h] = True
            self.was_in_web[h] = True
            self.str_t[h] = STR_DURATION
            self.yaw[h, 0] = np.where(side > 0, 90.0, 270.0)
            self.yaw[h, 1] = np.where(side > 0, 270.0, 90.0)
            self.prev_pen[h] = np.maximum(d - REACH, 0.0)
        if h_repot.any():
            h = idx[h_repot]
            self.drill[h] = 3
            # str about to expire (40-80 ticks left), full HP, mid range.
            # Every tick is the repot-instantly lesson R_REPOT pays for.
            self.str_t[h] = self.rng.integers(40, 80, size=(len(h), 2)).astype(np.int32)
            d = self.rng.uniform(4.0, 6.0, size=len(h)).astype(np.float32)
            side = np.where(self.rng.random(len(h)) < 0.5, -1.0, 1.0).astype(np.float32)
            self.px[h, 0] = -d / 2 * side
            self.px[h, 1] = d / 2 * side
            self.pz[h] = 0.0
            self.yaw[h, 0] = np.where(side > 0, 90.0, 270.0)
            self.yaw[h, 1] = np.where(side > 0, 270.0, 90.0)
            self.prev_pen[h] = np.maximum(d - REACH, 0.0)

    def _reset_mask(self, mask: np.ndarray):
        # Full kit reset per finished env. This must mirror reset(): the old
        # version only cleared movement/HP and left saturation, buffs,
        # inventories and eat state to rot - so after the first death every
        # respawned duel fought with depleted kits (damage 13 -> 0.8 collapse
        # at iter ~50) and training learned to fight naked. Spawn geometry
        # matches reset() too (SPAWN_DISTANCE apart, random side), not the
        # old hardcoded 8-block face-off.
        m = mask
        self.px[m, 0] = -SPAWN_DISTANCE / 2
        self.px[m, 1] = SPAWN_DISTANCE / 2
        self.py[m] = 0.0
        self.pz[m] = 0.0
        self.vx[m] = 0.0
        self.vy[m] = 0.0
        self.vz[m] = 0.0
        side = np.where(self.rng.random(int(m.sum())) < 0.5, -1.0, 1.0).astype(np.float32)
        self.px[m, 0] = -SPAWN_DISTANCE / 2 * side
        self.px[m, 1] = SPAWN_DISTANCE / 2 * side
        self.yaw[m, 0] = np.where(side > 0, 90.0, 270.0)
        self.yaw[m, 1] = np.where(side > 0, 270.0, 90.0)
        self.pitch[m] = 0.0
        self.on_ground[m] = True
        self.sprinting[m] = False
        self.health[m] = 20.0
        self.saturation[m] = 20.0
        self.absorb[m] = 0.0
        self.axe_cd[m] = AXE_COOLDOWN_TICKS
        self.has_axe[m] = False
        self.blocking[m] = False
        self.raise_t[m] = 0
        self.shield_off[m] = 0
        self.fire_t[m] = 0
        self.str_t[m] = STR_DURATION
        self.spd_t[m] = SPD_DURATION
        self.fir_t[m] = FIR_DURATION
        self.weave[m] = False
        self.weave_t[m] = 0
        self.n_weave[m] = N_WEAVE_POTS
        self.in_web[m] = False
        self.was_in_web[m] = False
        wside = np.where(self.rng.random((int(m.sum()), N_WEBS)) < 0.5, -1.0, 1.0).astype(np.float32)
        self.web_x[m] = self.rng.uniform(-8.0, 8.0, size=(int(m.sum()), N_WEBS)).astype(np.float32)
        self.web_z[m] = self.rng.uniform(4.0, 10.0, size=(int(m.sum()), N_WEBS)).astype(np.float32) * wside
        self.n_gap[m] = 128
        self.n_egap[m] = 2
        self.n_heal[m] = N_HEAL_POTS
        self.n_str[m] = N_STR_POTS
        self.n_spd[m] = N_SPD_POTS
        self.n_fir[m] = N_FIR_POTS
        self.eating[m] = 0
        self.ate_ench[m] = False
        self.cooldown[m] = COOLDOWN_TICKS
        self.hurt_timer[m] = 0
        self.since_hit[m] = 999
        self.since_hurt[m] = 999
        self.since_opp_whiff[m] = 999
        self.since_took_crit[m] = 999
        self.since_str_out[m] = 999
        self.prev_attack[m] = False
        self.prev_swap[m] = False
        self.spl_cd[m] = 0
        self.eat_cd[m] = 0
        self.blk_sup[m] = 0
        self.coward[m] = 0.0
        self.turtle_bill[m] = 0.0
        self.eat_bill[m] = 0.0
        self.prev_pen[m] = np.maximum(self._dist()[m] - REACH, 0.0)
        self.ep_ret[m] = 0
        self.ep_dmg[m] = 0
        self.ep_len[m] = 0
        # respawns also roll drills, so the mix holds all run long.
        self.drill[m] = 0
        self._apply_drill(np.nonzero(m)[0])

    # ------------------------------------------------------------------ viewer
    def snapshot(self, env_id: int = 0) -> dict:
        return {
            "tick": int(self.tick),
            "arena": float(self.arena),
            "players": [
                {
                    "x": float(self.px[env_id, p]),
                    "y": float(self.py[env_id, p]),
                    "z": float(self.pz[env_id, p]),
                    "yaw": float(self.yaw[env_id, p]),
                    "pitch": float(self.pitch[env_id, p]),
                    "health": float(self.health[env_id, p]),
                    "cooldown": float(self.cooldown[env_id, p] / COOLDOWN_TICKS),
                    "sprint": bool(self.sprinting[env_id, p]),
                    "ground": bool(self.on_ground[env_id, p]),
                }
                for p in (0, 1)
            ],
        }
