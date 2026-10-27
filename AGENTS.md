# AGENTS.md — Money SMP kit knowledge (from an HT3 player)

Context captured from a human who is **HT3 at this kit**, so it outranks any
guess in the code or the sim. Where this file conflicts with `mc_pvp/sim.py`,
**this file wins** and the sim gets fixed.

Status of the doc: these were the questions asked to fill in the gaps the sim
does not model. Answers are recorded inline. Anything marked **OPEN** is still
unanswered and still blocks work.

---

## Ground truth that changed the sim

### 1. Sprint-hit knockback (was `SPRINT_KNOCKBACK_BONUS_H = 0.5`)

> "Sprint-hit deal the most knockback (190% to be exact)"

Knockback is **1.9x** a standing hit, i.e. multiplicative, not additive.
Applied in `mc_pvp/sim.py` as `SPRINT_KNOCKBACK_MULT = 1.9`
(`BASE_KNOCKBACK_H * 1.9` when the attacker is sprinting).

The old additive `+0.5` was 2.67x, which overshoots and would have taught the
policy wrong combo spacing.

### 2. Does a sprint hit reset your sprint?

> "It does not reset your sprint (this is why w tap and s tap is still yk)"

**No.** `SPRINT_CANCELS_ON_HIT = False`. This is precisely why W-tap / S-tap
resets exist and are worth learning — if the sim cancelled sprint on hit, the
policy would never be able to chain.

---

## Q&A: melee

**Q1. Opener — the fight starts 16 blocks apart, both buffed. Sprint-jump-crit
straight in, or patient walk-up?**

> "When they encounter you should always eat to full saturation then sprint and
> try to get a crit WITH AN axe"

Sequence: **saturation to full -> sprint in -> axe crit** as the first damage.
Not a sword opener — the axe carries the opener.

**Q2. When do you take a trade vs disengage? Is there an HP rule?**

> "When you take a trade there is one golden rule if you can follow this you
> gotta win most trade EAT TO FULL SATURATION before every trade but if you
> cant dont or else your gonna lose dura"

The rule is **saturation, not HP**. Full saturation = the trade is worth taking
(regeneration backfills the damage). Unsaturated = do not trade. Breaking this
loses on durability even when it looks even on HP.

**Q3. After landing a hit, chase to keep the combo or reset spacing? What keeps
a combo going?**

> "Combo should only happen when either enemy shield is disabled or you shield
> sac (let enemy break the shield but only if enemy shield is already disabled).
> There is w-tap and s-tap. W-tap is normally but S-tap IF you are too close"

- Combo only when **enemy shield is disabled**, or after a **shield sac**
  (deliberately let them break your shield — but only when *their* shield is
  already disabled).
- **W-tap** is the default combo extender; **S-tap** when too close.

**Q4. Do you actually jump-crit in real fights? And if they crit you first?**

> "Crit always happen at the start of the trade. You should mainly go for crit
> IF they crit you do not trade it with sprint hit you lose more hp and dura"

- Crits happen at the **start** of a trade, not randomly mid-fight.
- If they crit you, **do not answer with a sprint hit** — you lose HP and
  durability on that exchange.

**Q5. Axe — shield-break only, or raw damage too? How is the disable set up?**

> "Axe is use at the start of the trade and use to shield disable :1"

Axe is the **opener** and the **shield-disable** tool. Those are its two jobs.

**Q6. When the enemy whiffs in front of you, what is the conversion?**

> "If enemy whiff you could just crit chain"

A whiff licenses a **crit chain** — the punish is a crit sequence, not a single
pity hit.

---

## Q&A: healing

**Q7. Gappling — how far is safe to eat at, and what HP do you go eat?**

> "Safe distance is where you can eat without getting hit there is specific
> block. At 10 hp is good to go eat but you could agro and use health pot"

- "Safe" is defined by **whether they can reach you**, not an absolute range.
- **~10 HP** is the threshold to disengage and eat.
- Alternative at low HP: stay aggressive and let a **health pot** cover you.

**Q8. Egap — panic button, or planned pre-fight buff?**

> "Use this like normal gap but only if you cant eat gap safely (should not be
> use everytime)"

Egap is a **gapple substitute when eating a normal gapple is not safe**. Not a
pre-fight ritual, not a panic reflex.

**Q9. Splash heal (Instant Health II, +8) — when do you throw it?**

> "You throw this when either you can't gap safely or agro"

Same trigger family as the egap: **can't gapple safely, or going agro**.

**Q10. Pre-fight pot priority? Fire res mandatory?**

> "Just pot all of them"

No priority needed — **pot everything**. (So: no ordered curriculum required.)

**Q11. Mid-fight re-potting — suicide, or required?**

> "You need to repot instantly or else you will lose dmg and dura (look down and
> throw the pot is always faster)"

**Required, and immediately on expiry.** "Look down to throw" is a real speed
trick — the bridge should throw with pitch down rather than fumbling, and the
bot acts on the tick anyway.

---

## Q&A: shield protocol

**Q12. When is the shield up vs down? How do you beat a turtling shield?**

> "Use shield only when enemy can hit you (3 block) if you get sprint hit always
> do not use a shield. If some hold up their shield you could either hold your
> shield longer wait until they hit and hit back or just shield disable"

- Shield **only inside ~3 blocks** (when they can actually reach you).
- After **taking a sprint hit, do not shield** (you are in the KB window).
- vs a turtle: either **out-wait them** (hold longer, hit back after their
  swing) or **shield-disable with the axe**.

**Q13. Your shield just got disabled for 5s — what's the protocol?**

> "If your shield disable you should play more passive"

Play **passive** for the disable duration.

Also from Q3: a **shield sac** (letting them break your shield deliberately) is
legal *only* when the enemy shield is already disabled.

---

## Q&A: spacing / positioning

**Q14. Where do you hover — just outside 3 blocks baiting whiffs? Can you give
numbers to reward?**

> "Spacing is kinda hard topic"

**OPEN** — no numbers given. Deliberately not being invented here; if a
quantitative version shows up (e.g. time spent at 3-4 blocks vs face-hugging),
it becomes a reward term.

**Q15. Does the arena edge matter on the plains?**

> "Getting backed to anything is dangerous bc you could get headwebed and
> fullboxed and died"

Being **backed against anything** is dangerous — head web then full box into a
death. Edge proximity is a real cost, and webs are part of why.

**Q16. Durability — does it decide games?**

> "durability decide games for sure there is quickmend and perfect mend. quick
> mend just put your shield always (totem in offhand) then just spam xp. perfect
> mend is basically take off your shield then mend all of the armor to has the
> same dura (350 but if you cant reach that 300 and yk)"

**Durability decides games.** Two protocols:

- **Quick mend** — shield always up (totem in offhand), then spam XP.
- **Perfect mend** — take the shield off, then mend all armour to the *same*
  durability, ideally **350**, or **300** if 350 is not reachable.

Confirmed sequence (see A below):

> "Take off the shield -> put totem in offhand -> mend until 1 armor piece reach
> n (350) -> after 1 armor piece reach take it off -> repeat"

So perfect mend is applied **one piece at a time**: mend a piece to 350, remove
it from the mend loop, move to the next, repeat until every piece matches.

Kit has **3 stacks of XP bottles**.

---

## Follow-ups that were answered

**A. Mending mechanics — recorded above** (3 stacks of XP bottles; shield off;
totem to offhand; mend piece to 350; take that piece off; repeat).

**B. Answering a crit — recorded above.**

> "You try to get a crit back. Or you could do sword technique to start a combo"

Trade back with **your own crit**, or use **sword technique to open a combo**
instead of trading sprint hits.

**C. "Agro" when low HP — recorded above.**

> "So if you are winning that trade you could not eat and try to go for
> quickdrop because they would has low saturation"

Agro = you are **winning the exchange**, so skip eating and push for a
**quickdrop** — they are low on saturation and cannot sustain the trade.

**D. Potting at spawn — recorded above.**

> "Usually in diamond smp they pot at the start of the fight (dont walk up)"

Pot **at the start, before walking up**. Do not pot mid-approach.

**F. "Too close" for an S-tap — recorded above.**

> "Too close if let say you sprint hit them but they still be able to hit you"

Too close = **you sprint-hit them and they can still reach you back**.

---

## Still OPEN

- **E. Egap selection.** Answer was "you could put egap in hotbar no? but the kit
  didn't so erm idk". Unresolved: the live kit has no egap hotbar slot, so this
  is a **deployment** question (add egap to the hotbar, or bind egap to a
  dedicated key) rather than a sim question. Do not invent a hotbar layout.
- **Q14 spacing numbers** — still no quantitative definition.
- **Webs / head web + fullbox** are referenced by the player (Q15) but **not
  modeled** in phase 1 of the sim. Fire res, mending/durability, and XP bottles
  are likewise not yet modeled.

---

## Rules for future agents on this repo

1. Do not re-add an HP-lead timeout prize; timeouts score `R_TIMEOUT = -3.0`
   plus capped stall bills (bloodless `coward`, out-of-range `turtle_bill`,
   bloodless `eat_bill`) for both sides (see `mc_pvp/sim.py`). A timeout
   must stay strictly worse than losing a fight, or the policy turtles
   into draws.
2. Do not let the whiff penalty charge per tick — attack is edge-triggered.
3. Keep `mc_pvp/sim.py:obs()` and `ppo-bot/.../Observation.java` in lockstep.
   A reordered or rescaled feature silently lobotomises the deployed policy.
4. If a change touches physics constants, re-run `python -m mc_pvp.validate`
   (currently 14/14) and re-export with `tools/export_policy.py`.
5. `python -u` when piping trainer output, always.

## Implemented HT3 rewards (in `mc_pvp/sim.py`, from the interview above)

Mechanics added: saturation regen (vanilla ~1 HP / 4s while saturated — this is
what makes the golden rule real), `since_opp_whiff` / `since_took_crit` /
`since_str_out` punish-window timers.

| constant | value | pays for |
|---|---|---|
| `R_SAT_HIT` | +0.03 | hit landed while saturated (>=18) |
| `R_OPENER` | +0.15 | episode's first damage is an axe crit |
| `R_WHIFFPUNISH` | +0.15 | crit within 1s of enemy whiff |
| `R_CRITBACK` | +0.10 | own crit within 1s of taking a crit |
| `R_DISABLE` | +0.20 | landing a shield disable |
| `R_BLOCK` | +0.02 | shield cleanly negates a hit |
| `R_EAT_SAFE` | +0.10 | finished eating at <=12 HP out of reach |
| `R_POT_HEAL` | +0.05 | splash heal thrown at <=12 HP |
| `R_REPOT` | +0.10 | str re-thrown near/after expiry |
| `R_EGAP_WASTE` | -0.05 | egap burned while safe |
| `R_TURTLE` | -0.001/tick | shield up while enemy out of range |
| `R_EDGE` | -0.002/tick | within 2 blocks of the wall |
| `R_SHELTER` | 2x approach, outward | hold distance while shield disabled |

NOT yet modeled: durability/mending + XP bottles, totems, webs/headweb.
