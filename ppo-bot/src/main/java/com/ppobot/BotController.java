package com.ppobot;

import net.minecraft.commands.CommandSourceStack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.permissions.PermissionSet;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.item.AxeItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;

/**
 * Drives one HeroBot fake player with the trained policy, one decision per tick.
 *
 * <p>The kit policy outputs 4 continuous heads (yaw, pitch, forward, strafe)
 * and 10 Bernoulli heads: jump, sprint, attack, block, swap, eat, splash
 * heal/str/spd/fir. Melee executes through HeroBot's action pack; hands
 * (shield, food, splash pots) go through vanilla server-side use calls,
 * exactly the way a right-click would. Weapon slots and pot stacks are found
 * by scanning the inventory every tick, so any kit layout works - but
 * {@link #kit} lays out the Diamond SMP kit in fixed slots anyway.
 */
public final class BotController {
    /** Must match mc_pvp.sim.MAX_LOOK_PER_TICK. */
    public static final float MAX_LOOK = 20.0f;
    public static final double REACH = 3.0;
    public static final double HITBOX_SLOP = 0.1;

    // Bernoulli head indices. Must match train.py:pack_action.
    private static final int D_JUMP = 0;
    private static final int D_SPRINT = 1;
    private static final int D_ATTACK = 2;
    private static final int D_BLOCK = 3;
    private static final int D_SWAP = 4;
    private static final int D_EAT = 5;
    private static final int D_HEAL = 6;
    private static final int D_STR = 7;
    private static final int D_SPD = 8;
    private static final int D_FIR = 9;
    private static final int D_WEAVE = 10;  // phase-2: weaving splash at own feet

    private Policy policy;
    private boolean running;
    private String botName;
    private String targetName;
    private double floorY;

    private boolean prevAttack;
    private int sinceHit = 999;
    private int sinceHurt = 999;
    private int prevBotHurt;
    private int prevTargetHurt;

    private String pendingBot;
    private String pendingTarget;
    private int pendingTicks;
    private float lastFwd = 0.0f;
    private float lastAtkProb = 0.0f;
    private float lastBlockProb = 0.0f;
    private float lastEatProb = 0.0f;
    private float lastSplProb = 0.0f;
    private float lastSprintProb = 0.0f;

    // kit hands state
    private int axeCdTicks = 20;
    private int nHeal = 4;
    private int nStr = 9;
    private boolean reselectWeapon;
    private boolean wasUsing;
    // toddler governors: the half-trained policy machine-guns pots and
    // chain-eats (the sim lets both go unpunished). These only suppress
    // obviously-wasteful actions - heal at full HP, double-chugging a fresh
    // buff, throwing twice in the same second - and relax as the brain
    // learns restraint. Not tactics: target selection is still all policy.
    private int tickCount;
    private int lastSplashTick = -999;
    private int lastEatEndTick = -999;
    private int blockSuppress;
    private boolean wasEatingFood;

    /**
     * Human reaction time. Discrete decisions (attack, block, eat, pots, swap,
     * jump, sprint) run this many ticks late; aim and movement stay live, like
     * a player tracking with the mouse. Bridge-only: the sim and the trained
     * weights are untouched. -Dppobot.latency=N overrides (0 = off).
     */
    public static final int LATENCY_TICKS = Integer.getInteger("ppobot.latency", 5);
    private final java.util.ArrayDeque<boolean[]> discQueue = new java.util.ArrayDeque<>();

    private String lastNote = "";

    public void setPolicy(Policy policy) {
        this.policy = policy;
    }

    public boolean isRunning() {
        return running;
    }

    public String lastNote() {
        return lastNote;
    }

    public boolean start(MinecraftServer server, String bot, String target) {
        ServerPlayer b = server.getPlayerList().getPlayerByName(bot);
        ServerPlayer t = server.getPlayerList().getPlayerByName(target);
        if (b == null) {
            lastNote = "no such player: " + bot;
            return false;
        }
        if (t == null) {
            lastNote = "no such target: " + target;
            return false;
        }
        if (policy == null) {
            lastNote = "no policy loaded (run /ppobot reload)";
            return false;
        }
        if (!HeroActions.init()) {
            lastNote = "HeroBot unavailable: " + HeroActions.problem();
            return false;
        }
        if (HeroActions.pack(b) == null) {
            lastNote = "that player is not a HeroBot bot";
            return false;
        }

        this.botName = bot;
        this.targetName = target;
        this.floorY = b.getY();
        this.prevAttack = false;
        this.sinceHit = 999;
        this.sinceHurt = 999;
        this.prevBotHurt = 0;
        this.prevTargetHurt = 0;
        this.axeCdTicks = 20;
        this.reselectWeapon = false;
        this.wasUsing = false;
        rescan(b);
        this.discQueue.clear();
        this.running = true;
        this.lastNote = "driving " + bot + " against " + target;
        return true;
    }

    public void stop(MinecraftServer server) {
        if (running && server != null) {
            ServerPlayer b = server.getPlayerList().getPlayerByName(botName);
            if (b != null) {
                b.stopUsingItem();
                HeroActions.stopAll(HeroActions.pack(b));
            }
        }
        running = false;
        lastNote = "stopped";
    }

    public void pending(String bot, String target) {
        this.pendingBot = bot;
        this.pendingTarget = target;
        this.pendingTicks = 120;     // 6 seconds for the profile fetch + join
    }

    /** Per-tick kit inventory scan: weapon/food slots, pot slots, obs counts. */
    private int swordSlot = -1;
    private int axeSlot = -1;
    private int gapSlot = -1;
    private int egapSlot = -1;
    private int healSlot = -1;
    private int strSlot = -1;
    private int spdSlot = -1;
    private int firSlot = -1;
    private int weaveSlot = -1;

    private void rescan(ServerPlayer b) {
        swordSlot = axeSlot = gapSlot = egapSlot = -1;
        healSlot = strSlot = spdSlot = firSlot = weaveSlot = -1;
        nHeal = 0;
        nStr = 0;
        Inventory inv = b.getInventory();
        for (int s = 0; s < inv.getContainerSize(); s++) {
            ItemStack st = inv.getItem(s);
            if (st.isEmpty()) {
                continue;
            }
            boolean hotbar = s < Inventory.getSelectionSize();
            // 1.21.11 has no SwordItem class (weapons are component-driven);
            // the vanilla swords/axes tags identify them instead.
            if (st.is(ItemTags.SWORDS)) {
                if (swordSlot < 0 || (hotbar && swordSlot >= Inventory.getSelectionSize())) {
                    swordSlot = s;
                }
            } else if (st.getItem() instanceof AxeItem) {
                if (axeSlot < 0 || (hotbar && axeSlot >= Inventory.getSelectionSize())) {
                    axeSlot = s;
                }
            } else if (st.getItem() == Items.GOLDEN_APPLE) {
                if (gapSlot < 0 || (hotbar && gapSlot >= Inventory.getSelectionSize())) {
                    gapSlot = s;
                }
            } else if (st.getItem() == Items.ENCHANTED_GOLDEN_APPLE) {
                if (egapSlot < 0) {
                    egapSlot = s;
                }
            } else {
                // HeroBot can only select hotbar slots, so throwing needs
                // a hotbar pot; counts cover the whole inventory.
                String kind = Observation.splashKind(st);
                if (kind != null) {
                    boolean bar = s < Inventory.getSelectionSize();
                    switch (kind) {
                        case "heal" -> {
                            nHeal += st.getCount();
                            if (healSlot < 0 || (bar && healSlot >= Inventory.getSelectionSize())) {
                                healSlot = s;
                            }
                        }
                        case "str" -> {
                            nStr += st.getCount();
                            if (strSlot < 0 || (bar && strSlot >= Inventory.getSelectionSize())) {
                                strSlot = s;
                            }
                        }
                        case "spd" -> {
                            if (spdSlot < 0 || (bar && spdSlot >= Inventory.getSelectionSize())) {
                                spdSlot = s;
                            }
                        }
                        case "fir" -> {
                            if (firSlot < 0 || (bar && firSlot >= Inventory.getSelectionSize())) {
                                firSlot = s;
                            }
                        }
                        case "weave" -> {
                            if (weaveSlot < 0 || (bar && weaveSlot >= Inventory.getSelectionSize())) {
                                weaveSlot = s;
                            }
                        }
                    }
                }
            }
        }
    }

    private void selectWeapon(Object pack, ServerPlayer b) {
        boolean hasAxe = b.getMainHandItem().getItem() instanceof AxeItem;
        int want = hasAxe ? swordSlot : axeSlot;
        // Mirror the sim: holding swap toggles every tick. The policy learns
        // to pulse it; a held bit just flaps, which is self-punishing.
        if (want >= 0 && want < Inventory.getSelectionSize()) {
            HeroActions.setSlot(pack, want);
        }
    }

    private void selectSlot(Object pack, int slot) {
        if (slot >= 0 && slot < Inventory.getSelectionSize()) {
            HeroActions.setSlot(pack, slot);
        }
    }

    /** Right-click the held stack, exactly like a player pressing RMB. */
    private static boolean rightClick(ServerPlayer b, InteractionHand hand) {
        ItemStack held = hand == InteractionHand.MAIN_HAND
                ? b.getMainHandItem() : b.getOffhandItem();
        if (held.isEmpty()) {
            return false;
        }
        InteractionResult r = b.gameMode.useItem(b, (ServerLevel) b.level(), held, hand);
        return r.consumesAction();
    }

    /**
     * Give the bot the exact Money SMP kit: run the server's own datapack
     * function on the bot, then add the training extras the datapack
     * doesn't cover (active pre-buffs, saturation). One source of truth -
     * the old parallel manifest drifted and silently gave nothing.
     */
    public boolean kit(MinecraftServer server) {
        if (!running) {
            lastNote = "not running";
            return false;
        }
        CommandSourceStack src = server.createCommandSourceStack()
                .withPermission(PermissionSet.ALL_PERMISSIONS)
                .withSuppressedOutput();
        var cmds = server.getCommands();
        String b = botName;
        cmds.performPrefixedCommand(src, "execute as " + b + " run function kits:money_smp");
        ServerPlayer check = server.getPlayerList().getPlayerByName(b);
        if (check == null || check.getItemBySlot(net.minecraft.world.entity.EquipmentSlot.CHEST).isEmpty()) {
            lastNote = "kit datapack gave nothing (is kits:money_smp loaded? run /reload)";
            return false;
        }
        cmds.performPrefixedCommand(src, "gamemode survival " + b);
        // pre-buffed, like sim reset: Str II 1:30, Speed I + Fire Res 8:00.
        // The durations match the datapack's long_ pots (fixed from 3-min).
        cmds.performPrefixedCommand(src, "effect give " + b + " minecraft:strength 90 1 true");
        cmds.performPrefixedCommand(src, "effect give " + b + " minecraft:speed 480 0 true");
        cmds.performPrefixedCommand(src, "effect give " + b + " minecraft:fire_resistance 480 0 true");
        // hunger never gates sprint, exactly like the sim
        cmds.performPrefixedCommand(src,
                "effect give " + b + " minecraft:saturation infinite 255 true");
        ServerPlayer p = server.getPlayerList().getPlayerByName(b);
        if (p != null) {
            HeroActions.setSlot(HeroActions.pack(p), 0);
            keepHealOnBar(p);
            rescan(p);
        }
        lastNote = "kitted " + b + " (money_smp datapack + pre-buffs)";
        return true;
    }

    /**
     * The datapack keeps heal pots in the inventory, but HeroBot can only
     * select hotbar slots - so the bot could never throw one. Swap a single
     * heal pot onto hotbar.5 (pickaxe, a phase-2 tool the sim doesn't use)
     * with the displaced stack. Same items, reachable layout.
     */
    private void keepHealOnBar(ServerPlayer p) {
        Inventory inv = p.getInventory();
        int healAt = -1;
        for (int s = 0; s < inv.getContainerSize(); s++) {
            if (s == 5) {
                continue;
            }
            if ("heal".equals(Observation.splashKind(inv.getItem(s)))) {
                healAt = s;
                break;
            }
        }
        if (healAt < 0) {
            return;
        }
        ItemStack tmp = inv.getItem(5).copy();
        inv.setItem(5, inv.getItem(healAt).copy());
        inv.setItem(healAt, tmp);
        inv.setChanged();
    }

    public String status(MinecraftServer server) {
        if (!running) {
            return "idle" + (lastNote.isEmpty() ? "" : " (" + lastNote + ")");
        }
        ServerPlayer b = server.getPlayerList().getPlayerByName(botName);
        ServerPlayer t = server.getPlayerList().getPlayerByName(targetName);
        if (b == null || t == null) {
            return "lost bot or target";
        }
        return String.format("%s vs %s | hp %.1f/%.1f | dist %.2f | sprint %s (food %d) | "
                        + "fwd %.2f spr %.2f | atk %.2f blk %.2f eat %.2f spl %.2f | heal %d str %d | "
                        + "held %s using %s | policy %s",
                botName, targetName, b.getHealth(), t.getHealth(),
                Math.hypot(t.getX() - b.getX(), t.getZ() - b.getZ()),
                b.isSprinting(), b.getFoodData().getFoodLevel(),
                lastFwd, lastSprintProb, lastAtkProb, lastBlockProb, lastEatProb, lastSplProb,
                nHeal, nStr,
                heldName(b), usingName(b),
                policy == null ? "none" : policy.obsDim + "x" + policy.hidden
                        + "/" + policy.disc);
    }

    public void tick(MinecraftServer server) {
        if (pendingBot != null) {
            ServerPlayer b = server.getPlayerList().getPlayerByName(pendingBot);
            if (b != null) {
                start(server, pendingBot, pendingTarget);
                pendingBot = null;
            } else if (--pendingTicks <= 0) {
                pendingBot = null;
                lastNote = "bot never appeared";
            }
            return;
        }

        if (!running || policy == null) {
            return;
        }

        ServerPlayer bot = server.getPlayerList().getPlayerByName(botName);
        ServerPlayer target = server.getPlayerList().getPlayerByName(targetName);
        if (bot == null || target == null || !bot.isAlive() || !target.isAlive()) {
            running = false;
            lastNote = "bot or target died/left";
            return;
        }

        // --- timers ------------------------------------------------------
        // Edge-detect the hurt flash rather than testing "hurtTime > 0": the
        // flash lasts 10 ticks, so the naive version pins sinceHurt at 0 for
        // half a second and feeds the policy a value the sim never produces.
        int th = target.hurtTime;
        if (th > prevTargetHurt) {
            sinceHit = 0;
        } else {
            sinceHit = Math.min(sinceHit + 1, 999);
        }
        prevTargetHurt = th;

        int bh = bot.hurtTime;
        if (bh > prevBotHurt) {
            sinceHurt = 0;
        } else {
            sinceHurt = Math.min(sinceHurt + 1, 999);
        }
        prevBotHurt = bh;
        axeCdTicks = Math.min(axeCdTicks + 1, 20);
        tickCount++;
        blockSuppress = Math.max(blockSuppress - 1, 0);

        rescan(bot);

        // --- policy ------------------------------------------------------
        float[] obs = Observation.build(bot, target, floorY, sinceHit, sinceHurt,
                axeCdTicks / 20.0f, nHeal, nStr);
        Policy.Out out = policy.forward(obs);
        if (LATENCY_TICKS > 0) {
            discQueue.addLast(out.disc.clone());
            boolean[] late = discQueue.size() > LATENCY_TICKS
                    ? discQueue.pollFirst() : new boolean[out.disc.length];
            System.arraycopy(late, 0, out.disc, 0, out.disc.length);
        }

        float newYaw = bot.getYRot() + out.mu[0] * MAX_LOOK;
        float newPitch = (float) Observation.clamp(
                bot.getXRot() + out.mu[1] * MAX_LOOK, -90.0, 90.0);

        // HeroBot's action pack consumes our inputs during the *next* player
        // tick, so a swing lands one tick after this decision. Test reach where
        // both players will be then; otherwise a chase at sprint speed whiffs
        // through a body-length of gap that closed after the check.
        Vec3 eye = bot.getEyePosition().add(bot.getDeltaMovement());
        Vec3 dir = Vec3.directionFromRotation(newPitch, newYaw);
        boolean inReach = target.getBoundingBox()
                .move(target.getDeltaMovement())
                .inflate(HITBOX_SLOP)
                .clip(eye, eye.add(dir.scale(REACH)))
                .isPresent();

        Object pack = HeroActions.pack(bot);
        if (pack == null) {
            running = false;
            lastNote = "lost the HeroBot action pack";
            return;
        }

        boolean disc4 = out.disc.length > D_SWAP && out.disc[D_SWAP];
        boolean wantBlock = out.disc.length > D_BLOCK && out.disc[D_BLOCK];
        boolean wantEat = out.disc.length > D_EAT && out.disc[D_EAT];

        lastFwd = out.mu[2];
        lastSprintProb = prob(out, D_SPRINT);
        lastAtkProb = prob(out, D_ATTACK);
        lastBlockProb = prob(out, D_BLOCK);
        lastEatProb = prob(out, D_EAT);
        lastSplProb = Math.max(Math.max(prob(out, D_HEAL), prob(out, D_STR)),
                Math.max(prob(out, D_SPD), prob(out, D_FIR)));

        HeroActions.look(pack, newYaw, newPitch);
        HeroActions.setForward(pack, out.mu[2]);
        HeroActions.setStrafing(pack, out.mu[3]);
        // HeroBot's pack.setSprinting() only stores a flag it never reads
        // back in onUpdate, so drive real sprint state on the player itself.
        // Mirror the sim: sprint needs forward input, and sprint cancels on
        // hit (applied below via shouldCancelSprint).
        boolean wantSprint = out.disc[D_SPRINT] && out.mu[2] > 0.1f;
        bot.setSprinting(wantSprint);
        // Hunger gates vanilla sprint: keep topped up so the bot never walks.
        bot.getFoodData().setFoodLevel(20);
        if (out.disc[D_JUMP]) {
            HeroActions.jump(pack);
        }

        // --- hands: block / swap / eat / splash --------------------------
        // Eating locks the hands in the sim (no attack, no block). Suppress
        // weapon use while a gapple is going down; vanilla stops eating if we
        // swap slots, so don't.
        boolean eatingFood = bot.isUsingItem()
                && (bot.getUseItem().getItem() == Items.GOLDEN_APPLE
                    || bot.getUseItem().getItem() == Items.ENCHANTED_GOLDEN_APPLE);
        boolean usingShield = bot.isUsingItem()
                && bot.getUseItem().getItem() == Items.SHIELD;
        if (wasEatingFood && !eatingFood) {
            lastEatEndTick = tickCount;
        }
        wasEatingFood = eatingFood;

        if (reselectWeapon && !bot.isUsingItem()) {
            selectSlot(pack, swordSlot);
            reselectWeapon = false;
        }
        // failsafe: hands free and holding junk (gapple, pot, pick, cobweb)
        // means a select/rightClick missed somewhere - a pvper always comes
        // home to the sword. Axe stays allowed: it is a real weapon.
        if (!bot.isUsingItem() && !bot.getMainHandItem().is(ItemTags.SWORDS)
                && !bot.getMainHandItem().is(ItemTags.AXES)) {
            selectSlot(pack, swordSlot);
        }

        if (eatingFood) {
            // hands busy: hold still-ish, keep looking, nothing else
        } else {
            if (disc4) {
                selectWeapon(pack, bot);
            }
            int splSlot = governedSplash(bot, out);
            boolean wantHands = (wantEat && eatAllowed(bot)) || splSlot >= 0;
            // one set of hands: food and pots need the shield down first.
            // The sim never shields while eating; splash is instant there,
            // but turtling behind a shield to chug looks broken, so stop.
            if (wantHands && usingShield) {
                bot.stopUsingItem();
                usingShield = false;
            }
            if (wantEat && eatAllowed(bot) && !bot.isUsingItem()) {
                int slot = (bot.getHealth() <= 12.0f && egapSlot >= 0) ? egapSlot : gapSlot;
                // gapples only work from the hotbar (HeroBot can only
                // select slots 0-8); kit() keeps a stack in hotbar.7.
                if (slot >= 0 && slot < Inventory.getSelectionSize()) {
                    selectSlot(pack, slot);
                    if (rightClick(bot, InteractionHand.MAIN_HAND)) {
                        reselectWeapon = true;
                    }
                }
            } else if (wantBlock && blockSuppress <= 0
                    && Observation.shieldOff(bot) <= 0.0f
                    && bot.getOffhandItem().getItem() == Items.SHIELD) {
                if (!bot.isUsingItem()) {
                    rightClick(bot, InteractionHand.OFF_HAND);
                }
            } else if (usingShield) {
                bot.stopUsingItem();
            }
            // splash: instant throw at own feet, then back to the sword
            if (splSlot >= 0 && !bot.isUsingItem()) {
                selectSlot(pack, splSlot);
                if (rightClick(bot, InteractionHand.MAIN_HAND)) {
                    reselectWeapon = true;
                    lastSplashTick = tickCount;
                    blockSuppress = 10;   // open up after potting, don't re-turtle
                }
            }
        }
        wasUsing = bot.isUsingItem();

        boolean hasAxe = bot.getMainHandItem().is(ItemTags.AXES);
        // Sim parity on swing timing: the sim resolves attack as
        // "button edge fires at ANY charge, or holding auto-swings once the
        // cooldown is past 0.9" (sim.py step() attacking). The bridge used to
        // drop every edge below 0.9 - diag now shows ~25% of the policy's
        // attack edges land there, so those darts vanished on the server.
        // ppobot.tap=true restores the sim's edge branch (weak hits pay the
        // vanilla damage floor, exactly like the sim's COOLDOWN_DAMAGE_FLOOR).
        float cd = bot.getAttackStrengthScale(0.5f);
        boolean wantAttack = out.disc[D_ATTACK];
        boolean tapEnabled = Boolean.getBoolean("ppobot.tap");
        boolean tapEdge = tapEnabled && wantAttack && !prevAttack;
        // a raised shield and a swing are mutually exclusive for a human
        boolean shieldUp = bot.isUsingItem() && bot.getUseItem().getItem() == Items.SHIELD;
        boolean attackNow = wantAttack && (cd >= 0.9f || tapEdge)
                && inReach && !eatingFood && !shieldUp;
        prevAttack = wantAttack;
        if (attackNow) {
            HeroActions.attack(pack);
            if (hasAxe) {
                axeCdTicks = 0;
            }
        }
    }

    /** Eat only when there is something to restore, with a gap between gapples. */
    private boolean eatAllowed(ServerPlayer bot) {
        boolean needy = bot.getHealth() < 20.0f
                || bot.getFoodData().getSaturationLevel() < 18.0f;
        return needy && (tickCount - lastEatEndTick) >= 40;
    }

    /**
     * The policy's splash pick, minus the waste: at most one throw per 30
     * ticks, heals only below full HP, buffs only when mostly expired.
     */
    private int governedSplash(ServerPlayer bot, Policy.Out out) {
        if (out.disc.length <= D_FIR) {
            return -1;
        }
        if ((tickCount - lastSplashTick) < 30) {
            return -1;
        }
        // inventory pots are unreachable (HeroBot selects hotbar only)
        if (out.disc[D_HEAL] && healSlot >= 0
                && healSlot < Inventory.getSelectionSize()
                && bot.getHealth() < 20.0f) {
            return healSlot;
        }
        if (out.disc[D_STR] && strSlot >= 0
                && strSlot < Inventory.getSelectionSize()
                && Observation.effectFrac(bot, net.minecraft.world.effect.MobEffects.STRENGTH,
                        Observation.STR_MAX) < 0.9f) {
            return strSlot;
        }
        if (out.disc[D_SPD] && spdSlot >= 0
                && spdSlot < Inventory.getSelectionSize()
                && Observation.effectFrac(bot, net.minecraft.world.effect.MobEffects.SPEED,
                        Observation.SPD_MAX) < 0.9f) {
            return spdSlot;
        }
        if (out.disc[D_FIR] && firSlot >= 0
                && firSlot < Inventory.getSelectionSize()
                && Observation.effectFrac(bot,
                        net.minecraft.world.effect.MobEffects.FIRE_RESISTANCE,
                        Observation.FIR_MAX) < 0.9f) {
            return firSlot;
        }
        // weaving: throw at own feet when stuck in a web, effect mostly out.
        // Old 10-head brains have no D_WEAVE bit: the length check keeps them safe.
        if (out.disc.length > D_WEAVE && out.disc[D_WEAVE] && weaveSlot >= 0
                && weaveSlot < Inventory.getSelectionSize()
                && Observation.inWeb(bot)
                && Observation.weavingFrac(bot) < 0.9f) {
            return weaveSlot;
        }
        return -1;
    }

    private static float prob(Policy.Out out, int d) {
        return d < out.probs.length ? out.probs[d] : 0.0f;
    }

    private static String heldName(ServerPlayer b) {
        ItemStack h = b.getMainHandItem();
        if (h.isEmpty()) {
            return "empty";
        }
        String id = net.minecraft.core.registries.BuiltInRegistries.ITEM.getKey(h.getItem()).toString();
        return id.replace("minecraft:", "");
    }

    private static String usingName(ServerPlayer b) {
        if (!b.isUsingItem()) {
            return "-";
        }
        String id = net.minecraft.core.registries.BuiltInRegistries.ITEM
                .getKey(b.getUseItem().getItem()).toString();
        return id.replace("minecraft:", "") + ":" + b.getUseItemRemainingTicks();
    }
}
