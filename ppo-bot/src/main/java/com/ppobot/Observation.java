package com.ppobot;

import net.minecraft.core.component.DataComponents;
import net.minecraft.core.Holder;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.effect.MobEffect;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.AxeItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.alchemy.PotionContents;
import net.minecraft.world.phys.Vec3;

/**
 * Builds the exact 49-float kit observation the sim trains on.
 *
 * Features 0-25 are the old melee vector. 26-40 are own kit state
 * (saturation, absorption, axe out, axe cooldown, shield, buff timers,
 * consumable counts, eating), 41-45 are the opponent's readable kit state,
 * 46-48 are the phase-2 web reads (own stuck, enemy weaving time, weave pots).
 * Keep this in lockstep with mc_pvp/sim.py:obs() - a reordered or
 * rescaled feature here silently lobotomises the policy.
 */
public final class Observation {
    public static final int DIM = 49;

    /** STR_DURATION = 20*90, SPD/FIR_DURATION = 20*480 in sim.py. WEAVE 60s. */
    public static final float STR_MAX = 1800.0f;
    public static final float SPD_MAX = 9600.0f;
    public static final float FIR_MAX = 9600.0f;
    public static final float WEAVE_MAX = 1200.0f;

    private Observation() {
    }

    public static float[] build(ServerPlayer bot, ServerPlayer target, double floorY,
                                int sinceHit, int sinceHurt,
                                float axeCd01, int nHeal, int nStr) {
        Vec3 v = bot.getDeltaMovement();
        Vec3 ov = target.getDeltaMovement();

        double dx = target.getX() - bot.getX();
        double dz = target.getZ() - bot.getZ();
        // Chest (target feet + 1.2) vs eye (bot feet + 1.62), exactly like sim.py:obs().
        // The old feet-vs-feet version aimed ~8 deg high at 3 blocks, which is
        // why the bot only connected when the target jumped into the ray.
        double dy = (target.getY() + 1.2) - (bot.getY() + 1.62);
        double dist = Math.hypot(dx, dz);

        double yaw = Math.toRadians(bot.getYRot());
        double fx = -Math.sin(yaw), fz = Math.cos(yaw);      // forward
        double rx = -Math.cos(yaw), rz = -Math.sin(yaw);     // right

        double relF = dx * fx + dz * fz;
        double relR = dx * rx + dz * rz;

        double aimYaw = Math.toDegrees(Math.atan2(relR, relF)) / 180.0;
        double wantPitch = Math.toDegrees(-Math.atan2(dy, Math.max(dist, 1e-6)));
        double aimPitch = clamp((wantPitch - bot.getXRot()) / 45.0, -1.0, 1.0);

        double selfF = v.x * fx + v.z * fz;
        double selfR = v.x * rx + v.z * rz;
        double relVf = (ov.x - v.x) * fx + (ov.z - v.z) * fz;
        double relVr = (ov.x - v.x) * rx + (ov.z - v.z) * rz;

        double height = Math.max(0.0, bot.getY() - floorY);

        float[] o = new float[DIM];
        int i = 0;
        o[i++] = (float) (v.x * 5.0);
        o[i++] = (float) (v.z * 5.0);
        o[i++] = (float) (v.y * 5.0);
        o[i++] = (float) (height / 2.0);
        o[i++] = bot.onGround() ? 1.0f : 0.0f;
        // 0.0 adjust ticks: the sim feeds raw cooldown/COOLDOWN_TICKS, and the
        // 0.5 the client uses for the attack indicator would read ~4% high.
        o[i++] = bot.getAttackStrengthScale(0.0f);
        o[i++] = bot.getHealth() / 20.0f;
        o[i++] = bot.hurtTime / 10.0f;
        o[i++] = bot.isSprinting() ? 1.0f : 0.0f;

        o[i++] = (float) (relF / 5.0);
        o[i++] = (float) (relR / 5.0);
        o[i++] = (float) (dist / 5.0);
        o[i++] = (float) aimYaw;
        o[i++] = (float) aimPitch;
        o[i++] = (float) clamp((dist - 3.0) / 5.0, -1.0, 1.0);

        o[i++] = target.getAttackStrengthScale(0.0f);
        o[i++] = target.getHealth() / 20.0f;
        o[i++] = target.onGround() ? 1.0f : 0.0f;
        o[i++] = target.isSprinting() ? 1.0f : 0.0f;

        o[i++] = (float) (selfF * 5.0);
        o[i++] = (float) (selfR * 5.0);
        o[i++] = (float) (relVf * 5.0);
        o[i++] = (float) (relVr * 5.0);

        o[i++] = (float) clamp(sinceHit / 40.0, 0.0, 1.0);
        o[i++] = (float) clamp(sinceHurt / 40.0, 0.0, 1.0);

        o[i++] = 1.0f;   // arena-wall margin: no walls in the real world

        // --- own kit state (26-40) ---
        o[i++] = bot.getFoodData().getSaturationLevel() / 20.0f;
        o[i++] = (float) clamp(bot.getAbsorptionAmount() / 8.0, 0.0, 1.0);
        o[i++] = bot.getMainHandItem().is(ItemTags.AXES) ? 1.0f : 0.0f;
        o[i++] = clamp(axeCd01, 0.0f, 1.0f);
        o[i++] = bot.isBlocking() ? 1.0f : 0.0f;
        o[i++] = shieldOff(bot);
        o[i++] = effectFrac(bot, MobEffects.STRENGTH, STR_MAX);
        o[i++] = effectFrac(bot, MobEffects.SPEED, SPD_MAX);
        o[i++] = effectFrac(bot, MobEffects.FIRE_RESISTANCE, FIR_MAX);
        o[i++] = weavingFrac(bot);
        o[i++] = (float) clamp(countItem(bot, Items.GOLDEN_APPLE) / 128.0, 0.0, 1.0);
        o[i++] = (float) clamp(countItem(bot, Items.ENCHANTED_GOLDEN_APPLE) / 2.0, 0.0, 1.0);
        o[i++] = (float) clamp(nHeal / 4.0, 0.0, 1.0);
        o[i++] = (float) clamp(nStr / 9.0, 0.0, 1.0);
        o[i++] = eatingFrac(bot);

        // --- opponent kit reads (41-45) ---
        o[i++] = target.isBlocking() ? 1.0f : 0.0f;
        o[i++] = shieldOff(target);
        o[i++] = target.getMainHandItem().is(ItemTags.AXES) ? 1.0f : 0.0f;
        o[i++] = effectFrac(target, MobEffects.STRENGTH, STR_MAX);
        o[i++] = (float) clamp(target.getAbsorptionAmount() / 8.0, 0.0, 1.0);
        // --- web reads (46-48): own stuck, enemy weaving time, weave pots ---
        o[i++] = inWeb(bot) ? 1.0f : 0.0f;
        o[i++] = weavingFrac(target);
        o[i++] = (float) clamp(countWeave(bot) / 3.0, 0.0, 1.0);
        return o;
    }

    /** Shield-disable cooldown as a 0..1 fraction (vanilla puts the shield item on cooldown). */
    public static float shieldOff(ServerPlayer p) {
        ItemStack off = p.getOffhandItem();
        if (off.getItem() != Items.SHIELD) {
            return 0.0f;
        }
        return p.getCooldowns().getCooldownPercent(off, 0.0f);
    }

    public static float effectFrac(ServerPlayer p, Holder<MobEffect> eff, float max) {
        MobEffectInstance inst = p.getEffect(eff);
        if (inst == null) {
            return 0.0f;
        }
        return (float) clamp(inst.getDuration() / (double) max, 0.0, 1.0);
    }

    /** Weaving time 0..1 (60s max in sim). Looked up by id so a missing
     * registry entry degrades to 0 instead of breaking the build. */
    public static Holder<MobEffect> weavingHolder() {
        try {
            var key = net.minecraft.resources.ResourceLocation.parse("minecraft:weaving");
            var got = net.minecraft.core.registries.BuiltInRegistries.MOB_EFFECT.get(key);
            if (got.isPresent()) {
                return net.minecraft.core.registries.BuiltInRegistries.MOB_EFFECT.wrapAsHolder(got.get());
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    public static float weavingFrac(ServerPlayer p) {
        Holder<MobEffect> h = weavingHolder();
        if (h == null) {
            return 0.0f;
        }
        return effectFrac(p, h, WEAVE_MAX);
    }

    /** Eating progress 1..0 while a gapple is going down, else 0. */
    public static boolean inWeb(ServerPlayer p) {
        // cobweb slows when the feet box overlaps it: sample the block at
        // feet and at waist. Cheap, tick-safe, no chunk loads (loaded already).
        var level = p.level();
        var feet = p.blockPosition();
        if (level.getBlockState(feet).is(net.minecraft.world.level.block.Blocks.COBWEB)) {
            return true;
        }
        return level.getBlockState(feet.above()).is(net.minecraft.world.level.block.Blocks.COBWEB);
    }

    public static int countWeave(ServerPlayer p) {
        int n = 0;
        Inventory inv = p.getInventory();
        for (int s = 0; s < inv.getContainerSize(); s++) {
            ItemStack st = inv.getItem(s);
            if (st.getItem() == Items.SPLASH_POTION
                    && "weave".equals(splashKind(st))) {
                n += st.getCount();
            }
        }
        return n;
    }
    public static float eatingFrac(ServerPlayer p) {
        if (!p.isUsingItem()) {
            return 0.0f;
        }
        ItemStack use = p.getUseItem();
        if (use.getItem() != Items.GOLDEN_APPLE
                && use.getItem() != Items.ENCHANTED_GOLDEN_APPLE) {
            return 0.0f;
        }
        return p.getUseItemRemainingTicks() / 32.0f;
    }

    public static int countItem(ServerPlayer p, net.minecraft.world.item.Item item) {
        int n = 0;
        Inventory inv = p.getInventory();
        for (int s = 0; s < inv.getContainerSize(); s++) {
            ItemStack st = inv.getItem(s);
            if (st.getItem() == item) {
                n += st.getCount();
            }
        }
        if (p.getOffhandItem().getItem() == item) {
            n += p.getOffhandItem().getCount();
        }
        return n;
    }

    /**
     * Classify a splash potion by its effects. Returns "heal", "str", "spd",
     * "fir", "weave", or null when it is not a recognised kit splash.
     */
    public static String splashKind(ItemStack s) {
        if (s.getItem() != Items.SPLASH_POTION) {
            return null;
        }
        PotionContents c = s.get(DataComponents.POTION_CONTENTS);
        if (c == null) {
            return null;
        }
        MobEffect strength = MobEffects.STRENGTH.value();
        MobEffect speed = MobEffects.SPEED.value();
        MobEffect fire = MobEffects.FIRE_RESISTANCE.value();
        for (MobEffectInstance inst : c.getAllEffects()) {
            MobEffect e = inst.getEffect().value();
            if (e == strength) {
                return "str";
            }
            if (e == speed) {
                return "spd";
            }
            if (e == fire) {
                return "fir";
            }
            // Instant Health has no duration barcode: match by id string.
            String id = inst.getDescriptionId();
            if (id.contains("heal")) {
                return "heal";
            }
            // Weaving (trial-chamber movement): id match, survives renames.
            if (id.contains("weaving")) {
                return "weave";
            }
        }
        return null;
    }

    public static double clamp(double x, double lo, double hi) {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    public static float clamp(float x, float lo, float hi) {
        return x < lo ? lo : (x > hi ? hi : x);
    }
}
