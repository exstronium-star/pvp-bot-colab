package com.ppobot;

import net.minecraft.server.level.ServerPlayer;

import java.lang.reflect.Method;

/**
 * Thin reflective adapter over HeroBot's BotPlayerActionPack.
 *
 * We deliberately avoid a compile-time HeroBot dependency: the mod then builds
 * without the HeroBot jar on the classpath, and only needs it at runtime. Every
 * method used here is part of the pack's public surface:
 *
 *   getActionPack()          (ServerPlayerInterface, mixed into ServerPlayer)
 *   setForward / setStrafing / setSprinting / setSlot
 *   look(float yaw, float pitch)
 *   start(ActionType, Action)          ActionType.ATTACK / JUMP
 *   Action.once()
 */
public final class HeroActions {
    private static Method getActionPack;
    private static Method setForward;
    private static Method setStrafing;
    private static Method setSprinting;
    private static Method setSlot;
    private static Method look;
    private static Method start;
    private static Method once;
    private static Method stopAll;

    private static Object attackType;
    private static Object jumpType;

    private static boolean ready;
    private static String problem = "HeroBot adapter not initialised";

    private HeroActions() {
    }

    public static synchronized boolean init() {
        if (ready) {
            return true;
        }
        try {
            Class<?> iface = Class.forName("hero.bane.herobot.bot.connection.ServerPlayerInterface");
            Class<?> pack = Class.forName("hero.bane.herobot.bot.BotPlayerActionPack");
            Class<?> type = Class.forName("hero.bane.herobot.bot.BotPlayerActionPack$ActionType");
            Class<?> action = Class.forName("hero.bane.herobot.bot.BotPlayerActionPack$Action");

            getActionPack = iface.getMethod("getActionPack");
            setForward = pack.getMethod("setForward", float.class);
            setStrafing = pack.getMethod("setStrafing", float.class);
            setSprinting = pack.getMethod("setSprinting", boolean.class);
            setSlot = pack.getMethod("setSlot", int.class);
            look = pack.getMethod("look", float.class, float.class);
            start = pack.getMethod("start", type, action);
            once = action.getMethod("once");
            stopAll = pack.getMethod("stopAll");

            attackType = type.getField("ATTACK").get(null);
            jumpType = type.getField("JUMP").get(null);

            ready = true;
            problem = null;
            return true;
        } catch (Throwable t) {
            problem = t.getClass().getSimpleName() + ": " + t.getMessage();
            ready = false;
            return false;
        }
    }

    public static String problem() {
        return problem;
    }

    public static boolean ready() {
        return ready;
    }

    public static Object pack(ServerPlayer player) {
        if (!ready) {
            return null;
        }
        try {
            return getActionPack.invoke(player);
        } catch (Throwable t) {
            fail("getActionPack", t);
            return null;
        }
    }

    public static void setForward(Object pack, float v) {
        call(setForward, pack, v);
    }

    public static void setStrafing(Object pack, float v) {
        call(setStrafing, pack, v);
    }

    public static void setSprinting(Object pack, boolean v) {
        call(setSprinting, pack, v);
    }

    public static void setSlot(Object pack, int v) {
        // HeroBot's setSlot is carpet-style 1-based (bytecode does
        // setSelectedSlot(slot - 1)); our codebase is 0-based everywhere.
        call(setSlot, pack, v + 1);
    }

    public static void look(Object pack, float yaw, float pitch) {
        call(look, pack, yaw, pitch);
    }

    public static void attack(Object pack) {
        action(pack, attackType);
    }

    public static void jump(Object pack) {
        action(pack, jumpType);
    }

    public static void stopAll(Object pack) {
        call(stopAll, pack);
    }

    private static void action(Object pack, Object type) {
        if (!ready || pack == null) {
            return;
        }
        try {
            Object a = once.invoke(null);
            start.invoke(pack, type, a);
        } catch (Throwable t) {
            fail("start", t);
        }
    }

    private static void call(Method m, Object pack, Object... args) {
        if (!ready || pack == null) {
            return;
        }
        try {
            m.invoke(pack, args);
        } catch (Throwable t) {
            fail(m.getName(), t);
        }
    }

    private static void fail(String what, Throwable t) {
        if (problem == null) {
            problem = what + " failed: " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
    }
}
