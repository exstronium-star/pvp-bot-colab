package com.ppobot;

import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.loader.api.FabricLoader;

import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.phys.Vec3;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * Server-side bridge: the trained PPO policy drives a HeroBot fake player.
 *
 *   /playerspawn Sparring          (or /ppobot spawn Sparring)
 *   /ppobot start Sparring         attach the policy to that bot
 *   /ppobot kit                    survival + plain diamond sword
 *   /ppobot status | stop | reload
 *
 * Weights live in config/ppobot/weights.bin and are written by
 * tools/export_policy.py, so retraining never requires rebuilding the mod.
 */
public class PpoBot implements ModInitializer {
    public static final String MOD_ID = "ppo-bot";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    public static final BotController CONTROLLER = new BotController();

    public static Path weightsPath() {
        return FabricLoader.getInstance().getConfigDir()
                .resolve("ppobot")
                .resolve("weights.bin");
    }

    public static boolean reloadPolicy() {
        Path path = weightsPath();
        try {
            Policy policy = Policy.load(path);
            CONTROLLER.setPolicy(policy);
            LOGGER.info("loaded policy {}x{} from {}", policy.obsDim, policy.hidden, path);
            return true;
        } catch (Exception e) {
            CONTROLLER.setPolicy(null);
            LOGGER.error("could not load policy from {}: {}", path, e.toString());
            return false;
        }
    }

    @Override
    public void onInitialize() {
        ServerTickEvents.END_SERVER_TICK.register(CONTROLLER::tick);

        boolean hero = HeroActions.init();
        if (!hero) {
            LOGGER.warn("HeroBot not found yet ({}); commands will still register",
                    HeroActions.problem());
        }
        reloadPolicy();

        CommandRegistrationCallback.EVENT.register((dispatcher, registry, environment) ->
                dispatcher.register(Commands.literal("ppobot")
                        .requires(Commands.hasPermission(Commands.LEVEL_GAMEMASTERS))

                        .then(Commands.literal("reload").executes(ctx -> {
                            boolean ok = reloadPolicy();
                            ctx.getSource().sendSuccess(() -> Component.literal(
                                    ok ? "policy reloaded: " + weightsPath()
                                       : "reload failed, see server log"), false);
                            return ok ? 1 : 0;
                        }))

                        .then(Commands.literal("stop").executes(ctx -> {
                            CONTROLLER.stop(ctx.getSource().getServer());
                            ctx.getSource().sendSuccess(
                                    () -> Component.literal("ppobot stopped"), false);
                            return 1;
                        }))

                        .then(Commands.literal("status").executes(ctx -> {
                            String s = CONTROLLER.status(ctx.getSource().getServer());
                            ctx.getSource().sendSuccess(() -> Component.literal(s), false);
                            return 1;
                        }))

                        .then(Commands.literal("kit").executes(ctx -> {
                            boolean ok = CONTROLLER.kit(ctx.getSource().getServer());
                            ctx.getSource().sendSuccess(
                                    () -> Component.literal(CONTROLLER.lastNote()), false);
                            return ok ? 1 : 0;
                        }))

                        .then(Commands.literal("spawn")
                                .then(Commands.argument("name", StringArgumentType.word())
                                        .executes(ctx -> doSpawn(ctx,
                                                StringArgumentType.getString(ctx, "name"),
                                                senderName(ctx)))
                                        .then(Commands.argument("target", StringArgumentType.word())
                                                .executes(ctx -> doSpawn(ctx,
                                                        StringArgumentType.getString(ctx, "name"),
                                                        StringArgumentType.getString(ctx, "target"))))))

                        .then(Commands.literal("start")
                                .then(Commands.argument("name", StringArgumentType.word())
                                        .executes(ctx -> doStart(ctx,
                                                StringArgumentType.getString(ctx, "name"),
                                                senderName(ctx)))
                                        .then(Commands.argument("target", StringArgumentType.word())
                                                .executes(ctx -> doStart(ctx,
                                                        StringArgumentType.getString(ctx, "name"),
                                                        StringArgumentType.getString(ctx, "target"))))))));

        LOGGER.info("ppo-bot ready: /ppobot spawn <name> [target] then /ppobot kit");
    }

    /** Sender's own name when the command comes from a player, else requires an explicit target. */
    private static String senderName(CommandContext<CommandSourceStack> ctx) {
        return ctx.getSource().getEntity() instanceof ServerPlayer p ? p.getGameProfile().name() : "";
    }

    /** Spawn the bot with `playerspawn` at the sender (player) or world spawn (console), and arm the auto-start. */
    private static int doSpawn(CommandContext<CommandSourceStack> ctx, String name, String target) {
        if (target.isEmpty()) {
            ctx.getSource().sendFailure(Component.literal(
                    "need an online player or an explicit target: /ppobot spawn <name> <target>"));
            return 0;
        }
        CommandSourceStack src = botSource(ctx.getSource());
        src.getServer().getCommands().performPrefixedCommand(src, "playerspawn " + name);
        CONTROLLER.pending(name, target);
        ctx.getSource().sendSuccess(() -> Component.literal(
                "spawning " + name + " (target " + target + ")..."), false);
        return 1;
    }

    /** Attach the policy to an already-spawned bot. */
    private static int doStart(CommandContext<CommandSourceStack> ctx, String name, String target) {
        if (target.isEmpty()) {
            ctx.getSource().sendFailure(Component.literal(
                    "need an online player or an explicit target: /ppobot start <name> <target>"));
            return 0;
        }
        boolean ok = CONTROLLER.start(ctx.getSource().getServer(), name, target);
        ctx.getSource().sendSuccess(() -> Component.literal(CONTROLLER.lastNote()), false);
        return ok ? 1 : 0;
    }

    /** Bot spawn position: the sender's feet when sent by a player, else world spawn (console). */
    private static CommandSourceStack botSource(CommandSourceStack src) {
        MinecraftServer server = src.getServer();
        if (src.getEntity() instanceof ServerPlayer self) {
            return src.withPosition(self.position()).withSuppressedOutput();
        }
        BlockPos spawn = server.getRespawnData().pos();
        return src.withPosition(Vec3.atCenterOf(spawn)).withSuppressedOutput();
    }
}
