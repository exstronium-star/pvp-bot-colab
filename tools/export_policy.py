"""Export a PPO checkpoint to the flat binary the Fabric bridge mod reads.

The bridge cannot run PyTorch, and the policy is a 26 -> 128 -> 128 MLP, so we
ship the weights and do the forward pass in plain Java. This keeps the deployed
bot a fake player with no Python process and no socket in the tick loop.

    python tools/export_policy.py                      # runs/ppo.pt -> config/ppobot/weights.bin
    python tools/export_policy.py --ckpt runs/ppo.pt --out somewhere/weights.bin

Format (little-endian):
    magic   char[4]  "PPOB"
    version uint32   1
    obs     uint32
    hidden  uint32
    cont    uint32
    disc    uint32
    w1      f32[hidden*obs]      row-major, out x in
    b1      f32[hidden]
    w2      f32[hidden*hidden]   row-major
    b2      f32[hidden]
    w_mu    f32[cont*hidden]
    b_mu    f32[cont]
    w_log   f32[disc*hidden]
    b_log   f32[disc]
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

MAGIC = b"PPOB"
VERSION = 1


def load_arrays(ckpt: str) -> dict[str, np.ndarray]:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    need = ["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
            "mu.weight", "mu.bias", "logits.weight", "logits.bias"]
    missing = [k for k in need if k not in sd]
    if missing:
        raise SystemExit(f"checkpoint is missing {missing}; got {sorted(sd)}")
    return {k: sd[k].detach().numpy().astype(np.float32) for k in need}


def reference_forward(w: dict[str, np.ndarray], obs: np.ndarray):
    """Exactly what the Java side must reproduce - used as a parity check."""
    h = np.tanh(obs @ w["trunk.0.weight"].T + w["trunk.0.bias"])
    h = np.tanh(h @ w["trunk.2.weight"].T + w["trunk.2.bias"])
    mu = np.tanh(h @ w["mu.weight"].T + w["mu.bias"])
    logits = h @ w["logits.weight"].T + w["logits.bias"]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return mu, logits, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ppo.pt")
    ap.add_argument("--out", default="ppo-bot/run/config/ppobot/weights.bin")
    ap.add_argument("--copy-to", default=None,
                    help="also write a copy here, e.g. the live server's "
                         "config/ppobot/weights.bin")
    args = ap.parse_args()

    w = load_arrays(args.ckpt)
    hidden, obs_dim = w["trunk.0.weight"].shape
    cont, _ = w["mu.weight"].shape
    disc, _ = w["logits.weight"].shape

    order = ["trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
             "mu.weight", "mu.bias", "logits.weight", "logits.bias"]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<5I", VERSION, obs_dim, hidden, cont, disc))
        for k in order:
            f.write(w[k].tobytes(order="C"))

    # parity check against the torch policy (widened the same way PPO.load does,
    # so old 46x10 checkpoints verify against the current 49x11 sim)
    from mc_pvp.ppo import PPO
    net = PPO.load(args.ckpt, obs_dim)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 0.5, (4, obs_dim)).astype(np.float32)
    mu, logits, probs = reference_forward(w, obs)
    with torch.no_grad():
        t_mu, _, t_logits, _ = net.forward(torch.as_tensor(obs))
    # old checkpoints have fewer heads: compare the overlapping prefix
    d_old = logits.shape[1]
    t_logits = t_logits.numpy()[:, :d_old]
    t_probs = 1.0 / (1.0 + np.exp(-t_logits))
    print(f"obs_dim {obs_dim}  hidden {hidden}  cont {cont}  disc {disc}")
    print(f"mu max abs error      {np.abs(mu - t_mu.numpy()).max():.2e}")
    print(f"logit max abs error   {np.abs(logits - t_logits).max():.2e}")
    print(f"attack prob (sample)  {t_probs[:, 2].round(2)}")
    print(f"exported heads: obs {obs_dim} disc {d_old} (bridge accepts 10 or 11)")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")

    if args.copy_to:
        dst = pathlib.Path(args.copy_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(out.read_bytes())
        print(f"copied to {dst}")


if __name__ == "__main__":
    main()
