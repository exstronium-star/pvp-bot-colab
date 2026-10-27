package com.ppobot;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The trained PPO policy, as a plain MLP.
 *
 * <p>The sword-only brain is 26 -&gt; 128 -&gt; 128 with 4 continuous heads and
 * 3 Bernoulli heads (jump, sprint, attack). The kit brain is 49 -&gt; 128
 * -&gt; 128 with 11 Bernoulli heads (jump, sprint, attack, block, swap, eat,
 * splash heal/str/spd/fir/weave). Head widths come from the weight file, so the
 * controller must map them by name, never by index constant here.
 */
public final class Policy {
    public static final int CONT = 4;

    public final int obsDim;
    public final int hidden;
    public final int disc;

    private final float[] w1, b1, w2, b2, wmu, bmu, wlog, blog;

    private Policy(int obsDim, int hidden, int disc,
                   float[] w1, float[] b1, float[] w2, float[] b2,
                   float[] wmu, float[] bmu, float[] wlog, float[] blog) {
        this.obsDim = obsDim;
        this.hidden = hidden;
        this.disc = disc;
        this.w1 = w1;
        this.b1 = b1;
        this.w2 = w2;
        this.b2 = b2;
        this.wmu = wmu;
        this.bmu = bmu;
        this.wlog = wlog;
        this.blog = blog;
    }

    public static final class Out {
        public final float[] mu = new float[CONT];
        public final float[] probs;
        public final boolean[] disc;

        Out(int discDim) {
            probs = new float[discDim];
            disc = new boolean[discDim];
        }
    }

    public static Policy load(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

        byte[] magic = new byte[4];
        bb.get(magic);
        if (!new String(magic, StandardCharsets.US_ASCII).equals("PPOB")) {
            throw new IOException("not a PPOB weight file: " + path);
        }
        int version = bb.getInt();
        if (version != 1) {
            throw new IOException("unsupported weight version " + version);
        }
        int obs = bb.getInt();
        int hidden = bb.getInt();
        int cont = bb.getInt();
        int disc = bb.getInt();
        if (cont != CONT) {
            throw new IOException("policy cont mismatch: " + cont);
        }
        if (obs != Observation.DIM) {
            throw new IOException("policy obs mismatch: file has " + obs
                    + " but this build expects " + Observation.DIM
                    + " (export from the matching sim before deploying)");
        }
        if (disc != 10 && disc != 11) {
            // 10 = pre-weave brain, 11 = current. Both load; the controller
            // length-checks D_WEAVE so old brains just never weave.
            throw new IOException("policy disc mismatch: " + disc
                    + " (want 10 or 11)");
        }

        float[] w1 = read(bb, hidden * obs);
        float[] b1 = read(bb, hidden);
        float[] w2 = read(bb, hidden * hidden);
        float[] b2 = read(bb, hidden);
        float[] wmu = read(bb, cont * hidden);
        float[] bmu = read(bb, cont);
        float[] wlog = read(bb, disc * hidden);
        float[] blog = read(bb, disc);

        return new Policy(obs, hidden, disc, w1, b1, w2, b2, wmu, bmu, wlog, blog);
    }

    private static float[] read(ByteBuffer bb, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = bb.getFloat();
        }
        return a;
    }

    public Out forward(float[] x) {
        if (x.length != obsDim) {
            throw new IllegalArgumentException("expected " + obsDim + " obs, got " + x.length);
        }

        float[] h = new float[hidden];
        for (int i = 0; i < hidden; i++) {
            h[i] = (float) Math.tanh(dot(w1, i * obsDim, x, obsDim) + b1[i]);
        }

        float[] h2 = new float[hidden];
        for (int i = 0; i < hidden; i++) {
            h2[i] = (float) Math.tanh(dot(w2, i * hidden, h, hidden) + b2[i]);
        }

        Out out = new Out(disc);
        for (int c = 0; c < CONT; c++) {
            out.mu[c] = (float) Math.tanh(dot(wmu, c * hidden, h2, hidden) + bmu[c]);
        }
        for (int d = 0; d < disc; d++) {
            float logit = dot(wlog, d * hidden, h2, hidden) + blog[d];
            float p = (float) (1.0 / (1.0 + Math.exp(-logit)));
            out.probs[d] = p;
            out.disc[d] = p > 0.5f;
        }
        return out;
    }

    private static float dot(float[] w, int off, float[] x, int n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += w[off + i] * x[i];
        }
        return sum;
    }
}
