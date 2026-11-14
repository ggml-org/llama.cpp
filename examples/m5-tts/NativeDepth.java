package com.m5.offline;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

final class NativeDepth implements Closeable {
    private static final int WIDTH = 512, VOCAB = 2051, BOOKS = 16;
    private final NativeBackbone core;
    private final ByteBuffer mappedWeights;
    private final FloatBuffer weights;
    private final float[] input = new float[WIDTH];
    private final float[] projectedHidden = new float[WIDTH];
    private static native void project(ByteBuffer weights, int offset, float[] hidden, float[] output);
    static { System.loadLibrary("m5_depth_heads"); }
    private final FloatBuffer local = FloatBuffer.allocate(2 * WIDTH);
    private final FloatBuffer decoded = FloatBuffer.allocate(WIDTH);
    private final float[] logits = new float[VOCAB];
    private final float[] ranked = new float[VOCAB];

    NativeDepth(File weightsFile, List<String> command) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(weightsFile, "r")) {
            long bytes = (BOOKS * VOCAB * WIDTH * 3L + WIDTH * WIDTH) * Float.BYTES;
            if (file.length() != bytes) throw new IOException("Invalid depth weight size");
            mappedWeights = file.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, bytes).order(ByteOrder.LITTLE_ENDIAN);
            weights = mappedWeights.asFloatBuffer();
        }
        core = new NativeBackbone(command);
    }

    private int sample(FloatBuffer hidden, FloatBuffer noise, int book, boolean allowEos) {
        hidden.duplicate().get(input);
        project(mappedWeights, book * VOCAB * WIDTH, input, logits);
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < VOCAB; i++) {
            logits[i] = (i == 0 && !allowEos ? -1e5f : logits[i]) / 0.9f;
            max = Math.max(max, logits[i]);
        }
        System.arraycopy(logits, 0, ranked, 0, VOCAB);
        Arrays.sort(ranked);
        float cutoff = ranked[VOCAB - 10];
        float sum = 0;
        for (int i = 0; i < VOCAB; i++) {
            logits[i] = logits[i] >= cutoff ? (float) Math.exp(logits[i] - max) : 0;
            sum += logits[i];
        }
        int selected = 0;
        float best = -1;
        for (int i = 0; i < VOCAB; i++) {
            float score = (logits[i] / sum) / noise.get(book * VOCAB + i);
            if (score > best) {
                best = score;
                selected = i;
            }
        }
        return selected;
    }

    long[] run(FloatBuffer hidden, FloatBuffer noise, boolean allowEos, FloatBuffer next) throws IOException {
        final int projection = BOOKS * VOCAB * WIDTH;
        final int projected = projection + WIDTH * WIDTH;
        final int audio = projected + BOOKS * VOCAB * WIDTH;
        core.reset();
        long[] codes = new long[BOOKS];
        codes[0] = sample(hidden, noise, 0, allowEos);
        local.clear();
        project(mappedWeights, projection, input, projectedHidden);
        local.put(projectedHidden);
        for (int book = 1; book < BOOKS; book++) {
            int offset = projected + ((book - 1) * VOCAB + (int) codes[book - 1]) * WIDTH;
            for (int j = 0; j < WIDTH; j++) local.put(weights.get(offset + j));
            local.flip();
            core.run(local, decoded);
            codes[book] = sample(decoded, noise, book, true);
            local.clear();
        }
        next.clear();
        for (int j = 0; j < WIDTH; j++) {
            float total = 0;
            for (int book = 0; book < BOOKS; book++) {
                total += weights.get(audio + (book * VOCAB + (int) codes[book]) * WIDTH + j);
            }
            next.put(total);
        }
        next.flip();
        return codes;
    }

    public void close() throws IOException { core.close(); }
}
