package com.m5.offline;

import java.io.File;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

// Run with app_process on the device, using the official Android M5 engine.
public final class AndroidBenchmark {
    private static int chunk(String name) {
        String value = System.getenv(name);
        return value == null ? 4 : Integer.parseInt(value);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            throw new IllegalArgumentException("usage: model_dir output_dir threads seed text [backbone command...]");
        }
        File model = new File(args[0]);
        File output = new File(args[1]);
        if (!output.isDirectory() && !output.mkdirs()) {
            throw new IllegalArgumentException("cannot create output directory");
        }
        int threads = Integer.parseInt(args[2]);
        long seed = Long.parseLong(args[3]);
        try (M5Engine engine = new M5Engine(model, threads, System.err::println,
                                           Arrays.asList(args).subList(5, args.length),
                                           chunk("M5_FIRST_CHUNK"), chunk("M5_CHUNK"),
                                           System.getenv().getOrDefault("M5_DEPTH_MODEL", "depth_frame.onnx"))) {
            int runs = Integer.parseInt(System.getenv().getOrDefault("M5_RUNS", "4"));
            if (runs < 1) throw new IllegalArgumentException("M5_RUNS must be positive");
            for (int run = 0; run < runs; ++run) {
                File wav = new File(output, "run-" + run + ".wav");
                M5Engine.Synthesis result = engine.synthesize(
                    args[4], 0, seed, wav, new AtomicBoolean(), ignored -> {});
                System.out.printf(Locale.ROOT,
                    "{\"run\":%d,\"elapsed_s\":%.6f,\"audio_s\":%.6f,\"rtf\":%.6f,"
                    + "\"first_pcm_s\":%.6f,\"backbone_s\":%.6f,\"depth_s\":%.6f,\"codec_s\":%.6f}%n",
                    run, result.elapsedSeconds, result.audioSeconds, result.elapsedSeconds / result.audioSeconds,
                    result.firstPcmSeconds, result.backboneSeconds, result.depthSeconds, result.codecSeconds);
            }
            engine.finishProfiles();
        }
    }
}
