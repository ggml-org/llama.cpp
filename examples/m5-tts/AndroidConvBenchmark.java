package com.m5.offline;

import ai.onnxruntime.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

public final class AndroidConvBenchmark {
    public static void main(String[] args) throws Exception {
        byte[] data = Files.readAllBytes(Paths.get(args[1]));
        FloatBuffer input = ByteBuffer.allocateDirect(data.length).order(ByteOrder.nativeOrder()).asFloatBuffer();
        input.put(ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()).rewind();
        FloatBuffer expected = ByteBuffer.wrap(Files.readAllBytes(Paths.get(args[2])))
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(4);
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(args[0], options);
                 OnnxTensor tensor = OnnxTensor.createTensor(env, input, new long[]{1, 384, 1286})) {
                double[] times = new double[8];
                float maxError = 0;
                for (int i = 0; i <= times.length; i++) {
                    long start = System.nanoTime();
                    try (OrtSession.Result result = session.run(Collections.singletonMap("x", tensor))) {
                        FloatBuffer output = ((OnnxTensor) result.get(0)).getFloatBuffer();
                        if (i > 0) {
                            times[i - 1] = (System.nanoTime() - start) / 1e6;
                        } else {
                            for (int c = 0; c < 384; c++) {
                                for (int row = 0; row < 1280; row++) {
                                    float value = output.get(c * 1280 + row);
                                    if (!Float.isFinite(value)) throw new IllegalStateException("nonfinite output");
                                    maxError = Math.max(maxError, Math.abs(value - expected.get(row * 384 + c)));
                                }
                            }
                        }
                    }
                }
                Arrays.sort(times);
                System.out.printf(Locale.ROOT, "ORT median_ms=%.3f max_abs_error=%.8g%n",
                                  (times[3] + times[4]) / 2, maxError);
            }
        }
    }
}
