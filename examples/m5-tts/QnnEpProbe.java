import java.lang.reflect.Method;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

// Reflection-only ORT probe: compiles with JDK 8 and runs via app_process.
// usage: QnnEpProbe <model> <cpu|nnapi|xnnpack|qnn> [reps] [input] [shape,csv]
public final class QnnEpProbe {
    public static void main(String[] args) throws Exception {
        String model = args[0];
        String provider = args.length > 1 ? args[1] : "cpu";
        int reps = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        String input = args.length > 3 ? args[3] : "x";
        String[] dims = (args.length > 4 ? args[4] : "1,384,134").split(",");
        long[] shape = new long[dims.length];
        int count = 1;
        for (int i = 0; i < dims.length; ++i) {
            shape[i] = Long.parseLong(dims[i]);
            count *= shape[i];
        }

        Class<?> envCls = Class.forName("ai.onnxruntime.OrtEnvironment");
        System.out.println("AVAILABLE_PROVIDERS=" + envCls.getMethod("getAvailableProviders").invoke(null));
        Object env = envCls.getMethod("getEnvironment").invoke(null);
        Class<?> optCls = Class.forName("ai.onnxruntime.OrtSession$SessionOptions");
        Object opts = optCls.getConstructor().newInstance();
        optCls.getMethod("setIntraOpNumThreads", int.class).invoke(opts, 4);
        optCls.getMethod("setInterOpNumThreads", int.class).invoke(opts, 1);

        if (provider.equals("qnn")) {
            Map<String, String> cfg = new HashMap<String, String>();
            cfg.put("backend_path", "libQnnHtp.so");
            cfg.put("htp_performance_mode", "burst");
            optCls.getMethod("addQnn", Map.class).invoke(opts, cfg);
        } else if (provider.equals("nnapi")) {
            optCls.getMethod("addNnapi").invoke(opts);
        } else if (provider.equals("xnnpack")) {
            Map<String, String> cfg = new HashMap<String, String>();
            cfg.put("intra_op_num_threads", "4");
            optCls.getMethod("addXnnpack", Map.class).invoke(opts, cfg);
        }

        Method create = null;
        for (Method m : envCls.getMethods()) {
            Class<?>[] p = m.getParameterTypes();
            if (m.getName().equals("createSession") && p.length == 2
                    && p[0] == String.class && p[1] == optCls) { create = m; break; }
        }
        long t0 = System.nanoTime();
        Object session = create.invoke(env, model, opts);
        System.out.println("SESSION_CREATE_MS=" + (System.nanoTime() - t0) / 1e6);
        System.out.println("INPUTS=" + session.getClass().getMethod("getInputNames").invoke(session));

        Class<?> tensorCls = Class.forName("ai.onnxruntime.OnnxTensor");
        Method createTensor = tensorCls.getMethod("createTensor", envCls, FloatBuffer.class, long[].class);
        float[] data = new float[count];
        Random r = new Random(1234);
        for (int i = 0; i < data.length; ++i) data[i] = (float) (r.nextGaussian() * 0.5);
        Map<String, Object> feed = new HashMap<String, Object>();
        feed.put(input, createTensor.invoke(null, env, FloatBuffer.wrap(data), shape));

        Method run = null;
        for (Method m : session.getClass().getMethods()) {
            Class<?>[] p = m.getParameterTypes();
            if (m.getName().equals("run") && p.length == 1 && p[0] == Map.class) { run = m; break; }
        }
        Method close = Class.forName("ai.onnxruntime.OrtSession$Result").getMethod("close");
        for (int i = 0; i < 2; ++i) close.invoke(run.invoke(session, feed));

        double total = 0.0;
        for (int i = 0; i < reps; ++i) {
            long t = System.nanoTime();
            Object res = run.invoke(session, feed);
            double ms = (System.nanoTime() - t) / 1e6;
            close.invoke(res);
            total += ms;
            System.out.println("RUN_MS=" + String.format(Locale.ROOT, "%.3f", ms));
        }
        System.out.println("AVG_MS=" + String.format(Locale.ROOT, "%.3f", total / reps));
    }
}
