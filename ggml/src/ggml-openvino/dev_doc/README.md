# OpenVINO Qwen development documentation

The default supported Qwen3.5/Qwen3.6 GPU path is **dynamic stateless execution with in-place fused recurrent ops and stable MTP sampling**. Unified KV is the recommended multi-slot layout because it enables native in-place attention KV updates. No new inference switch is needed for the recurrent/sampling default.

- [Backend setup and cache usage](../README.md): runtime requirements, environment settings and restrictions.
- [Implementation](OPENVINO-RECURRENT-CACHE.md): buffer ownership, fusion, rollback, fallback and source map.
- [Multislot MTP graph](MTP-SAMPLING-GRAPH.md): why active-slot variants occurred and how stable translation works.
- [Environment variables for development](OpenVINO-ENV-VARS.md): debug output, switches for default optimizations, experimental options, plugin properties and older variable names.
