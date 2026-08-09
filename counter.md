# counter

Times the user's input produced a materially better decision than my default.
Update only when the user asks.

## Count: 17

## Examples (2026-08-06 session)

1. **Rotation + cooldown=0 test** - I concluded the corruption was AMD-specific; the user's test ("removing the cooldown should instantly corrupt the rtx?") proved it is Vulkan-wide and duplicate-id driven.

2. **"It's not the model"** - I attributed a failure to model randomness; the user's 200-run knowledge + the CUDA IQ2 test proved it is the tier/Vulkan.

3. **Sentinel + mask must stay** - I claimed copy-on-read eliminates them; the user asked "is the sentinel not still necessary?" and was right (alignment + Vulkan safety).

4. **-no-cnv invalidates corruption tests** - EOS "failures" were ambiguous without conversation mode; I had judged corruption from run counts.

5. **--fit-target 64 was missing** - the correct fit flag changed the measured config.

6. **-ehs -1 autofit** - the valid config revealed the tier is ~61 tok/s (faster than my invalid S=96 numbers).

7. **Native+lazy+madvise instead of the custom pool** - "use llama's native rampool and send a release... load them in vram from disk" replaced two committed pool phases with a simpler, better design.

8. **"RAM allocation is not actually instant"** - caught that the pool's thousands of per-slice mallocs are slow vs one native allocation.

9. **Hash is of the memory bytes, not the output** - "I said generate a hash that can only be generated from the memory" - avoided a float-tolerance mess.

10. **"Try 1024"** - shrinking the hash sample from 16KB to 1KB recovered ~4 tok/s.

11. **Copy-at-init doubles VRAM** - "will we lose the ability to use the 3gb for actual slots?" - caught the transient double-buffer that would OOM an 8GB card.

12. **Uniform first-S startup** - the user chose it over my heatmap-seeded idea.

13. **32+8 memory-fits constraint** - "we crash and oom if the model cant fit in the ram + gpu" - shaped the startup as memory distribution, not just warming.

14. **"Are you sure there is no other way?"** - led to lifting the gate and discovering the real n_tokens>1 blocker was a mask shape assert, not the predicted kernel crash.

15. **Kernel speed loss unacceptable** - pushed to the count+rank kernel fix (v3 reference) over batch-split.

16. **Deferred release** - madvise after verification, less CPU overhead.

17. **Streaming to GPU instead of second disk read** - "move the layers into the gpu in 128mb chunks" - better than my re-read fallback.
