# moe-trace

Logs which routed experts a mixture-of-experts model selects — one
`layer,expert` CSV row per routed activation — for a prompt evaluation plus a
short greedy continuation.

Use it to measure expert-routing skew before investing in hot-expert caching,
pinning, or offload placement: a near-uniform router has no hot set to exploit,
while a skewed one rewards keeping its favourite experts resident.

```sh
MOE_TRACE_OUT=trace.csv \
llama-moe-trace -m model.gguf -f prompt.txt -n 64 -ngl 99

# hottest (layer,expert) pairs
sort trace.csv | uniq -c | sort -rn | head

# fraction of all (layer,expert) slots that ever fired
awk -F, '{u[$0]=1} END {print length(u)}' trace.csv
```

Tracing hooks the eval callback and downloads only the tiny `ffn_moe_topk`
id tensors, so overhead is negligible.

Example finding that motivated the tool: Qwen3.8-Flash-Next (512 experts,
top-10) activated 98.6% of all (layer,expert) slots within ~1.3k tokens with a
hottest-slot share of only ~2.5x uniform — i.e. no exploitable hot set.
