#!/usr/bin/env bash
# Speculative decoding: adaptive draft sizing on this fork against fixed n, and against pinned
# upstream running the same draft model.
#
# Why this suite exists separately: llama-bench does no speculative decoding at all, and
# llama-batched-bench is not a proxy (its batch axis is parallel sequences; speculative decoding is
# n_seqs = 1 with the draft width). So this drives llama-server and reads its timings block.
#
# Why two workloads: adaptive drafting exists because a fixed n can only be right for one kind of
# content. WORKLOG.local.md measured the same target swinging 38.3 t/s on prose against 52.8 on
# JSON. A single workload - especially a random word list, which is neither - would show the
# mechanism at its worst and prove nothing. Prose and structured output are the two ends.
#
# Which spec method each arm uses, and why it is not one method for everything:
#
#   Upstream at the pinned merge base names --spec-type draft-dflash but cannot load a DFlash2
#   sidecar: it fails with "wrong number of tensors; expected 81, got 58", which WORKLOG.local.md
#   documents as the signature of the loader being absent rather than the GGUF being bad. So the
#   cross-build comparison uses MTP, which both builds support and which needs no sidecar because
#   the target carries its own nextn layers. DFlash2 arms are fork-only.
#
#   With the same method and the same n, fork-fixed against mainline-fixed isolates the build, and
#   fork-fixed against fork-adaptive isolates the drafting policy. A no-speculation arm anchors both
#   so the speedup is against bare decode rather than against another speculative setting.
#
# NPREDICT matters: adaptive sizes drafts from an EMA of accepted tokens, so it needs enough tokens
# to converge. A 64-token probe measured adaptive *losing* on JSON (20.5 against 29.5 t/s) purely
# because the controller had not settled. 300 matches the worklog.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
OUT=${OUT:-$HERE/results-spec.jsonl}
PORT=${PORT:-8099}

FORK_BIN=${FORK_BIN:-$REPO/build/bin/llama-server}
MAIN_BIN=${MAIN_BIN:-$REPO/../llama.cpp-upstream/build/bin/llama-server}

HF=/home/laurent/.cache/huggingface/hub
TARGET=${TARGET:-$HF/models--unsloth--Qwen3.8-27B-GGUF/snapshots/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-Q4_K_XL.gguf}
DRAFT=${DRAFT:-$HF/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/57ab3265056d4024870b0621cfc2c127537020ed/Qwen3.8-27B-DFlash2-Q8_0.gguf}

NPREDICT=${NPREDICT:-300}      # matches the worklog's 300-token greedy runs
# Context depths. Speculative throughput moves with depth in both directions: draft cost grows, and
# acceptance shifts with how predictable the continuation is. The README already cites MTP at 31K as
# its own datapoint, so a depth-0 number would not describe real use. Depth is created by prefixing
# deterministic filler, identical across arms; the real depth is taken from the server's prompt_n
# rather than assumed.
DEPTHS=${DEPTHS:-0 4096 16384 32768}
CTX=${CTX:-49152}
ROUNDS=${ROUNDS:-2}
VRAM_IDLE_MB=${VRAM_IDLE_MB:-4096}
GAP_SECONDS=${GAP_SECONDS:-10}

vram_mb() { echo $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo 0) / 1048576 )); }
settle() { for _ in $(seq 1 120); do [ "$(vram_mb)" -le "$VRAM_IDLE_MB" ] && { sleep "$GAP_SECONDS"; return 0; }; sleep 2; done; sleep "$GAP_SECONDS"; }

STOPPED=""
restore() { [ -n "$STOPPED" ] && { echo "restarting: $STOPPED"; for c in $STOPPED; do docker start "$c" >/dev/null 2>&1 || true; done; STOPPED=""; }; stop_server; }
stop_server() { [ -n "${SRV:-}" ] && kill -TERM "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; SRV=""; return 0; }
trap restore EXIT INT TERM

preflight() {
  local busy; busy=$(pgrep -af "bin/llama-server|bin/llama-bench|bin/llama-cli" | grep -v "run-spec-suite" || true)
  [ -n "$busy" ] && { echo "refusing to start, GPU busy:" >&2; echo "$busy" | sed 's/^/  /' >&2; exit 1; }
  if command -v docker >/dev/null 2>&1; then
    for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
      docker inspect "$c" --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null \
        | grep -qE "/dev/dri|/dev/kfd" && { echo "stopping GPU container: $c"; docker stop "$c" >/dev/null 2>&1 && STOPPED="$STOPPED $c"; }
    done
  fi
  for f in "$TARGET" "$DRAFT"; do [ -f "$f" ] || { echo "missing model: $f" >&2; exit 1; }; done
  for b in "$FORK_BIN" "$MAIN_BIN"; do [ -x "$b" ] || { echo "missing binary: $b" >&2; exit 1; }; done
  # Describe both models from their GGUF metadata, not from filenames: an append-only log is only
  # useful if every row says which target, which draft and which quant produced it.
  MODEL_DESC=$(python3 - "$TARGET" "$DRAFT" <<'PY'
import json, os, sys
sys.path.insert(0, "gguf-py")
def describe(path):
    d = {"file": os.path.basename(path),
         "size_mb": round(os.path.getsize(os.path.realpath(path)) / 1048576)}
    try:
        from gguf import GGUFReader
        r = GGUFReader(path)
        for k in r.fields:
            if k == "general.architecture":
                d["arch"] = str(r.fields[k].contents())
            elif k == "general.file_type":
                d["file_type"] = int(r.fields[k].contents())
            elif k == "general.size_label":
                d["size_label"] = str(r.fields[k].contents())
        import collections
        d["quants"] = dict(collections.Counter(
            str(t.tensor_type).split(".")[-1] for t in r.tensors).most_common(4))
    except Exception as e:
        d["describe_error"] = str(e)
    return d
print(json.dumps({"target": describe(sys.argv[1]), "draft": describe(sys.argv[2])}))
PY
)
  PWR_LEVEL=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level 2>/dev/null || echo unknown)
  PWR_STATE=$(cat /sys/class/drm/card1/device/power_dpm_state 2>/dev/null || echo unknown)
  echo "preflight ok: vram=$(vram_mb)MB power=$PWR_LEVEL/$PWR_STATE"
  echo "  target: $(python3 -c 'import json,sys;d=json.loads(sys.argv[1])["target"];print(d["file"], d.get("arch"), d["size_mb"], "MB", d.get("quants"))' "$MODEL_DESC")"
  echo "  draft:  $(python3 -c 'import json,sys;d=json.loads(sys.argv[1])["draft"];print(d["file"], d.get("arch"), d["size_mb"], "MB", d.get("quants"))' "$MODEL_DESC")"
}

serve() {  # serve <bin> <spec-type> <extra args...>
  local bin=$1 spec=$2; shift 2
  local spec_args=()
  case "$spec" in
    none)         spec_args=() ;;                                   # bare decode baseline
    draft-mtp)    spec_args=(--spec-type draft-mtp) ;;              # target's own nextn layers
    draft-dflash) spec_args=(--spec-type draft-dflash -md "$DRAFT") ;;
  esac
  settle
  "$bin" -m "$TARGET" "${spec_args[@]}" \
         --host 127.0.0.1 --port "$PORT" -fa on -ngl 999 --spec-draft-ngl 99 -c "$CTX" "$@" >/tmp/spec-server.log 2>&1 &
  SRV=$!
  for _ in $(seq 1 240); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0; sleep 2; done
  echo "  server failed to start, see /tmp/spec-server.log" >&2; stop_server; return 1
}

probe() {  # probe <build> <policy> <workload> <round> <depth> <spec-method>
  python3 - "$1" "$2" "$3" "$4" "$PORT" "$NPREDICT" "$OUT" "$5" "$6" "$MODEL_DESC" <<'PY'
import json, sys, random, urllib.request, datetime
build, policy, workload, rnd, port, npred, out, depth, spec_method, model_desc = sys.argv[1:11]
depth = int(depth)

# Deterministic filler, identical bytes for every arm at a depth so the prompt cache can reuse the
# prefix and no arm sees an easier context.
#
# It has to be *coherent* text. A first version used random word soup and the model simply stopped:
# a JSON probe at depth 16384 returned one token and 0.00 t/s, because 16k of out-of-distribution
# noise is nothing like the long documents this is meant to represent. Real prose from the repo's
# own docs keeps the context in distribution and makes the depth axis mean something.
filler = ""
if depth:
    import pathlib
    srcs = ["README.md", "CONTRIBUTING.md", "docs/build.md", "docs/development/token_generation_performance_tips.md"]
    corpus = ""
    root = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else pathlib.Path(".")
    for rel in srcs:
        for base in (pathlib.Path("."), pathlib.Path(__file__).resolve().parents[1] if "__file__" in dir() else pathlib.Path(".")):
            f = base / rel
            if f.is_file():
                corpus += f.read_text(errors="ignore") + "\n\n"
                break
    if not corpus:
        corpus = ("Long context behaviour depends on how much of the sequence the model must attend "
                  "over, and on how predictable the continuation is. ") * 200
    need = int(depth * 4.0)                      # ~4 chars per token, deliberately generous
    while len(corpus) < need:
        corpus += corpus
    filler = corpus[:need] + "\n\n"
prompts = {
 "prose": "Write a continuous, flowing essay of several paragraphs about how tidal patterns shaped "
          "the settlement of coastal towns. Do not use lists or headings.",
 "json":  "Return ONLY a JSON array of 12 objects, no prose. Each object must have exactly the keys "
          '"id" (integer), "name" (string), "category" (string), "score" (number) and "active" '
          "(boolean). Begin your reply with [ and end it with ].",
}
req = {"prompt": filler + prompts[workload], "n_predict": int(npred), "temperature": 0,
       "cache_prompt": True}   # reuse the filler prefix across workloads at one depth
r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=1800)
d = json.load(r); t = d.get("timings", {})
row = {"suite": "spec", "build": build, "policy": policy, "workload": workload, "round": int(rnd),
       "depth_requested": depth, "prompt_n": t.get("prompt_n"),
       "spec_method": spec_method, "models": json.loads(model_desc),
       "power_dpm_level": __import__("os").environ.get("PWR_LEVEL", "unknown"),
       "power_dpm_state": __import__("os").environ.get("PWR_STATE", "unknown"),
       "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
       "prompt_per_second": t.get("prompt_per_second"),
       "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}
row["degenerate"] = (row["predicted_n"] or 0) < int(npred) * 0.9
open(out, "a").write(json.dumps(row) + "\n")
acc = ""
if row["draft_n"]:
    acc = f"  accept {100.0*(row['draft_n_accepted'] or 0)/row['draft_n']:.0f}%"
flag = "  DEGENERATE (stopped early, excluded)" if row["degenerate"] else ""
print(f"  {build:8} {policy:12} {workload:5} d={depth:<6} n={row['predicted_n'] or 0:4} "
      f"{row['predicted_per_second'] or 0:6.2f} t/s{acc}{flag}")
PY
}

preflight
export PWR_LEVEL PWR_STATE
: > "$OUT"
for round in $(seq 1 "$ROUNDS"); do
  echo "=== round $round ==="
  # arms are ordered so the two builds alternate, limiting drift bias between them
  for arm in "mainline:none:none:" \
             "fork:none:none:" \
             "mainline:mtp-n3:draft-mtp:--spec-draft-n-max 3" \
             "fork:mtp-n3:draft-mtp:--spec-draft-n-max 3" \
             "fork:mtp-adaptive:draft-mtp:--spec-draft-n-max 7 --spec-draft-adaptive --spec-draft-n-min 3" \
             "mainline:mtp-n7:draft-mtp:--spec-draft-n-max 7" \
             "fork:mtp-n7:draft-mtp:--spec-draft-n-max 7" \
             "fork:dflash-n3:draft-dflash:--spec-draft-n-max 3" \
             "fork:dflash-adaptive:draft-dflash:--spec-draft-n-max 7 --spec-draft-adaptive --spec-draft-n-min 3" \
             "fork:dflash-n7:draft-dflash:--spec-draft-n-max 7"; do
    IFS=: read -r build policy spec args <<< "$arm"
    bin=$FORK_BIN; [ "$build" = mainline ] && bin=$MAIN_BIN
    echo "-- $build $policy ($spec)"
    if serve "$bin" "$spec" $args; then
      for d in $DEPTHS; do
        for w in prose json; do probe "$build" "$policy" "$w" "$round" "$d" "$spec" || echo "  probe failed"; done
      done
      stop_server
    fi
  done
done
echo "SPEC_SUITE_DONE -> $OUT"
