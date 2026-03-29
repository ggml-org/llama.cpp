#!/usr/bin/env python3
"""llama.cpp cross-backend profiler analysis tool.

Usage:
    python -m tools.profiler.profiler profile.json
    python -m tools.profiler.profiler profile.json --chrome-trace trace.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


OP_EVENT = 0
COPY_EVENT = 1

TYPE_NAMES = {0: "OP", 1: "COPY"}


@dataclass
class ProfileRecord:
    type: int
    name: str
    backend_id: int
    split_id: int
    start_ns: int
    duration_ns: int
    bytes: int
    extra: Optional[str]
    ne_src0: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    ne_src1: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, f"UNKNOWN({self.type})")

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0

    @property
    def bandwidth_gbps(self) -> float:
        """Bandwidth in GB/s (only meaningful for copy events)."""
        if self.duration_ns == 0 or self.bytes == 0:
            return 0.0
        return self.bytes / self.duration_ns

    @staticmethod
    def _fmt_ne(ne: list[int]) -> str:
        dims = [n for n in ne if n > 0]
        if not dims:
            return ""
        return "[" + ", ".join(str(d) for d in dims) + "]"

    @property
    def shape_str(self) -> str:
        """Human-readable tensor shapes, e.g. '[4096, 4096] x [4096, 1]'."""
        s0 = self._fmt_ne(self.ne_src0)
        s1 = self._fmt_ne(self.ne_src1)
        if s0 and s1:
            return s0 + " x " + s1
        return s0 or s1

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "backend_id": self.backend_id,
            "split_id": self.split_id,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "bytes": self.bytes,
            "extra": self.extra,
            "ne_src0": self.ne_src0,
            "ne_src1": self.ne_src1,
        }


@dataclass
class OpStats:
    name: str
    event_type: int
    backend_id: int
    count: int = 0
    total_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    total_bytes: int = 0
    representative_ne: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def avg_ns(self) -> float:
        return self.total_ns / self.count if self.count > 0 else 0

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    @property
    def min_us(self) -> float:
        return self.min_ns / 1000.0

    @property
    def max_us(self) -> float:
        return self.max_ns / 1000.0

    @property
    def bandwidth_gbps(self) -> float:
        if self.total_ns == 0 or self.total_bytes == 0:
            return 0.0
        return self.total_bytes / self.total_ns

    @property
    def time_per_byte_ns(self) -> float:
        """Time per byte (lower = more efficient)."""
        if self.total_bytes == 0:
            return float("inf")
        return self.total_ns / self.total_bytes

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.event_type, f"UNKNOWN({self.event_type})")


class ProfileData:
    def __init__(self, records: list[ProfileRecord], metadata: dict):
        self.records = records
        self.metadata = metadata

    @classmethod
    def load(cls, filepath: str | Path) -> ProfileData:
        """Load a profiler JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if data.get("profiler") != "ggml":
            print(f"Warning: file may not be a ggml profiler output (profiler={data.get('profiler')})")

        records = []
        def _pad_ne(v):
            if isinstance(v, list) and len(v) < 4:
                return v + [0] * (4 - len(v))
            if not isinstance(v, list):
                return [0, 0, 0, 0]
            return v

        for r in data.get("records", []):
            # Support both old "ne" format and new "ne_src0"/"ne_src1" format
            ne_src0 = _pad_ne(r.get("ne_src0", r.get("ne", [0, 0, 0, 0])))
            ne_src1 = _pad_ne(r.get("ne_src1", [0, 0, 0, 0]))
            records.append(ProfileRecord(
                type=r.get("type", 0),
                name=r.get("name", "unknown"),
                backend_id=r.get("backend_id", 0),
                split_id=r.get("split_id", 0),
                start_ns=r.get("start_ns", 0),
                duration_ns=r.get("duration_ns", 0),
                bytes=r.get("bytes", 0),
                extra=r.get("extra"),
                ne_src0=ne_src0,
                ne_src1=ne_src1,
            ))

        backends_raw = data.get("backends", [])
        backends = []
        for b in backends_raw:
            backends.append({
                "id": b.get("id", 0),
                "name": b.get("name", "unknown"),
                "device": b.get("device", "unknown"),
                "device_type": b.get("device_type", 0),
            })

        metadata = {
            "version": data.get("version", 0),
            "total_records": data.get("total_records", len(records)),
            "total_ns": data.get("total_ns", sum(r.duration_ns for r in records)),
            "backends": backends,
        }

        return cls(records, metadata)

    @property
    def total_ns(self) -> int:
        return sum(r.duration_ns for r in self.records)

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    def stats(self) -> list[OpStats]:
        """Aggregate stats grouped by (name, type, backend_id)."""
        groups: dict[tuple, OpStats] = {}
        for rec in self.records:
            key = (rec.name, rec.type, rec.backend_id)
            if key not in groups:
                groups[key] = OpStats(
                    name=rec.name,
                    event_type=rec.type,
                    backend_id=rec.backend_id,
                    min_ns=rec.duration_ns,
                    max_ns=rec.duration_ns,
                    representative_ne=list(rec.ne_src0),
                )
            s = groups[key]
            s.count += 1
            s.total_ns += rec.duration_ns
            s.min_ns = min(s.min_ns, rec.duration_ns)
            s.max_ns = max(s.max_ns, rec.duration_ns)
            s.total_bytes += rec.bytes

            # Track the ne from the longest individual call
            if rec.duration_ns >= s.max_ns:
                s.representative_ne = list(rec.ne)

        return sorted(groups.values(), key=lambda s: s.total_ns, reverse=True)

    def top_operations(self, n: int = 10) -> list[OpStats]:
        """Return the N most time-consuming operations (aggregated)."""
        return self.stats()[:n]

    def top_kernels(self, n: int = 10) -> list[ProfileRecord]:
        """Return the N longest individual kernel executions."""
        return sorted(self.records, key=lambda r: r.duration_ns, reverse=True)[:n]

    def by_backend(self) -> dict[int, list[ProfileRecord]]:
        """Group records by backend ID."""
        groups: dict[int, list[ProfileRecord]] = {}
        for rec in self.records:
            groups.setdefault(rec.backend_id, []).append(rec)
        return dict(sorted(groups.items()))

    def timeline(self) -> list[ProfileRecord]:
        """Return records sorted by start_ns for timeline visualization."""
        return sorted(self.records, key=lambda r: r.start_ns)

    def inefficiency_ranking(self, n: int = 10) -> list[OpStats]:
        """Rank operations by time per byte (inefficiency). Lower is better."""
        all_stats = [s for s in self.stats() if s.total_bytes > 0 and s.event_type == OP_EVENT]
        return sorted(all_stats, key=lambda s: s.time_per_byte_ns, reverse=True)[:n]

    def summary(self) -> None:
        """Print a formatted summary table to stdout."""
        print(f"\n{'='*80}")
        print(f"  ggml Profiler Summary")
        print(f"{'='*80}")
        print(f"  Total records: {len(self.records)}")
        print(f"  Total time:    {self.total_ms:.2f} ms")
        print(f"  Unique ops:    {len(set((r.name, r.type, r.backend_id) for r in self.records))}")
        print(f"{'='*80}\n")

        stats = self.stats()
        if not stats:
            print("  No profiling data.\n")
            return

        print(f"  {'TYPE':<5} {'BKND':>4}  {'Operation':<28} {'%Time':>7}  {'Count':>6}  "
              f"{'Total':>10}  {'Avg':>10}  {'Min':>10}  {'Max':>10}  {'Bytes':>10}")
        print(f"  {'':->5} {'':->4}  {'':->28} {'':->7}  {'':->6}  "
              f"{'(ms)':>10}  {'(us)':>10}  {'(us)':>10}  {'(us)':>10}  {'':->10}")

        for s in stats:
            pct = 100.0 * s.total_ns / self.total_ns if self.total_ns > 0 else 0

            line = (f"  {s.type_name:<5} {s.backend_id:>4}  {s.name:<28} {pct:>6.1f}%  "
                    f"{s.count:>6}  {s.total_ms:>10.2f}  {s.avg_us:>10.2f}  "
                    f"{s.min_us:>10.2f}  {s.max_us:>10.2f}")

            if s.total_bytes > 0:
                bw = s.bandwidth_gbps
                bytes_str = f"{s.total_bytes / 1e6:.1f} MB"
                if s.event_type == COPY_EVENT:
                    line += f"  {bw:>8.2f} GB/s"
                else:
                    line += f"  {bytes_str:>10}"
            else:
                line += f"  {'':>10}"

            # Tensor shape from longest call
            shape_dims = [n for n in s.representative_ne if n > 0]
            if shape_dims:
                line += f"  [{', '.join(str(d) for d in shape_dims)}]"

            print(line)

        backend_groups = self.by_backend()
        if len(backend_groups) > 1:
            print(f"\n  --- By Backend ---")
            for bid, recs in sorted(backend_groups.items()):
                bk_total = sum(r.duration_ns for r in recs)
                bk_pct = 100.0 * bk_total / self.total_ns if self.total_ns > 0 else 0
                print(f"  Backend {bid}: {bk_total / 1e6:.2f} ms ({bk_pct:.1f}%) — {len(recs)} records")

        inef = self.inefficiency_ranking(5)
        if inef:
            print(f"\n  --- Top 5 Inefficient Operations (time/byte) ---")
            for s in inef:
                print(f"  {s.name:<28} {s.time_per_byte_ns / 1000:.2f} us/byte  "
                      f"({s.count} calls, {s.total_bytes / 1e6:.1f} MB)")

        top_k = self.top_kernels(5)
        print(f"\n  --- Top 5 Longest Kernels ---")
        for rec in top_k:
            shape = f" {rec.shape_str}" if rec.shape_str else ""
            print(f"  {rec.type_name:<5} {rec.name:<28} {rec.duration_us:>10.2f} us{shape}  "
                  f"(split={rec.split_id}, backend={rec.backend_id})")

        print()

    def export_chrome_trace(self, filepath: str | Path) -> None:
        """Export as Chrome Trace Event format for chrome://tracing."""
        events = []

        # Build backend name mapping and remap to non-negative PIDs
        # (Chrome cannot handle negative PIDs)
        backend_ids = sorted(set(rec.backend_id for rec in self.records))
        backend_names: dict[int, str] = {}
        pid_map: dict[int, int] = {}

        # Use metadata from JSON if available
        metadata_backends = self.metadata.get("backends", [])
        backend_by_id: dict[int, dict] = {b["id"]: b for b in metadata_backends}

        device_type_names = {0: "CPU", 1: "GPU", 2: "ACCEL"}
        for idx, bid in enumerate(backend_ids):
            pid_map[bid] = idx
            if bid in backend_by_id:
                binfo = backend_by_id[bid]
                dev_type = binfo.get("device_type", 0)
                dev_name = binfo.get("device", "")
                type_name = device_type_names.get(dev_type, "Device")
                if dev_name and dev_name != "unknown":
                    backend_names[bid] = f"{type_name}: {dev_name}"
                else:
                    backend_names[bid] = f"{type_name}: {binfo.get('name', f'Backend {bid}')}"
            else:
                backend_names[bid] = f"Backend {bid}"

        # Process metadata events
        for bid in backend_ids:
            pid = pid_map[bid]
            events.append({
                "ph": "M",  # metadata
                "pid": pid,
                "name": "process_name",
                "args": {"name": backend_names[bid]},
            })

        # Group records by (backend_id, split_id) and lay them out sequentially
        # since we don't have reliable global timestamps across backends.
        # Within each group, events are cumulative.
        from collections import OrderedDict
        groups: OrderedDict[tuple, list[ProfileRecord]] = OrderedDict()
        for rec in self.records:
            key = (rec.backend_id, rec.split_id)
            groups.setdefault(key, []).append(rec)

        # Assign timestamps: each group starts after the previous one,
        # and events within a group are sequential (cumulative duration).
        global_ts = 0.0  # microseconds
        for key, recs in groups.items():
            backend_id, split_id = key
            pid = pid_map[backend_id]
            tid = f"split_{split_id}"

            for rec in recs:
                cat = "copy" if rec.type == COPY_EVENT else "compute"
                events.append({
                    "ph": "X",  # complete event
                    "pid": pid,
                    "tid": tid,
                    "name": rec.name,
                    "ts": global_ts,
                    "dur": rec.duration_ns / 1000.0,  # us
                    "cat": cat,
                    "args": {
                        "bytes": rec.bytes,
                        "duration_us": rec.duration_ns / 1000.0,
                        "shape": rec.shape_str,
                    },
                })
                global_ts += rec.duration_ns / 1000.0

            # Add a small gap between groups for visual separation
            global_ts += 1.0

        trace = {"traceEvents": events}
        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2)

        print(f"Chrome trace exported to: {filepath}")
        print(f"Open chrome://tracing in Chrome/Edge and load this file.")

    def export_html_viewer(self, filepath: str | Path, max_records: int = 0) -> None:
        """Export a self-contained interactive HTML timeline viewer using Canvas."""
        import json as json_mod

        metadata_backends = self.metadata.get("backends", [])
        backend_by_id: dict[int, dict] = {b["id"]: b for b in metadata_backends}

        backend_names: dict[int, str] = {}
        for bid in sorted(set(rec.backend_id for rec in self.records)):
            binfo = backend_by_id.get(bid, {})
            name = binfo.get("name", f"Backend {bid}")
            device = binfo.get("device", "")
            backend_names[bid] = device if device and device != "unknown" else name

        events: list[dict] = []
        cum_us = 0.0
        for rec in self.records:
            dur_us = rec.duration_ns / 1000.0
            events.append({
                "n": rec.name,
                "d": dur_us,
                "s": rec.shape_str,
                "b": rec.bytes,
                "t": rec.type,
                "bid": rec.backend_id,
                "start": cum_us,
            })
            cum_us += dur_us
        total_us = cum_us

        if max_records > 0 and len(events) > max_records:
            stride = len(events) // max_records
            events = events[::stride][:max_records]

        if total_us == 0:
            print("No profiling data to export.")
            return

        header_stats = str(len(events)) + ' events | ' + f'{total_us/1000:.1f}' + ' ms'

        # Build backend name map with string keys for JSON
        bn_str = {str(k): v for k, v in backend_names.items()}

        # --- HTML ---
        html = (
            '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
            '<title>ggml Profiler</title>\n<style>\n'
            '*{margin:0;padding:0;box-sizing:border-box}\n'
            'body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#eee;'
            'display:flex;flex-direction:column;height:100vh;overflow:hidden}\n'
            '#hd{background:#16213e;padding:8px 16px;display:flex;align-items:center;'
            'gap:16px;border-bottom:1px solid #0f3460;flex-shrink:0}\n'
            '#hd h1{font-size:15px;color:#e94560}\n'
            '#hd .st{font-size:11px;color:#888}\n'
            '#tb{background:#16213e;padding:6px 16px;border-bottom:1px solid #0f3460;'
            'display:flex;align-items:center;gap:6px;flex-shrink:0}\n'
            '#tb button{background:#0f3460;color:#eee;border:none;padding:5px 12px;'
            'cursor:pointer;border-radius:3px;font-size:11px}\n'
            '#tb button:hover{background:#e94560}\n'
            '#vi{font-size:10px;color:#888;margin-left:auto}\n'
            '#main{flex:1;display:flex;flex-direction:column;overflow:hidden}\n'
            '#cw{flex-shrink:0;overflow:hidden;position:relative}\n'
            '#c{display:block}\n'
            '#stats{flex:1;overflow-y:auto;background:#1a1a2e;border-top:1px solid #0f3460}\n'
            '#stats table{width:100%;border-collapse:collapse;font-size:11px}\n'
            '#stats thead{position:sticky;top:0;z-index:1}\n'
            '#stats th{text-align:left;padding:6px 10px;color:#888;background:#16213e;'
            'border-bottom:1px solid #0f3460;font-weight:normal;font-size:10px;'
            'text-transform:uppercase;letter-spacing:0.5px}\n'
            '#stats th.r{text-align:right}\n'
            '#stats td{padding:4px 10px;border-bottom:1px solid rgba(15,52,96,0.4)}\n'
            '#stats td.r{text-align:right;font-variant-numeric:tabular-nums;font-family:monospace,system-ui}\n'
            '#stats .l0 td{background:rgba(30,30,54,0.6)}\n'
            '#stats .l0:hover td{background:rgba(40,40,70,0.8)}\n'
            '#stats .l1:hover td,.l2:hover td{background:rgba(35,35,60,0.5)}\n'
            '#stats .tog{cursor:pointer;user-select:none;color:#666;'
            'width:16px;display:inline-block;text-align:center;font-size:9px}\n'
            '#stats .tog:hover{color:#e94560}\n'
            '#stats .pct-cell{position:relative}\n'
            '#stats .pct-bg{position:absolute;left:0;top:1px;bottom:1px;border-radius:2px;pointer-events:none}\n'
            '#stats .pct-tx{position:relative}\n'
            '#tt{position:fixed;background:#16213e;border:1px solid #e94560;'
            'padding:10px;border-radius:5px;font-size:11px;display:none;'
            'z-index:100;pointer-events:none;max-width:280px;line-height:1.6}\n'
            '#lg{background:#16213e;padding:6px 16px;border-top:1px solid #0f3460;'
            'font-size:10px;flex-shrink:0}\n'
            '</style></head><body>\n'
            '<div id="hd"><h1>ggml Profiler Timeline</h1>'
            '<span class="st">' + header_stats + '</span></div>\n'
            '<div id="tb">'
            '<button onclick="fitAll()">Fit</button>'
            '<button onclick="zoomTo(1000000)">1s</button>'
            '<button onclick="zoomTo(100000)">100ms</button>'
            '<button onclick="zoomTo(10000)">10ms</button>'
            '<button onclick="zoomTo(1000)">1ms</button>'
            '<button onclick="zoomTo(100)">100\u03bcs</button>'
            '<span id="vi"></span></div>\n'
            '<div id="main">\n'
            '<div id="cw"><canvas id="c"></canvas></div>\n'
            '<div id="stats"></div>\n'
            '</div>\n'
            '<div id="tt"></div>\n'
            '<div id="lg"></div>\n'
            '<script>\n'
        )

        # --- Inject data ---
        html += 'var EVENTS=' + json_mod.dumps(events, separators=(',', ':')) + ';\n'
        html += 'var BACKENDS=' + json_mod.dumps(bn_str, separators=(',', ':')) + ';\n'
        html += 'var TOTAL_US=' + repr(total_us) + ';\n'

        # --- JavaScript (plain string, no f-strings) ---
        js = r"""
// Pre-process: group events by lane
var LANE_IDS=[],seen={};
for(var i=0;i<EVENTS.length;i++){var b=EVENTS[i].bid;if(!(b in seen)){LANE_IDS.push(b);seen[b]=true;}}
LANE_IDS.sort(function(a,b){return a-b;});
var LANE_EVENTS={};
for(var i=0;i<LANE_IDS.length;i++)LANE_EVENTS[LANE_IDS[i]]=[];
for(var i=0;i<EVENTS.length;i++)LANE_EVENTS[EVENTS[i].bid].push(EVENTS[i]);

// Constants
var LANE_H=32,LABEL_W=150,MINIMAP_H=28,AXIS_H=18;
var TOP_PAD=MINIMAP_H+AXIS_H;
var BAR_PAD=3,COPY_PAD=8;

// Colors
var OP_COL={'MUL_MAT':'#4285f4','FLASH_ATTN_EXT':'#e879a0','ADD':'#81c784',
'ROPE':'#ce93d8','GET_ROWS':'#ffab91','CPY':'#b0bec5','CONCAT':'#90caf9',
'SCALE':'#80deea','MUL':'#a5d6a7','SOFT_MAX':'#fff176','RMS_NORM':'#ffcc80',
'SILU':'#ef9a9a','CONT':'#80cbc4','RESHAPE':'#9fa8da','VIEW':'#a1887f',
'PERMUTE':'#90a4ae','TRANSPOSE':'#c5e1a5','UNARY':'#f48fb1'};
function hash(s){var h=0;for(var i=0;i<s.length;i++)h=((h<<5)-h)+s.charCodeAt(i);return Math.abs(h);}
function col(n){return OP_COL[n]||('hsl('+hash(n)%360+',60%,55%)');}
function fmtT(us){if(us>=1e6)return(us/1e6).toFixed(2)+'s';if(us>=1e3)return(us/1e3).toFixed(2)+'ms';return us.toFixed(1)+'\u03bcs';}
function fmtB(b){if(!b)return'';if(b>=1e9)return(b/1e9).toFixed(1)+'GB';if(b>=1e6)return(b/1e6).toFixed(1)+'MB';if(b>=1e3)return(b/1e3).toFixed(1)+'KB';return b+'B';}
function fmtSh(s){if(!s)return'';return s.replace(/[\[\],]| x /g,function(m){return'<span style="color:#e8a040">'+m+'</span>';});}

// Canvas state
var canvas=document.getElementById('c');
var ctx,canvasW,canvasH,viewW;
var scale=1,offsetUs=0;
var hoveredEv=null,isDragging=false,dragStartX,dragStartOff;

function setup(){
  var dpr=window.devicePixelRatio||1;
  canvasW=canvas.parentElement.clientWidth;
  canvasH=Math.max(200,LANE_IDS.length*LANE_H+TOP_PAD+4);
  canvas.width=Math.round(canvasW*dpr);
  canvas.height=Math.round(canvasH*dpr);
  canvas.style.width=canvasW+'px';
  canvas.style.height=canvasH+'px';
  ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  viewW=canvasW-LABEL_W;
  document.getElementById('cw').style.height=canvasH+'px';
}

// Binary search: first event where start+d >= t
function bsFirst(evts,t){
  var lo=0,hi=evts.length;
  while(lo<hi){var m=(lo+hi)>>1;if(evts[m].start+evts[m].d<t)lo=m+1;else hi=m;}
  return lo;
}
// Binary search: find event containing time t
function bsHit(evts,t){
  var lo=0,hi=evts.length-1;
  while(lo<=hi){
    var m=(lo+hi)>>1;var ev=evts[m];
    if(t<ev.start)hi=m-1;
    else if(t>ev.start+ev.d)lo=m+1;
    else return ev;
  }
  return null;
}

// Pre-render minimap to offscreen canvas
var mmCanvas;
function buildMinimap(){
  var dpr=window.devicePixelRatio||1;
  mmCanvas=document.createElement('canvas');
  mmCanvas.width=Math.round(canvasW*dpr);
  mmCanvas.height=Math.round(MINIMAP_H*dpr);
  var mc=mmCanvas.getContext('2d');
  mc.scale(dpr,dpr);
  mc.fillStyle='#0d1117';
  mc.fillRect(0,0,canvasW,MINIMAP_H);
  var mmScale=canvasW/TOTAL_US;
  for(var li=0;li<LANE_IDS.length;li++){
    var evts=LANE_EVENTS[LANE_IDS[li]];
    var step=Math.max(1,Math.floor(evts.length/(canvasW*2)));
    mc.globalAlpha=0.6;
    for(var i=0;i<evts.length;i+=step){
      var ev=evts[i];
      mc.fillStyle=col(ev.n);
      mc.fillRect(ev.start*mmScale,2,Math.max(0.5,ev.d*mmScale),MINIMAP_H-4);
    }
  }
  mc.globalAlpha=1;
}

function clampOffset(){
  var maxOff=TOTAL_US-viewW/scale;
  if(maxOff<0)maxOff=0;
  if(offsetUs<0)offsetUs=0;
  if(offsetUs>maxOff)offsetUs=maxOff;
}

function render(){
  ctx.clearRect(0,0,canvasW,canvasH);
  var visStart=offsetUs,visEnd=offsetUs+viewW/scale;

  // Minimap
  ctx.drawImage(mmCanvas,0,0,canvasW,MINIMAP_H);
  var vpX=offsetUs/TOTAL_US*canvasW,vpW=viewW/scale/TOTAL_US*canvasW;
  ctx.strokeStyle='#e94560';ctx.lineWidth=2;
  ctx.strokeRect(vpX,1,Math.max(2,vpW),MINIMAP_H-2);
  ctx.fillStyle='rgba(233,69,96,0.15)';
  ctx.fillRect(vpX,1,Math.max(2,vpW),MINIMAP_H-2);

  // Time axis background
  ctx.fillStyle='#12122a';
  ctx.fillRect(LABEL_W,MINIMAP_H,viewW,AXIS_H);

  // Time axis ticks
  var rangeUs=visEnd-visStart;
  if(rangeUs>0){
    var raw=rangeUs/8;
    var mag=Math.pow(10,Math.floor(Math.log10(raw)));
    var iv;if(raw/mag<2)iv=2*mag;else if(raw/mag<5)iv=5*mag;else iv=10*mag;
    var firstTick=Math.ceil(visStart/iv)*iv;
    ctx.fillStyle='#555';ctx.font='9px monospace';
    ctx.strokeStyle='rgba(255,255,255,0.06)';ctx.lineWidth=1;
    for(var t=firstTick;t<=visEnd;t+=iv){
      var tx=LABEL_W+(t-offsetUs)*scale;
      ctx.beginPath();ctx.moveTo(tx,TOP_PAD);ctx.lineTo(tx,canvasH);ctx.stroke();
      ctx.fillText(fmtT(t),tx+3,MINIMAP_H+AXIS_H-4);
    }
  }

  // Lanes
  for(var li=0;li<LANE_IDS.length;li++){
    var bid=LANE_IDS[li];
    var y=TOP_PAD+li*LANE_H;

    // Background
    ctx.fillStyle=li%2===0?'#1a1a2e':'#1c1c34';
    ctx.fillRect(LABEL_W,y,viewW,LANE_H);

    // Events (clipped to event area)
    ctx.save();
    ctx.beginPath();ctx.rect(LABEL_W,y,viewW,LANE_H);ctx.clip();
    var evts=LANE_EVENTS[bid];
    if(evts&&evts.length>0){
      var si=bsFirst(evts,visStart);
      for(var i=si;i<evts.length;i++){
        var ev=evts[i];
        if(ev.start>visEnd)break;
        var x=LABEL_W+(ev.start-offsetUs)*scale;
        var w=ev.d*scale;
        ctx.fillStyle=col(ev.n);
        if(ev.t===1){
          ctx.globalAlpha=0.7;
          ctx.fillRect(x,y+COPY_PAD,Math.max(0.5,w),LANE_H-2*COPY_PAD);
          ctx.globalAlpha=1;
        }else{
          ctx.fillRect(x,y+BAR_PAD,Math.max(0.5,w),LANE_H-2*BAR_PAD);
        }
        if(w>50){
          ctx.fillStyle='#fff';ctx.font='10px system-ui';
          ctx.fillText(ev.n,x+3,y+LANE_H/2+3,w-6);
        }
      }
    }
    ctx.restore();

    // Hover highlight
    if(hoveredEv&&hoveredEv.bid===bid){
      var hx=LABEL_W+(hoveredEv.start-offsetUs)*scale;
      var hw=hoveredEv.d*scale;
      ctx.save();
      ctx.beginPath();ctx.rect(LABEL_W,y,viewW,LANE_H);ctx.clip();
      ctx.strokeStyle='#fff';ctx.lineWidth=2;
      ctx.strokeRect(hx-1,y+2,Math.max(3,hw+2),LANE_H-4);
      ctx.restore();
    }

    // Lane separator
    ctx.strokeStyle='#0f3460';ctx.lineWidth=0.5;
    ctx.beginPath();ctx.moveTo(0,y+LANE_H-0.5);ctx.lineTo(canvasW,y+LANE_H-0.5);ctx.stroke();

    // Label background + text
    ctx.fillStyle='#16213e';ctx.fillRect(0,y,LABEL_W,LANE_H);
    ctx.fillStyle='#ccc';ctx.font='11px system-ui';
    ctx.fillText(BACKENDS[bid]||('B'+bid),8,y+LANE_H/2+4);
  }

  // Axis label area background (covers labels column in axis row)
  ctx.fillStyle='#16213e';ctx.fillRect(0,MINIMAP_H,LABEL_W,AXIS_H);
  ctx.fillStyle='#666';ctx.font='9px monospace';ctx.fillText('Time',8,MINIMAP_H+AXIS_H-4);

  // View info
  document.getElementById('vi').textContent=fmtT(visStart)+' \u2014 '+fmtT(visEnd)+' ('+fmtT(rangeUs)+' visible)';
}

// --- Zoom / Pan ---
function fitAll(){scale=viewW/TOTAL_US;offsetUs=0;render();}
function zoomTo(us){scale=viewW/us;render();}

canvas.addEventListener('wheel',function(e){
  e.preventDefault();
  var r=canvas.getBoundingClientRect();
  var mx=e.clientX-r.left-LABEL_W;
  if(mx<0)return;
  var mu=offsetUs+mx/scale;
  scale*=(e.deltaY>0?0.8:1.25);
  var minScale=viewW/TOTAL_US*0.5;
  if(scale<minScale)scale=minScale;
  offsetUs=mu-mx/scale;
  clampOffset();render();
},{passive:false});

canvas.addEventListener('mousedown',function(e){
  var r=canvas.getBoundingClientRect();
  var my=e.clientY-r.top;
  if(my<MINIMAP_H){
    var frac=(e.clientX-r.left)/canvasW;
    offsetUs=frac*TOTAL_US-viewW/scale/2;
    clampOffset();render();return;
  }
  isDragging=true;dragStartX=e.clientX;dragStartOff=offsetUs;
  canvas.style.cursor='grabbing';
});
document.addEventListener('mousemove',function(e){
  if(!isDragging)return;
  offsetUs=dragStartOff-(e.clientX-dragStartX)/scale;
  clampOffset();render();
});
document.addEventListener('mouseup',function(){
  if(isDragging){isDragging=false;canvas.style.cursor='default';}
});

// --- Tooltip ---
var tip=document.getElementById('tt');
canvas.addEventListener('mousemove',function(e){
  if(isDragging)return;
  var r=canvas.getBoundingClientRect();
  var mx=e.clientX-r.left,my=e.clientY-r.top;
  var li=Math.floor((my-TOP_PAD)/LANE_H);
  if(li<0||li>=LANE_IDS.length||mx<LABEL_W){
    if(hoveredEv){hoveredEv=null;render();}
    tip.style.display='none';return;
  }
  var bid=LANE_IDS[li];
  var mu=offsetUs+(mx-LABEL_W)/scale;
  var ev=bsHit(LANE_EVENTS[bid],mu);
  if(ev){
    if(hoveredEv!==ev){hoveredEv=ev;render();}
    var h='<b style="color:#e94560">'+ev.n+'</b><br>'+fmtT(ev.d)+' | '+(BACKENDS[ev.bid]||'B'+ev.bid);
    if(ev.s)h+='<br>Shape: '+fmtSh(ev.s);
    if(ev.b)h+='<br>Bytes: '+fmtB(ev.b);
    tip.innerHTML=h;tip.style.display='block';
    tip.style.left=Math.min(e.clientX+15,window.innerWidth-280)+'px';
    tip.style.top=Math.min(e.clientY+15,window.innerHeight-100)+'px';
  }else{
    if(hoveredEv){hoveredEv=null;render();}
    tip.style.display='none';
  }
});
canvas.addEventListener('mouseleave',function(){
  if(hoveredEv){hoveredEv=null;render();}
  tip.style.display='none';
});

// --- Keyboard ---
document.addEventListener('keydown',function(e){
  var step=viewW/scale*0.2;
  if(e.key==='ArrowLeft'){offsetUs-=step;clampOffset();render();}
  else if(e.key==='ArrowRight'){offsetUs+=step;clampOffset();render();}
  else if(e.key==='+'||e.key==='='){scale*=1.5;render();}
  else if(e.key==='-'){scale/=1.5;var mn=viewW/TOTAL_US*0.5;if(scale<mn)scale=mn;render();}
  else if(e.key==='Home'){fitAll();}
});

// --- Resize ---
window.addEventListener('resize',function(){setup();buildMinimap();render();});

// --- Legend ---
function buildLegend(){
  var counts={};
  for(var i=0;i<EVENTS.length;i++){var n=EVENTS[i].n;counts[n]=(counts[n]||0)+1;}
  var entries=[];for(var n in counts)entries.push([n,counts[n]]);
  entries.sort(function(a,b){return b[1]-a[1];});
  var top=entries.slice(0,12);
  var h='';
  for(var i=0;i<top.length;i++){
    h+='<span style="display:inline-block;margin:0 8px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:'+col(top[i][0])+';margin-right:4px;vertical-align:middle"></span>'+top[i][0]+'</span>';
  }
  document.getElementById('lg').innerHTML=h;
}

// --- Stats tree-table ---
function buildStats(){
  var ops={};
  for(var i=0;i<EVENTS.length;i++){
    var ev=EVENTS[i];
    if(!ops[ev.n])ops[ev.n]={name:ev.n,d:0,count:0,min:Infinity,max:0,backends:{}};
    var op=ops[ev.n];
    op.d+=ev.d;op.count++;
    if(ev.d<op.min)op.min=ev.d;if(ev.d>op.max)op.max=ev.d;
    var bk=String(ev.bid);
    if(!op.backends[bk])op.backends[bk]={bid:ev.bid,d:0,count:0,min:Infinity,max:0,shapes:{}};
    var b=op.backends[bk];
    b.d+=ev.d;b.count++;
    if(ev.d<b.min)b.min=ev.d;if(ev.d>b.max)b.max=ev.d;
    var sh=ev.s||'\u2014';
    if(!b.shapes[sh])b.shapes[sh]={d:0,count:0,min:Infinity,max:0};
    var s=b.shapes[sh];
    s.d+=ev.d;s.count++;
    if(ev.d<s.min)s.min=ev.d;if(ev.d>s.max)s.max=ev.d;
  }
  var sorted=[];for(var k in ops)sorted.push(ops[k]);
  sorted.sort(function(a,b){return b.d-a.d;});

  // Build flat row list
  var rows=[],rid=0;
  for(var oi=0;oi<sorted.length;oi++){
    var op=sorted[oi];
    var opId=rid++;
    var bkeys=[];for(var bk in op.backends)bkeys.push(bk);
    bkeys.sort(function(a,b){return op.backends[b].d-op.backends[a].d;});
    rows.push({id:opId,p:-1,lv:0,name:op.name,d:op.d,count:op.count,
      min:op.min,max:op.max,pct:op.d/TOTAL_US*100,ch:bkeys.length>0});
    for(var bi=0;bi<bkeys.length;bi++){
      var bdata=op.backends[bkeys[bi]];
      var bId=rid++;
      var bname=BACKENDS[bdata.bid]||('B'+bdata.bid);
      var skeys=[];for(var sk in bdata.shapes)skeys.push(sk);
      skeys.sort(function(a,b){return bdata.shapes[b].d-bdata.shapes[a].d;});
      rows.push({id:bId,p:opId,lv:1,name:bname,d:bdata.d,count:bdata.count,
        min:bdata.min,max:bdata.max,pct:bdata.d/TOTAL_US*100,ch:skeys.length>0});
      for(var si=0;si<skeys.length;si++){
        var sdata=bdata.shapes[skeys[si]];
        var sId=rid++;
        rows.push({id:sId,p:bId,lv:2,name:skeys[si],d:sdata.d,count:sdata.count,
          min:sdata.min,max:sdata.max,pct:sdata.d/TOTAL_US*100,ch:false});
      }
    }
  }

  // Render
  var h='<table><thead><tr><th style="width:30%">Operation</th>'
    +'<th class="r" style="width:12%">% Time</th>'
    +'<th class="r" style="width:12%">Total</th>'
    +'<th class="r" style="width:10%">Count</th>'
    +'<th class="r" style="width:12%">Avg</th>'
    +'<th class="r" style="width:12%">Min</th>'
    +'<th class="r" style="width:12%">Max</th>'
    +'</tr></thead><tbody>';

  for(var ri=0;ri<rows.length;ri++){
    var r=rows[ri];
    var indent=8+r.lv*20;
    var vis=r.lv===0?'':'display:none';
    var tog=r.ch?'<span class="tog" onclick="togRow('+r.id+',this)">\u25b6</span>'
      :'<span style="width:16px;display:inline-block"></span>';
    var nc;
    if(r.lv===0)nc='color:'+col(r.name)+';font-weight:bold';
    else if(r.lv===1)nc='color:#ccc';
    else nc='color:#888';
    var barC=r.lv===0?col(r.name):'rgba(100,140,200,0.3)';
    var barO=r.lv===0?'0.25':'0.2';

    h+='<tr class="l'+r.lv+'" data-id="'+r.id+'" data-p="'+r.p+'" style="'+vis+'">';
    var dn=r.lv===2?fmtSh(r.name):r.name;
    h+='<td style="padding-left:'+indent+'px">'+tog+'<span style="'+nc+'">'+dn+'</span></td>';
    h+='<td class="r pct-cell"><div class="pct-bg" style="width:'+Math.max(0.5,r.pct)+'%;background:'+barC+';opacity:'+barO+'"></div><span class="pct-tx">'+r.pct.toFixed(1)+'%</span></td>';
    h+='<td class="r">'+fmtT(r.d)+'</td>';
    h+='<td class="r">'+r.count.toLocaleString()+'</td>';
    h+='<td class="r">'+fmtT(r.d/r.count)+'</td>';
    h+='<td class="r">'+fmtT(r.min)+'</td>';
    h+='<td class="r">'+fmtT(r.max)+'</td>';
    h+='</tr>';
  }
  h+='</tbody></table>';
  document.getElementById('stats').innerHTML=h;
}

function togRow(pid,el){
  var exp=el.textContent==='\u25bc';
  el.textContent=exp?'\u25b6':'\u25bc';
  var children=document.querySelectorAll('#stats tr[data-p="'+pid+'"]');
  for(var i=0;i<children.length;i++){
    children[i].style.display=exp?'none':'';
    if(exp){
      // Collapse grandchildren too
      var cid=children[i].getAttribute('data-id');
      var ctog=children[i].querySelector('.tog');
      if(ctog)ctog.textContent='\u25b6';
      var gc=document.querySelectorAll('#stats tr[data-p="'+cid+'"]');
      for(var j=0;j<gc.length;j++)gc[j].style.display='none';
    }
  }
}

// --- Init ---
setup();buildMinimap();buildLegend();buildStats();fitAll();
"""

        html += js + '\n</script></body></html>'

        with open(filepath, "w") as f:
            f.write(html)

        print(f"HTML viewer exported to: {filepath}")
        print(f"Open in browser: file://{Path(filepath).resolve()}")


def load(filepath: str | Path) -> ProfileData:
    """Load a profiler JSON file."""
    return ProfileData.load(filepath)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="llama.cpp profiler analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.profiler.profiler profile.json
  python -m tools.profiler.profiler profile.json --chrome-trace trace.json
  python -m tools.profiler.profiler profile.json --top-ops 20
        """,
    )
    parser.add_argument("profile", help="Path to profiler JSON file")
    parser.add_argument("--chrome-trace", metavar="FILE",
                        help="Export as Chrome Trace Event format")
    parser.add_argument("--html-viewer", metavar="FILE",
                        help="Export as interactive HTML timeline viewer")
    parser.add_argument("--html-max-records", type=int, default=5000,
                        help="Max records per backend in HTML viewer (0=unlimited, downsample to reduce file size)")
    parser.add_argument("--top-ops", type=int, default=0,
                        help="Show top N operations (0 = show summary)")
    parser.add_argument("--top-kernels", type=int, default=0,
                        help="Show top N longest kernels")
    parser.add_argument("--inefficiency", action="store_true",
                        help="Show inefficiency ranking")

    args = parser.parse_args()

    data = load(args.profile)

    if args.chrome_trace:
        data.export_chrome_trace(args.chrome_trace)

    if args.html_viewer:
        data.export_html_viewer(args.html_viewer, max_records=args.html_max_records)

    if args.top_ops > 0:
        print(f"\nTop {args.top_ops} operations by total time:\n")
        for s in data.top_operations(args.top_ops):
            pct = 100.0 * s.total_ns / data.total_ns if data.total_ns > 0 else 0
            print(f"  {s.type_name:<5} {s.backend_id:>4}  {s.name:<28} {pct:>6.1f}%  "
                  f"{s.count:>6}x  {s.total_ms:>10.2f} ms  avg={s.avg_us:.2f} us")
        print()

    if args.top_kernels > 0:
        print(f"\nTop {args.top_kernels} longest kernels:\n")
        for rec in data.top_kernels(args.top_kernels):
            print(f"  {rec.type_name:<5} {rec.backend_id:>4}  {rec.name:<28} "
                  f"{rec.duration_us:>10.2f} us  split={rec.split_id}")
        print()

    if args.inefficiency:
        print("\nInefficiency ranking (time/byte for operations with data):\n")
        for s in data.inefficiency_ranking(10):
            print(f"  {s.name:<28} {s.time_per_byte_ns / 1000:>10.2f} us/byte  "
                  f"{s.count:>6} calls  {s.total_bytes / 1e6:.1f} MB")
        print()

    if args.top_ops == 0 and args.top_kernels == 0 and not args.inefficiency and not args.chrome_trace and not args.html_viewer:
        data.summary()


if __name__ == "__main__":
    main()
