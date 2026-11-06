<!--
	Llama Mono in logs and tables: llama-server's request log in columns (time, level, slot, message), llama-bench's
	results as the Markdown table it prints. Diffs have their own story (Diff.svelte).
-->
<script lang="ts">
	// time, level, source, message
	const log: [string, 'D' | 'E' | 'I' | 'W', string, string][] = [
		['12:04:31.207', 'I', 'srv', 'params_from_: Chat format: Content-only'],
		['12:04:31.208', 'I', 'slot 0', 'launch_slot_: id  0 | task 118 | processing task'],
		['12:04:31.208', 'D', 'slot 0', 'update_slots: n_past = 1024, cache reuse 87.5%'],
		['12:04:31.519', 'I', 'slot 0', 'update_slots: prompt done, n_tokens = 1076, 312.4 t/s'],
		['12:04:33.102', 'W', 'slot 1', 'update_slots: context 92% full, shifting 512 tokens'],
		['12:04:44.856', 'I', 'slot 0', 'print_timing: eval 512 tokens, 13.31 s, 38.47 t/s'],
		['12:04:44.857', 'I', 'srv', 'log_server_r: request: POST /v1/chat/completions 200'],
		['12:04:45.010', 'E', 'srv', 'log_server_r: request: POST /v1/embeddings 501']
	];

	const bench = `| model                          |       size |   params | backend | ngl |  test |             t/s |
| ------------------------------ | ---------: | -------: | ------- | --: | ----: | --------------: |
| gemma3 12B Q4_K - Medium       |   6.79 GiB |  11.77 B | Metal   |  99 | pp512 |  612.37 ± 2.14  |
| gemma3 12B Q4_K - Medium       |   6.79 GiB |  11.77 B | Metal   |  99 | tg128 |   38.47 ± 0.06  |
| qwen3 8B Q8_0                  |   8.11 GiB |   8.19 B | Metal   |  99 | pp512 |  845.02 ± 3.80  |
| qwen3 8B Q8_0                  |   8.11 GiB |   8.19 B | Metal   |  99 | tg128 |   41.93 ± 0.11  |`;
</script>

<figure class="panel dark">
	<figcaption>llama-server · request log</figcaption>

	<!-- prettier-ignore -->
	<pre class="screen">{#each log as [time, level, source, message], i (i)}<span class="time">{time}</span> <span class="lvl {level}">{level}</span> <span class="src">{source.padEnd(6)}</span> {message}
{/each}</pre>
</figure>

<figure class="panel light">
	<figcaption>llama-bench · Markdown output</figcaption>

	<pre class="screen">{bench}</pre>
</figure>

<div class="notes">
	<p>
		<b>Columns without a table.</b> A log has no grid, only spaces; it's the fixed cell that turns them
		into columns, so the times, the levels and the slots line up down the page.
	</p>

	<p>
		<b>Figures that compare.</b> Every digit is the same width, so 38.47 sits over 41.93 and a reader
		compares numbers by their position, without reading them.
	</p>

	<p>
		<b>Levels in their own cell.</b> I, W and E take one cell before the source, so a warning stands out
		by its colour without pushing its message out of line.
	</p>
</div>

<style>
	.panel {
		margin: 0;
		overflow: hidden;
		border-radius: 10px;
	}
	.panel.dark {
		--bg: #111111;
		--fg: #fafafa;
		--dim: #8a8a8a;
		--bar: #1c1c1c;
		--info: #93c5fd;
		--warn: #facc15;
		--err: #f87171;
	}
	.panel.light {
		--bg: #fafafa;
		--fg: #111111;
		--dim: #737373;
		--bar: #ececec;
		--info: #1d4ed8;
		--warn: #a16207;
		--err: #b91c1c;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.1);
	}
	figcaption {
		padding: 10px 14px;
		background: var(--bar);
		color: var(--dim);
		font: 500 12px var(--code);
	}
	.screen {
		margin: 0;
		padding: 16px 18px 20px;
		overflow-x: auto;
		background: var(--bg);
		color: var(--fg);
		font: 400 13px / 1.6 var(--code);
	}
	.time,
	.src {
		color: var(--dim);
	}
	.lvl {
		font-weight: 700;
	}
	.lvl.I {
		color: var(--info);
	}
	.lvl.D {
		color: var(--dim);
	}
	.lvl.W {
		color: var(--warn);
	}
	.lvl.E {
		color: var(--err);
	}
	.notes {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: 16px 40px;
	}
	.notes p {
		margin: 0;
		font: 400 15px / 1.5 var(--ui);
		color: var(--muted);
	}
	.notes b {
		font-weight: 600;
		color: var(--ink);
	}
</style>
