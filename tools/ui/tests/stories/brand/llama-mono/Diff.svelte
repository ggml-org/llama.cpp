<!--
	Llama Mono in a diff: one change to llama.cpp's defaults as a unified diff, with both line numbers, the + and − in
	their own cell and the changed words marked, on a light and a dark ground; then the same change split, the old
	file beside the new. The fixed cell is what keeps a changed line over the one it replaces.
-->
<script lang="ts">
	type Kind = ' ' | '+' | '-';
	// a line: its kind, its numbers in the old and the new file, and its text as [changed, text] segments
	type Line = { kind: Kind; new?: number; old?: number; segs: [boolean, string][] };

	const hunk = '@@ -214,8 +214,9 @@ struct common_params {';

	const lines: Line[] = [
		{ kind: ' ', new: 214, old: 214, segs: [[false, '    int32_t n_ctx        = 4096;']] },
		{
			kind: '-',
			old: 215,
			segs: [
				[false, '    int32_t n_batch      = '],
				[true, '512'],
				[false, ';    // logical batch']
			]
		},
		{
			kind: '+',
			new: 215,
			segs: [
				[false, '    int32_t n_batch      = '],
				[true, '2048'],
				[false, ';   // logical batch']
			]
		},
		{ kind: '+', new: 216, segs: [[true, '    int32_t n_ubatch     = 512;    // physical batch']] },
		{
			kind: ' ',
			new: 217,
			old: 216,
			segs: [[false, '    int32_t n_gpu_layers = -1;     // -1: all that fit']]
		},
		{
			kind: '-',
			old: 217,
			segs: [
				[false, '    bool    flash_attn   = '],
				[true, 'false'],
				[false, ';']
			]
		},
		{
			kind: '+',
			new: 218,
			segs: [
				[false, '    enum llama_flash_attn_type flash_attn = '],
				[true, 'LLAMA_FLASH_ATTN_TYPE_AUTO'],
				[false, ';']
			]
		},
		{ kind: ' ', new: 219, old: 218, segs: [[false, '    float   rope_freq_base = 0.0f;']] }
	];

	// the split view's rows: context on both sides, each removal beside the addition that replaces it
	const rows = (() => {
		const out: [Line | null, Line | null][] = [];

		let dels: Line[] = [];
		let adds: Line[] = [];

		const flush = () => {
			for (let i = 0; i < Math.max(dels.length, adds.length); i++)
				out.push([dels[i] ?? null, adds[i] ?? null]);
			dels = [];
			adds = [];
		};

		for (const l of lines) {
			if (l.kind === '-') dels.push(l);
			else if (l.kind === '+') adds.push(l);
			else {
				flush();
				out.push([l, l]);
			}
		}
		flush();

		return out;
	})();

	const sign = (k: Kind) => (k === '-' ? '−' : k);
	const cls = (k: Kind) => (k === '+' ? 'add' : k === '-' ? 'del' : '');
</script>

{#snippet code(l: Line)}
	<!-- prettier-ignore -->
	<span class="code">{#each l.segs as [hi, t], i (i)}<span class:hi>{t}</span>{/each}</span>
{/snippet}

{#each ['light', 'dark'] as const as tone (tone)}
	<figure class="panel {tone}">
		<figcaption>
			<span>common/common.h</span><span class="stat"><b class="a">+3</b> <b class="d">−2</b></span>
		</figcaption>

		<div class="screen">
			<div class="row hunk"><span class="code">{hunk}</span></div>

			{#each lines as l, i (i)}
				<div class="row unified {cls(l.kind)}">
					<span class="num">{l.old ?? ''}</span>

					<span class="num">{l.new ?? ''}</span>

					<span class="sign">{sign(l.kind)}</span>

					{@render code(l)}
				</div>
			{/each}
		</div>
	</figure>
{/each}

<figure class="panel light">
	<figcaption><span>common/common.h · split</span><span class="stat">old · new</span></figcaption>

	<div class="screen split">
		{#each rows as [a, b], i (i)}
			<div class="row half {a ? cls(a.kind) : 'empty'}">
				<span class="num">{a?.old ?? ''}</span>

				{#if a}{@render code(a)}{/if}
			</div>

			<div class="row half {b ? cls(b.kind) : 'empty'}">
				<span class="num">{b?.new ?? ''}</span>

				{#if b}{@render code(b)}{/if}
			</div>
		{/each}
	</div>
</figure>

<div class="notes">
	<p>
		<b>Over the line it replaces.</b> With every character in the same cell, 2048 sits right over the
		512 it changes, and the eye finds the edit by its place in the line.
	</p>

	<p>
		<b>Marks in their own cell.</b> The + and −, and both line numbers, take cells of their own before
		the code, so its indentation stays exactly as written.
	</p>

	<p>
		<b>Words, not only lines.</b> The changed words get a stronger tint than their line, in the same greens
		and reds on both grounds; colour never carries the change alone, the sign does.
	</p>
</div>

<style>
	.panel {
		margin: 0;
		overflow: hidden;
		border-radius: 10px;
	}
	.panel.light {
		--bg: #fafafa;
		--fg: #111111;
		--dim: #8a8a8a;
		--bar: #ececec;
		--a: #15803d;
		--d: #b91c1c;
		--add: rgb(22 163 74 / 0.1);
		--add-hi: rgb(22 163 74 / 0.28);
		--del: rgb(220 38 38 / 0.08);
		--del-hi: rgb(220 38 38 / 0.24);
		--hunk: rgb(29 78 216 / 0.07);
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.1);
	}
	.panel.dark {
		--bg: #111111;
		--fg: #fafafa;
		--dim: #737373;
		--bar: #1c1c1c;
		--a: #4ade80;
		--d: #f87171;
		--add: rgb(74 222 128 / 0.1);
		--add-hi: rgb(74 222 128 / 0.3);
		--del: rgb(248 113 113 / 0.1);
		--del-hi: rgb(248 113 113 / 0.3);
		--hunk: rgb(147 197 253 / 0.08);
	}
	figcaption {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		padding: 10px 14px;
		background: var(--bar);
		color: var(--dim);
		font: 500 12px var(--code);
	}
	.stat b {
		font-weight: 700;
	}
	.stat .a {
		color: var(--a);
	}
	.stat .d {
		color: var(--d);
	}
	.screen {
		display: grid;
		padding: 8px 0 12px;
		overflow-x: auto;
		background: var(--bg);
		color: var(--fg);
		font: 400 13px / 1.7 var(--code);
	}
	.row {
		display: grid;
		min-width: max-content;
		white-space: pre;
	}
	.unified {
		grid-template-columns: 5ch 5ch 2ch 1fr;
	}
	.hunk {
		padding: 2px 14px;
		background: var(--hunk);
		color: var(--dim);
	}
	.num {
		padding-right: 1ch;
		color: var(--dim);
		text-align: right;
		user-select: none;
	}
	.sign {
		padding-left: 0.5ch;
		font-weight: 700;
	}
	.code {
		padding-right: 2ch;
	}
	.add {
		background: var(--add);
	}
	.add .sign {
		color: var(--a);
	}
	.del {
		background: var(--del);
	}
	.del .sign {
		color: var(--d);
	}
	.add .hi {
		border-radius: 2px;
		background: var(--add-hi);
	}
	.del .hi {
		border-radius: 2px;
		background: var(--del-hi);
	}

	.split {
		grid-template-columns: 1fr 1fr;
		column-gap: 1px;
	}
	.half {
		grid-template-columns: 5ch 1fr;
		min-width: 0;
		overflow: hidden;
	}
	.half.empty {
		background: repeating-linear-gradient(-45deg, transparent 0 6px, rgb(0 0 0 / 0.04) 6px 7px);
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
