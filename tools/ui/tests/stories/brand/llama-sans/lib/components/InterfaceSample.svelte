<script lang="ts">
	let { glyphCount, guides }: { glyphCount: number; guides: boolean } = $props();
	const sizes = [12, 14, 16];
</script>

{#snippet plus()}
	<svg aria-hidden="true" class="ico" viewBox="0 0 10 10"><path d="M1 5h8M5 1v8" /></svg>
{/snippet}

<section aria-label="Interface sample" class="ui">
	<span class="label">Interface · buttons, chips and a card, all at the font’s own line height</span
	>

	<div class:guides class="rows">
		{#each sizes as s (s)}
			<div style:--s="{s}px" class="row">
				<span class="size">{s} px</span>

				<button class="btn primary" type="button"
					>{@render plus()}<span class="t">New project</span></button
				>

				<button class="btn" type="button"><span class="t">Cancel</span></button>

				<span class="chip"><span class="t">Beta</span></span>

				<span class="chip"><span class="t">v0.6 · {glyphCount} glyphs</span></span>
			</div>
		{/each}

		<article class="card">
			<div class="card-top">
				<h3 class="t">Llama Sans UI</h3>

				<span class="chip"><span class="t">Draft</span></span>
			</div>

			<p class="meta t">Updated 4 Nov 2026 · 3 weights · {glyphCount} glyphs</p>

			<p class="body t">
				A line box centred on the cap height, so labels, icons and badges line up in buttons without
				per-component nudging.
			</p>

			<div class="actions">
				<button class="btn" type="button"><span class="t">Details</span></button>

				<button class="btn primary" type="button"><span class="t">Install</span></button>
			</div>
		</article>
	</div>
</section>

<style>
	.ui,
	.rows {
		display: grid;
		gap: 14px;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
	}
	.size {
		width: 44px;
		font: 400 11px var(--mono);
		color: var(--muted);
	}
	.btn,
	.chip {
		font: 500 var(--s, 14px) / normal var(--display);
		color: var(--ink);
	}
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 0.45em;
		padding: 0.5em 0.95em;
		border-radius: 8px;
		border: 1px solid var(--rule);
		background: var(--sheet);
		cursor: pointer;
	}
	.btn.primary {
		background: var(--ink);
		color: var(--paper);
		border-color: var(--ink);
	}
	.btn:focus-visible {
		outline: 2px solid var(--guide);
		outline-offset: 2px;
	}
	.ico {
		width: 0.71em; /* the cap height */
		height: 0.71em;
		flex: none;
		stroke: currentColor;
		stroke-width: 1.6;
		fill: none;
	}
	.chip {
		display: inline-flex;
		align-items: center;
		padding: 0.2em 0.6em;
		border-radius: 999px;
		border: 1px solid var(--rule);
		font-size: calc(var(--s, 14px) - 2px);
		color: var(--muted);
	}
	.card {
		max-width: 420px;
		display: grid;
		gap: 6px;
		padding: 16px;
		border: 1px solid var(--rule);
		border-radius: 10px;
		background: var(--sheet);
		font-family: var(--display);
	}
	.card-top {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
	}
	h3 {
		margin: 0;
		font: 600 16px / normal var(--display);
	}
	.meta {
		margin: 0;
		font: 400 12px / normal var(--display);
		color: var(--muted);
	}
	.body {
		margin: 4px 0 8px;
		font: 400 14px / normal var(--display);
	}
	.actions {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
		--s: 13px;
	}
	.guides .t {
		outline: 1px dashed var(--guide);
	}
	.guides .ico {
		outline: 1px dashed var(--mark);
	}
</style>
