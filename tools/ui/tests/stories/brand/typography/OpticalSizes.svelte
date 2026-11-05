<!--
	Llama Sans's optical size axis: one phrase at one size, set at four points of opsz from UI (16) to Display (40), so
	the spacing and kerning change is all that differs. Browsers pick the point from font-size on their own.
-->
<script lang="ts">
	const stops = [
		{ label: 'UI', opsz: 16 },
		{ opsz: 24 },
		{ opsz: 32 },
		{ label: 'Display', opsz: 40 }
	];
</script>

<div class="sizes">
	{#each stops as s (s.opsz)}
		<div class="row">
			<p class="meta">
				<span>opsz {s.opsz}</span>{#if s.label}<span>{s.label}</span>{/if}
			</p>

			<p style:--opsz-pin={`'opsz' ${s.opsz}`} class="line">Tokens per second</p>
		</div>
	{/each}
</div>

<p class="note">
	The same letters at the same size: only the spacing and kerning move. In CSS there's nothing to
	set, because <code>font-optical-sizing: auto</code> maps opsz to the font-size in px, so 16px text gets
	UI, 40px and up gets Display, and the sizes between blend.
</p>

<style>
	.sizes {
		display: grid;
		gap: 12px;
	}
	.row {
		display: grid;
		gap: 2px;
		padding-top: 10px;
		border-top: 1px solid var(--rule);
	}
	.row > * {
		margin: 0;
	}
	.meta {
		display: flex;
		justify-content: space-between;
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.line {
		overflow: hidden;
		font: 600 clamp(36px, 6vw, 72px) / 1.1 var(--ui);
		white-space: nowrap;
	}
	.note {
		margin: 0;
		max-width: 68ch;
		font: 400 14px / 1.5 var(--ui);
		color: var(--muted);
	}
	code {
		font: 400 13px var(--code);
	}
</style>
