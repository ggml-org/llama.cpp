<!--
	The proposed matte-glass finish, after the reference but quieter: every surface is a gentle gradient between near
	shades of its own colour, with a hairline edge a step lighter than the fill and glyphs pressed lightly into the
	surface. No glows; only a faint neutral shadow. The colours are the status fills. Plain markup, so the real
	components stay as they are until the finish is chosen.
-->
<script lang="ts">
	import { Check, Minus, Plus, Send, X } from '@lucide/svelte';

	let { part }: { part: 'badges' | 'buttons' | 'count' | 'tray' } = $props();

	const statuses = [
		{ label: 'Ready', tone: 'success' },
		{ label: 'Loading', tone: 'warning' },
		{ label: 'Failed', tone: 'danger' },
		{ label: 'Info', tone: 'info' }
	] as const;
</script>

{#if part === 'tray'}
	<div class="tray">
		<button aria-label="Close" class="key lit danger"><X strokeWidth={3} /></button>

		<button aria-label="Minimise" class="key lit warning"><Minus strokeWidth={3} /></button>

		<button aria-label="Expand" class="key lit success"><Plus strokeWidth={3} /></button>
	</div>
{:else if part === 'count'}
	<div class="row">
		<span class="count danger">6</span>

		<span class="count danger small">12</span>

		<span class="count info small">3</span>

		<span class="count success small"><Check strokeWidth={3} /></span>
	</div>
{:else if part === 'badges'}
	<div class="row">
		{#each statuses as s (s.tone)}
			<span class="pill {s.tone}"><span class="dot"></span>{s.label}</span>
		{/each}

		<span class="pill neutral">Q4_K_M</span>
	</div>
{:else}
	<div class="row">
		<button class="btn primary">Load model</button>

		<button class="btn secondary">Cancel</button>

		<button aria-label="Send" class="btn round primary"><Send /></button>

		<button aria-label="Stop" class="btn round lit danger"><span class="stop"></span></button>
	</div>
{/if}

<style>
	.danger {
		--c: var(--fill-danger);
	}
	.warning {
		--c: var(--fill-warning);
	}
	.success {
		--c: var(--fill-success);
	}
	.info {
		--c: var(--fill-info);
	}

	/* a faint neutral shadow, the only lift */
	.tray,
	.btn,
	.count,
	.key {
		--lift: 0 1px 2px rgb(15 23 42 / 0.06), 0 4px 12px -6px rgb(15 23 42 / 0.1);
	}
	:global(.dark) .tray,
	:global(.dark) .btn,
	:global(.dark) .count,
	:global(.dark) .key {
		--lift: 0 1px 2px rgb(0 0 0 / 0.3), 0 4px 12px -6px rgb(0 0 0 / 0.4);
	}

	/* neutral: a shade lighter at the top than at the bottom, a hairline edge */
	.tray,
	.btn.secondary {
		background: linear-gradient(to bottom, #f8f8f9, #efeff1);
		border: 1px solid #e0e2e6;
		box-shadow:
			inset 0 1px 0 rgb(255 255 255 / 0.8),
			var(--lift);
	}
	:global(.dark) .tray,
	:global(.dark) .btn.secondary {
		background: linear-gradient(to bottom, #2f3032, #28292b);
		border-color: rgb(255 255 255 / 0.08);
		box-shadow:
			inset 0 1px 0 rgb(255 255 255 / 0.05),
			var(--lift);
	}

	.tray {
		display: inline-flex;
		gap: 14px;
		padding: 12px 16px;
		border-radius: 999px;
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 14px;
	}

	/* lit colour: a near-flat fill, a hairline a step lighter, the glyph a little darker and pressed in */
	.lit {
		background: linear-gradient(to bottom, oklch(from var(--c) calc(l + 0.02) c h), var(--c));
		border: 1px solid oklch(from var(--c) calc(l + 0.06) calc(c * 0.9) h);
		color: oklch(from var(--c) calc(l - 0.2) calc(c * 0.85) h);
		box-shadow:
			inset 0 1px 0 oklch(from var(--c) calc(l + 0.1) c h / 0.35),
			var(--lift);
	}
	.lit :global(svg) {
		filter: drop-shadow(0 0.5px 0 oklch(from var(--c) calc(l + 0.1) c h / 0.7));
	}

	.key {
		display: grid;
		place-items: center;
		width: 40px;
		height: 40px;
		padding: 0;
		border-radius: 50%;
		cursor: pointer;
	}
	.key :global(svg) {
		width: 18px;
		height: 18px;
	}

	/* the count badge: a thin page-coloured collar, a near-flat fill, a lightly raised count */
	.count {
		display: inline-grid;
		place-items: center;
		min-width: 36px;
		height: 36px;
		padding: 0 9px;
		border-radius: 999px;
		font: 600 18px/1 var(--font-sans, system-ui);
		color: white;
		text-shadow: 0 0.5px 1px oklch(from var(--c) calc(l - 0.25) c h / 0.5);
		background: linear-gradient(
			to bottom,
			oklch(from var(--c) calc(l - 0.02) c h),
			oklch(from var(--c) calc(l + 0.02) c h)
		);
		border: 1px solid oklch(from var(--c) calc(l + 0.05) calc(c * 0.9) h);
		box-shadow:
			0 0 0 2px var(--background),
			0 0 0 2.5px var(--border),
			var(--lift);
	}
	.count.small {
		min-width: 22px;
		height: 22px;
		padding: 0 6px;
		font-size: 12px;
	}
	.count :global(svg) {
		width: 12px;
		height: 12px;
	}

	/* soft status pill: a tint a shade lighter at the top, a hairline edge, a plain dot */
	.pill {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		height: 24px;
		padding: 0 10px;
		border-radius: 999px;
		font: 500 12px/1 var(--font-sans, system-ui);
		color: var(--fg);
		background: linear-gradient(
			to bottom,
			color-mix(in oklab, var(--surface), white 35%),
			var(--surface)
		);
		border: 1px solid var(--line);
		box-shadow: inset 0 1px 0 rgb(255 255 255 / 0.5);
	}
	:global(.dark) .pill {
		background: linear-gradient(
			to bottom,
			color-mix(in oklab, var(--surface), white 4%),
			var(--surface)
		);
		box-shadow: inset 0 1px 0 rgb(255 255 255 / 0.04);
	}
	.pill.success {
		--fg: var(--fg-success);
		--surface: var(--surface-success);
		--line: var(--line-success-soft);
	}
	.pill.warning {
		--fg: var(--fg-warning);
		--surface: var(--surface-warning);
		--line: var(--line-warning-soft);
	}
	.pill.danger {
		--fg: var(--fg-danger);
		--surface: var(--surface-danger);
		--line: var(--line-danger-soft);
	}
	.pill.info {
		--fg: var(--fg-info);
		--surface: var(--surface-info);
		--line: var(--line-info-soft);
	}
	.pill.neutral {
		--fg: var(--foreground);
		--surface: var(--surface-muted);
		--line: var(--line-default);
	}
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--c);
	}

	/* buttons */
	.btn {
		display: inline-grid;
		place-items: center;
		height: 36px;
		padding: 0 16px;
		border-radius: 999px;
		font: 500 14px/1 var(--font-sans, system-ui);
		cursor: pointer;
	}
	.btn.secondary {
		color: var(--foreground);
	}
	/* the primary key: dark neutral, a shade lighter at the top, a faint lit top edge */
	.btn.primary {
		color: #fafafa;
		background: linear-gradient(to bottom, #333335, #232325);
		border: 1px solid #1c1c1e;
		box-shadow:
			inset 0 1px 0 rgb(255 255 255 / 0.1),
			var(--lift);
	}
	:global(.dark) .btn.primary {
		color: #171717;
		background: linear-gradient(to bottom, #f6f6f7, #e6e6e8);
		border-color: #d6d7da;
		box-shadow:
			inset 0 1px 0 white,
			var(--lift);
	}
	.btn.round {
		width: 36px;
		height: 36px;
		padding: 0;
	}
	.btn.round :global(svg) {
		width: 16px;
		height: 16px;
	}
	.stop {
		width: 11px;
		height: 11px;
		border-radius: 3px;
		background: currentColor;
	}
</style>
