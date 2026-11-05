<!--
	The proposed matte-glass finish, after the reference but quieter: every surface is a gentle gradient between near
	shades of its own colour, with a hairline edge a step lighter than the fill and glyphs pressed lightly into the
	surface. No glows; only a faint neutral shadow. The colours are the status fills. Plain markup, so the real
	components stay as they are until the finish is chosen; the buttons are the real Button, keeping its size and
	radius, with only the finish added by a class.
-->
<script lang="ts">
	import { Check, Minus, Plus, Send, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

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
		<Button class="matte-primary">Load model</Button>

		<Button class="matte-secondary" variant="secondary">Cancel</Button>

		<Button aria-label="Send" class="matte-primary rounded-full" size="icon"><Send /></Button>

		<Button aria-label="Stop" class="matte-lit danger rounded-full" size="icon"
			><span class="stop"></span></Button
		>
	</div>
{/if}

<style>
	.danger,
	.row :global(.danger) {
		--c: var(--fill-danger);
	}
	.warning,
	.row :global(.warning) {
		--c: var(--fill-warning);
	}
	.success,
	.row :global(.success) {
		--c: var(--fill-success);
	}
	.info,
	.row :global(.info) {
		--c: var(--fill-info);
	}

	/* a faint, diffuse neutral shadow, the only lift; nothing tight under the bottom edge */
	.tray,
	.row,
	.count,
	.key {
		--lift: 0 4px 12px -6px rgb(15 23 42 / 0.1);
	}
	:global(.dark) .tray,
	:global(.dark) .row,
	:global(.dark) .count,
	:global(.dark) .key {
		--lift: 0 4px 12px -6px rgb(0 0 0 / 0.4);
	}

	/* neutral: a fill barely lighter at the top, an even hairline edge */
	.tray,
	.row :global(.matte-secondary) {
		background: linear-gradient(to bottom, #f7f7f8, #f1f1f3);
		border: 1px solid #e0e2e6;
		box-shadow: var(--lift);
	}
	:global(.dark) .tray,
	:global(.dark) .row :global(.matte-secondary) {
		background: linear-gradient(to bottom, #2d2e30, #292a2c);
		border-color: rgb(255 255 255 / 0.08);
		box-shadow: var(--lift);
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

	/* lit colour: a near-flat fill, an even hairline a step lighter, the glyph a little darker and pressed in */
	.lit,
	.row :global(.matte-lit) {
		background: linear-gradient(to bottom, oklch(from var(--c) calc(l + 0.02) c h), var(--c));
		border: 1px solid oklch(from var(--c) calc(l + 0.06) calc(c * 0.9) h);
		color: oklch(from var(--c) calc(l - 0.2) calc(c * 0.85) h);
		box-shadow: var(--lift);
	}
	.lit :global(svg),
	.row :global(.matte-lit svg) {
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
			color-mix(in oklab, var(--surface), white 20%),
			var(--surface)
		);
		border: 1px solid var(--line);
	}
	:global(.dark) .pill {
		background: linear-gradient(
			to bottom,
			color-mix(in oklab, var(--surface), white 4%),
			var(--surface)
		);
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

	/* buttons: the real Button's size and radius; only the finish changes */
	.row :global(.matte-secondary) {
		color: var(--foreground);
	}
	/* the primary key: dark neutral, barely lighter at the top, an even hairline edge */
	.row :global(.matte-primary) {
		color: #fafafa;
		background: linear-gradient(to bottom, #2e2e30, #262628);
		/* an even ring rather than a border, so the width stays the real button's */
		box-shadow:
			inset 0 0 0 1px #1c1c1e,
			var(--lift);
	}
	:global(.dark) .row :global(.matte-primary) {
		color: #171717;
		background: linear-gradient(to bottom, #f3f3f4, #ebebed);
		box-shadow:
			inset 0 0 0 1px #d6d7da,
			var(--lift);
	}
	.stop {
		width: 11px;
		height: 11px;
		border-radius: 3px;
		background: currentColor;
	}
</style>
