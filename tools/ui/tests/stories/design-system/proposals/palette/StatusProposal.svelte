<!--
	The proposed status colours for the UI, which otherwise stays neutral: success, warning, error and info, each as
	text, a soft surface, a line and a dot, in both themes. Prototyped here with classes; the tokens and components
	change only once it is chosen. `part` picks which pieces to show, so each sits next to its current counterpart;
	`saturated` shows the same hues with more colour.
-->
<script lang="ts">
	import { CircleAlert, CircleCheck, Info, TriangleAlert } from '@lucide/svelte';

	let {
		part,
		saturated = false
	}: { part: 'alerts' | 'badges' | 'dots' | 'text'; saturated?: boolean } = $props();

	const tones = [
		{
			icon: CircleCheck,
			key: 'success',
			label: 'Model loaded',
			note: 'Qwen 2.5 7B is ready on port 9931.'
		},
		{
			icon: TriangleAlert,
			key: 'warning',
			label: 'Low memory',
			note: 'Gemma 2 27B needs 16.6 GB; 12 GB is free.'
		},
		{
			icon: CircleAlert,
			key: 'danger',
			label: 'Download failed',
			note: 'The connection dropped at 64%. Try again.'
		},
		{
			icon: Info,
			key: 'info',
			label: 'Update available',
			note: 'llama.cpp b9726 adds faster prompt processing.'
		}
	];
	const badges = [
		['success', 'Loaded'],
		['warning', 'Too large'],
		['danger', 'Failed'],
		['info', 'Update']
	];
</script>

<div class:saturated class="status">
	{#if part === 'alerts'}
		<div class="alerts">
			{#each tones as t (t.key)}
				<div class="alert {t.key}" role="status">
					<t.icon size={16} strokeWidth={2} />

					<p><strong>{t.label}</strong>{t.note}</p>
				</div>
			{/each}
		</div>
	{:else if part === 'badges'}
		<div class="row">
			{#each badges as [key, label] (key)}<span class="badge {key}"><i></i>{label}</span>{/each}
		</div>
	{:else if part === 'dots'}
		<div class="row">
			{#each badges as [key, label] (key)}<span class="dot-label"
					><i class="dot {key}"></i>{label}</span
				>{/each}
		</div>
	{:else}
		<div class="fields">
			<p class="hint success">Port 9931 is free.</p>

			<p class="hint warning">Context above 32K uses more memory.</p>

			<p class="hint danger">Enter a number between 1 and 128.</p>

			<p class="hint info">Changes apply after a restart.</p>
		</div>
	{/if}
</div>

<style>
	/* the proposed set: hue for meaning, low chroma so it sits quietly on the greys */
	.status {
		--success: oklch(0.52 0.11 150);
		--warning: oklch(0.56 0.12 70);
		--danger: oklch(0.55 0.18 27);
		--info: oklch(0.52 0.08 250);
		--soft: 9%;
		--edge: 28%;
		font-family: var(--font-sans);
	}
	:global(.dark) .status {
		--success: oklch(0.78 0.11 150);
		--warning: oklch(0.82 0.11 80);
		--danger: oklch(0.72 0.15 27);
		--info: oklch(0.77 0.07 250);
		--soft: 13%;
		--edge: 32%;
	}
	/* the saturated variant: the same hues with more colour, still at text contrast in both themes */
	.status.saturated {
		--success: oklch(0.52 0.15 150);
		--warning: oklch(0.56 0.15 65);
		--danger: oklch(0.54 0.21 27);
		--info: oklch(0.52 0.13 255);
		--soft: 11%;
		--edge: 34%;
	}
	:global(.dark) .status.saturated {
		--success: oklch(0.79 0.16 150);
		--warning: oklch(0.83 0.15 78);
		--danger: oklch(0.73 0.19 27);
		--info: oklch(0.77 0.12 255);
		--soft: 16%;
		--edge: 38%;
	}
	.success {
		--tone: var(--success);
	}
	.warning {
		--tone: var(--warning);
	}
	.danger {
		--tone: var(--danger);
	}
	.info {
		--tone: var(--info);
	}

	.alerts {
		display: grid;
		gap: 8px;
	}
	.alert {
		display: grid;
		grid-template-columns: 16px 1fr;
		gap: 12px;
		padding: 12px 16px;
		border: 1px solid color-mix(in oklab, var(--tone) var(--edge), var(--border));
		border-radius: 8px;
		background: color-mix(in oklab, var(--tone) var(--soft), var(--card));
		color: var(--foreground);
		font-size: 14px;
	}
	.alert :global(svg) {
		margin-top: 2px;
		color: var(--tone);
	}
	.alert p {
		display: grid;
		gap: 2px;
		margin: 0;
		color: var(--muted-foreground);
	}
	.alert strong {
		font-weight: 500;
		color: var(--foreground);
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
	.badge {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		padding: 2px 8px;
		border-radius: 999px;
		background: color-mix(in oklab, var(--tone) var(--soft), var(--card));
		box-shadow: inset 0 0 0 1px color-mix(in oklab, var(--tone) var(--edge), transparent);
		color: var(--tone);
		font: 500 12px / 1.5 var(--font-sans);
	}
	.badge i,
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--tone);
	}
	.dot-label {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		margin-right: 8px;
		font-size: 13px;
		color: var(--foreground);
	}
	.dot {
		width: 8px;
		height: 8px;
		box-shadow: 0 0 0 3px color-mix(in oklab, var(--tone) 20%, transparent);
	}

	.fields {
		display: grid;
		gap: 6px;
	}
	.hint {
		margin: 0;
		color: var(--tone);
		font-size: 13px;
	}
</style>
