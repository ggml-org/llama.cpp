<!--
	The family's tester: one text set in Llama Sans and Llama Mono side by side, both editable and kept in step, with
	the size, weight, tracking and line-height shared. Llama Sans takes its optical size from the size (auto, as in
	the browser) or a pinned opsz; Llama Mono has none, and its ligatures can be turned off. It sits in OverviewFrame.
-->
<script lang="ts">
	import { specimen } from '../llama-sans/lib/specimen';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';

	const { axis } = specimen;

	const samples = [
		{ label: 'Pangram', text: 'The quick brown fox jumps over the lazy dog' },
		{ label: 'Interface', text: 'Settings · Models · New chat · Load GGUF' },
		{ label: 'Code', text: 'llama-server -m model.gguf --port 9931 -> ok != err' },
		{ label: 'Numbers', text: '0123456789 · 42.7 tok/s · 0x1F40 · 3/4' },
		{ label: 'Look-alikes', text: 'Il1| O0o rn m {[( )]} ,.;' }
	];

	let text = $state(samples[0].text);
	let size = $state(48);
	let weight = $state(axis.min);
	let tracking = $state(0); // thousandths of an em, on top of the font's own
	let leading = $state(1.25);
	let opszMode = $state<'auto' | 'pin'>('auto');
	let opsz = $state(16);
	let ligatures = $state(true);
	let fonts = $state(['sans', 'mono']);
	const show = $derived({ mono: fonts.includes('mono'), sans: fonts.includes('sans') });
	// the preset the text still matches, if any
	const preset = $derived(samples.find((s) => s.text === text)?.label ?? '');

	// what Llama Sans is set at: the browser maps opsz to the size in px, within the axis's 16–40
	const effective = $derived(opszMode === 'auto' ? Math.min(40, Math.max(16, size)) : opsz);
	const named = $derived(effective === 16 ? ' · UI' : effective === 40 ? ' · Display' : '');

	const rows = $derived(
		[
			{
				face: "'Llama Sans', var(--body)",
				id: 'sans' as const,
				meta: `opsz ${effective}${opszMode === 'auto' ? ' (auto)' : ''}${named}`,
				name: 'Llama Sans',
				pin: opszMode === 'auto' ? 'normal' : `'opsz' ${opsz}`
			},
			{
				face: 'var(--code)',
				id: 'mono' as const,
				meta: ligatures ? 'ligatures on' : 'ligatures off',
				name: 'Llama Mono',
				pin: 'normal'
			}
		].filter((r) => show[r.id])
	);
</script>

<div class="tester">
	<header>
		<h1>Tester</h1>

		<p>
			Type in either line: both follow. The size, weight, tracking and line are shared, so Llama
			Sans and Llama Mono compare like for like.
		</p>
	</header>

	<!-- the design system's own controls (in .ds, so the specimen's reset leaves them alone); there is no slider in it
	     yet, so the four ranges stay native -->
	<div class="controls ds">
		<div class="group">
			<span class="label">Fonts</span>

			<ToggleGroup.Root bind:value={() => fonts, (v) => v.length && (fonts = v)} type="multiple">
				<ToggleGroup.Item value="sans">Llama Sans</ToggleGroup.Item>

				<ToggleGroup.Item value="mono">Llama Mono</ToggleGroup.Item>
			</ToggleGroup.Root>
		</div>

		<div class="group">
			<span class="label">Text</span>

			<ToggleGroup.Root
				bind:value={() => preset, (v) => v && (text = samples.find((s) => s.label === v)!.text)}
				type="single"
			>
				{#each samples as s (s.label)}
					<ToggleGroup.Item value={s.label}>{s.label}</ToggleGroup.Item>
				{/each}
			</ToggleGroup.Root>
		</div>

		<label class="slider">
			<span class="label">Size <output>{size} px</output></span>

			<input bind:value={size} max="160" min="10" step="1" type="range" />
		</label>

		<label class="slider">
			<span class="label">Weight <output>{weight}</output></span>

			<input bind:value={weight} max={axis.max} min={axis.min} step="1" type="range" />
		</label>

		<label class="slider">
			<span class="label">Tracking <output>{tracking > 0 ? '+' : ''}{tracking}</output></span>

			<input bind:value={tracking} max="100" min="-60" step="1" type="range" />
		</label>

		<label class="slider">
			<span class="label">Line <output>{leading.toFixed(2)}</output></span>

			<input bind:value={leading} max="2" min="0.9" step="0.05" type="range" />
		</label>

		<div class="group">
			<span class="label">Llama Sans optical size</span>

			<ToggleGroup.Root
				bind:value={() => opszMode, (v) => v && (opszMode = v as 'auto' | 'pin')}
				type="single"
			>
				<ToggleGroup.Item value="auto">Auto, from size</ToggleGroup.Item>

				<ToggleGroup.Item value="pin">Pinned</ToggleGroup.Item>
			</ToggleGroup.Root>

			{#if opszMode === 'pin'}
				<label class="slider">
					<span class="label"
						>opsz <output>{opsz}{opsz === 16 ? ' · UI' : opsz === 40 ? ' · Display' : ''}</output
						></span
					>

					<input bind:value={opsz} max="40" min="16" step="1" type="range" />
				</label>
			{/if}
		</div>

		<div class="group">
			<span class="label">Llama Mono</span>

			<Label class="flex h-control-sm items-center gap-2 text-label-sm">
				<Switch bind:checked={ligatures} /> Ligatures
			</Label>
		</div>
	</div>

	<div class="rows">
		{#each rows as row (row.id)}
			<section aria-label={row.name} class="row">
				<p class="meta"><span>{row.name}</span><span>{row.meta}</span></p>

				<textarea
					bind:value={text}
					style:--face={row.face}
					style:--opsz-pin={row.pin}
					style:font-size="{size}px"
					style:font-variant-ligatures={row.id === 'mono' && !ligatures ? 'none' : 'normal'}
					style:font-weight={weight}
					style:letter-spacing="{tracking / 1000}em"
					style:line-height={leading}
					aria-label="{row.name} sample text"
					rows="1"
					spellcheck="false"
				></textarea>
			</section>
		{/each}
	</div>
</div>

<style>
	.tester {
		display: grid;
		gap: 32px;
		width: var(--column);
		margin: 0 auto;
		padding-top: clamp(32px, 5vw, 64px);
	}
	header {
		display: grid;
		gap: 8px;
	}
	header > * {
		margin: 0;
	}
	h1 {
		font: 700 clamp(36px, 5vw, 56px) / 1.1 var(--headline);
		letter-spacing: -0.01em;
	}
	header p {
		max-width: 62ch;
		font: 400 16px / 1.5 var(--ui);
		color: var(--muted);
	}

	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		gap: 16px 28px;
		padding: 16px 0;
		background: color-mix(in oklab, var(--paper) 88%, transparent);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid var(--rule);
	}
	/* on wide screens the bar stays in view while the text scrolls; on phones it's too tall for that */
	@media (min-width: 760px) {
		.controls {
			position: sticky;
			top: 0;
			z-index: 5;
		}
	}
	.group,
	.slider {
		display: grid;
		gap: 8px;
		align-content: start;
	}
	.slider {
		width: min(100%, 180px);
	}
	@media (max-width: 480px) {
		.controls > .slider {
			width: calc(50% - 14px);
		}
	}
	.label {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	output {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	input[type='range'] {
		width: 100%;
		margin: 0;
		accent-color: var(--ink);
	}
	.rows {
		display: grid;
		gap: 28px;
	}
	.row {
		display: grid;
		gap: 8px;
		padding-top: 10px;
		border-top: 1px solid var(--rule);
	}
	.meta {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin: 0;
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	textarea {
		box-sizing: border-box;
		width: 100%;
		min-height: 1.4em;
		margin: 0;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		font-family: var(--face);
		field-sizing: content;
		resize: none;
		overflow: hidden;
		overflow-wrap: anywhere;
		outline: none;
	}
	textarea:focus-visible {
		box-shadow: 0 2px 0 var(--mark);
	}
</style>
