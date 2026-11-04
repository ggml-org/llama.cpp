<script lang="ts">
	import { specimen } from '../specimen';

	let {
		guides = $bindable(),
		size = $bindable(),
		tracking = $bindable(),
		variant = $bindable(),
		weight = $bindable()
	}: {
		weight: number;
		variant: string;
		size: number;
		tracking: number;
		guides: boolean;
	} = $props();

	const { axis, styles, variants } = specimen;
	// the named weight closest to the slider; the glyph grid is drawn at named weights only
	const named = $derived(
		styles.reduce((a, s) => (Math.abs(s.weight - weight) < Math.abs(a.weight - weight) ? s : a))
			.weight
	);
	// on phones the bar floats at the bottom with weight and version only; the rest opens on request
	let more = $state(false);
</script>

<div class:more class="bar">
	<span class="label">Weight</span>

	<div aria-label="Weight" class="segmented weights" role="radiogroup">
		{#each styles as s (s.weight)}
			<label>
				<input
					checked={named === s.weight}
					name="weight"
					onchange={() => (weight = s.weight)}
					type="radio"
				/>

				<span>{s.name} <em>{s.weight}</em></span>
			</label>
		{/each}
	</div>

	<div class="group versions">
		<span class="label">Version</span>

		<div aria-label="Version" class="segmented" role="radiogroup">
			{#each variants as v (v)}
				<label>
					<input bind:group={variant} name="variant" type="radio" value={v} />

					<span>{v}</span>
				</label>
			{/each}
		</div>

		<span class="kbd">press V</span>

		<button
			aria-controls="more-controls"
			aria-expanded={more}
			class="more-toggle"
			onclick={() => (more = !more)}
			type="button"
		>
			{more ? 'Less' : 'More'}
		</button>
	</div>

	<div class="extra" id="more-controls">
		<label class="slider" for="wght">
			Fine <input
				bind:value={weight}
				id="wght"
				max={axis.max}
				min={axis.min}
				step="1"
				type="range"
			/>

			<output>{weight}</output>
		</label>

		<div class="row">
			<span class="label">Try it</span>

			<label class="slider">
				Size <input bind:value={size} max="160" min="16" type="range" /> <output>{size}</output>
			</label>

			<label class="slider">
				Tracking <input bind:value={tracking} max="120" min="-40" type="range" />

				<output>{tracking}</output>
			</label>

			<label class="toggle"><input bind:checked={guides} type="checkbox" /> Show line boxes</label>
		</div>
	</div>
</div>

<style>
	.bar {
		position: sticky;
		top: env(safe-area-inset-top, 0px);
		z-index: 5;
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
		align-items: center;
		padding-block: 10px;
		background: var(--paper);
		border-bottom: 1px solid var(--rule);
	}
	/* the fine slider and the Try it row sit in the bar's own flow on wide screens */
	.extra {
		display: contents;
	}
	.more-toggle {
		display: none;
	}
	/* on phones the bar floats at the bottom of the screen: weight and version always, the rest behind More */
	@media (max-width: 600px) {
		.bar {
			position: fixed;
			top: auto;
			left: max(8px, env(safe-area-inset-left, 0px));
			right: max(8px, env(safe-area-inset-right, 0px));
			bottom: calc(8px + env(safe-area-inset-bottom, 0px));
			z-index: 20;
			flex-direction: column;
			align-items: stretch;
			gap: 8px;
			max-height: min(70vh, 520px);
			overflow-y: auto;
			padding: 8px;
			border: 1px solid var(--rule);
			border-radius: 14px;
			box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22);
		}
		.bar > .label,
		.versions .label,
		.kbd {
			display: none;
		}
		.segmented label {
			flex: 1;
		}
		.segmented label span {
			text-align: center;
			padding: 8px 6px;
		}
		.versions {
			margin-left: 0;
		}
		.versions .segmented {
			flex: 1;
		}
		.more-toggle {
			display: block;
			padding: 8px 12px;
			border: 1px solid var(--rule);
			border-radius: 8px;
			background: none;
			color: var(--ink);
			font: 500 13px var(--mono);
			cursor: pointer;
		}
		.more-toggle:focus-visible {
			outline: 2px solid var(--guide);
			outline-offset: 2px;
		}
		.extra {
			display: none;
		}
		.more .extra {
			display: grid;
			gap: 10px;
			order: -1; /* opens above the switches, toward the page */
			padding: 4px 4px 10px;
			border-bottom: 1px dashed var(--rule);
		}
		.slider input,
		.row .slider input {
			flex: 1;
			width: auto;
			min-width: 0;
		}
		.row {
			display: grid;
			gap: 10px;
			padding-top: 0;
			border-top: 0;
		}
	}
	.group {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.group:first-of-type {
		margin-left: auto;
	}
	.kbd {
		font: 400 11px var(--mono);
		color: var(--muted);
	}
	.slider {
		display: flex;
		gap: 10px;
		align-items: center;
		font: 400 12px var(--mono);
		color: var(--muted);
	}
	.slider input {
		accent-color: var(--ink);
		width: 180px;
	}
	.slider output {
		min-width: 3ch;
		font-variant-numeric: tabular-nums;
	}
	.row {
		flex-basis: 100%;
		display: flex;
		flex-wrap: wrap;
		gap: 8px 20px;
		align-items: center;
		padding-top: 8px;
		border-top: 1px dashed var(--rule);
	}
	.row .slider input {
		width: 150px;
		accent-color: var(--guide);
	}
	.toggle {
		display: flex;
		gap: 8px;
		align-items: center;
		font: 400 12px var(--mono);
		color: var(--muted);
		cursor: pointer;
	}
	.toggle input {
		accent-color: var(--guide);
	}
</style>
