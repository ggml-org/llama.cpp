<!--
	The same real controls three times: as Llama UI ships them, then with two other accents set through the component
	tokens alone, as an app built on Llama UI would
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';

	const accent = (c: string) =>
		[
			`--button-primary-background: ${c}`,
			`--button-primary-background-hover: color-mix(in oklch, ${c} 90%, transparent)`,
			`--button-primary-edge: color-mix(in oklab, ${c}, black 25%)`,
			'--button-primary-text: white',
			`--badge-primary-background: ${c}`,
			'--badge-primary-text: white',
			`--switch-track-on: ${c}`
		].join(';');

	const looks = [
		{ label: 'As shipped', style: '' },
		{ label: 'Night sky', style: accent('var(--art-night-sky)') },
		{ label: 'Cinder', style: accent('var(--art-cinder)') }
	];
</script>

<div class="looks">
	{#each looks as look (look.label)}
		<div class="look card-surface rounded-card" style={look.style}>
			<span class="label">{look.label}</span>

			<div class="row">
				<Button>Load model</Button>

				<Button variant="secondary">Settings</Button>
			</div>

			<div class="row">
				<Switch aria-label="Keep in memory" checked />

				<span aria-hidden="true" class="text-body-sm">Keep in memory</span>

				<Badge class="ml-auto">Q4_K_M</Badge>
			</div>
		</div>
	{/each}
</div>

<style>
	.looks {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 16px;
	}

	.look {
		display: grid;
		gap: 16px;
		padding: 20px;
	}

	.label {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}
</style>
