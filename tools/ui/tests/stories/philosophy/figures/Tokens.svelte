<!-- One decision through the three token tiers to the screen: the primary button's fill, as the CSS has it -->
<script lang="ts">
	import { Button } from '$lib/components/ui/button';

	const tiers = [
		{ name: '--neutral-900', note: 'a raw value (neutral-200 in dark mode)', tier: 'Primitive' },
		{ name: '--action-primary', note: 'named by purpose', tier: 'Semantic' },
		{ name: '--button-primary-background', note: 'one component, one property', tier: 'Component' }
	];
</script>

<ol class="chain">
	{#each tiers as t (t.tier)}
		<li>
			<span class="tier">{t.tier}</span>

			<span class="swatch rounded-control elevated-control"></span>

			<code>{t.name}</code>

			<span class="note">{t.note}</span>
		</li>
	{/each}

	<li>
		<span class="tier">On screen</span>

		<div class="button"><Button>Send</Button></div>

		<code>&lt;Button&gt;</code>

		<span class="note">the real component</span>
	</li>
</ol>

<style>
	.chain {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		margin: 0;
		padding: 0;
		list-style: none;
	}

	li {
		position: relative;
		display: grid;
		gap: 10px;
		align-content: start;
		padding-right: 28px;
	}

	/* an arrow between the tiers */
	li:not(:last-child)::after {
		content: '→';
		position: absolute;
		top: 44px;
		right: 8px;
		color: var(--figure-muted);
	}

	.tier {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.swatch,
	.button {
		display: grid;
		place-items: center start;
		height: 56px;
	}

	.swatch {
		width: 56px;
		background: var(--button-primary-background);
	}

	code {
		font: 500 13px var(--mono);
		overflow-wrap: anywhere;
	}

	.note {
		color: var(--figure-muted);
		font-size: 13px;
	}

	@media (width < 640px) {
		.chain {
			gap: 28px;
		}

		li::after {
			display: none;
		}
	}
</style>
