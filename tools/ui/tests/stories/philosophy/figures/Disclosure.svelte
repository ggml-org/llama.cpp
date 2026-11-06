<!--
	Simple by default, deep when you want: the same app at three depths, with real controls. The first run asks for one
	click; settings add a system prompt; the advanced panel holds sampling and decoding. Each is uncovered by asking,
	shown left to right. The first run's step number carries the cinder.
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';

	const sampling = [
		['temperature', '0.8'],
		['top_p', '0.95'],
		['min_p', '0.05'],
		['repeat_penalty', '1.1']
	];
</script>

<div class="depths">
	<section class="depth card-surface rounded-card first">
		<span class="step">1 · First run</span>

		<p class="title">A model for this computer</p>

		<p class="note">Small, sees and hears, works offline</p>

		<div class="row">
			<Badge variant="outline">2.4 GB</Badge><Button size="sm">Download and start</Button>
		</div>
	</section>

	<section class="depth card-surface rounded-card">
		<span class="step">2 · When you ask</span>

		<p class="title">Settings</p>

		<div class="prompt rounded-panel bg-surface-lifted">
			You are a patient tutor. Explain step by step.
		</div>

		<label class="row"><Switch checked /><span>Keep the model loaded</span></label>
	</section>

	<section class="depth card-surface rounded-card">
		<span class="step">3 · When you dig</span>

		<p class="title">Advanced</p>

		<dl>
			{#each sampling as [key, value] (key)}<dt>{key}</dt>

				<dd>{value}</dd>{/each}
		</dl>

		<label class="row"><Switch /><span>Speculative decoding</span></label>
	</section>
</div>

<style>
	.depths {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
		gap: 8px;
		align-items: start;
	}

	.depth {
		display: grid;
		gap: 10px;
		padding: 16px;
	}

	.step {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.first .step {
		color: var(--art-cinder);
	}

	p {
		margin: 0;
	}

	.title {
		font-size: 17px;
		font-weight: 600;
	}

	.note {
		margin-top: -6px;
		color: var(--figure-muted);
		font-size: 13px;
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		font-size: 13px;
	}

	.prompt {
		padding: 8px 10px;
		color: var(--figure-muted);
		font-size: 13px;
	}

	dl {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 4px 12px;
		margin: 0;
		font: 500 12px var(--mono);
	}

	dt {
		color: var(--figure-muted);
	}

	dd {
		margin: 0;
	}
</style>
