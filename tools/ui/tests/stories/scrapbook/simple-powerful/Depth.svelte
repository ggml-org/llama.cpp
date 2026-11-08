<!--
	Proposal B · Three steps deep: how far in someone goes, left to right. Just ask, with an everyday question; choose in
	plain words, with the real RadioGroup; set every part, in exact terms. Most people stay on the first step, and the
	cinder mark is on it.
-->
<script lang="ts">
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';

	const plain = [
		{ id: 'quick', name: 'Quick answers' },
		{ id: 'balanced', name: 'Balanced' },
		{ id: 'careful', name: 'Take more time, think it through' }
	];

	const exact = [
		['model', 'qwen3.8-27b-Q4_K_M.gguf'],
		['--mmproj', 'mmproj-F16.gguf'],
		['--ctx-size', '32768'],
		['--temp', '0.8'],
		['--n-gpu-layers', '99']
	];
</script>

<ol class="depth">
	<li class="step card-surface rounded-card">
		<p class="n first">1 · Just ask</p>

		<div class="stage" inert>
			<p class="bubble you rounded-panel">Help me reply to my landlord about the broken heating</p>

			<p class="bubble reply">Of course. Here is a short, polite reply you can send…</p>
		</div>

		<p class="note">Where most people stay.</p>
	</li>

	<li class="step card-surface rounded-card">
		<p class="n">2 · Choose in plain words</p>

		<div class="stage" inert>
			<RadioGroup.Root value="balanced">
				{#each plain as option (option.id)}
					<div class="option">
						<RadioGroup.Item id="depth-{option.id}" value={option.id} />

						<Label for="depth-{option.id}">{option.name}</Label>
					</div>
				{/each}
			</RadioGroup.Root>
		</div>

		<p class="note">When the first choice isn't quite right.</p>
	</li>

	<li class="step card-surface rounded-card">
		<p class="n">3 · Set every part</p>

		<dl class="stage exact">
			{#each exact as [k, v] (k)}
				<dt>{k}</dt>

				<dd>{v}</dd>
			{/each}
		</dl>

		<p class="note">For whoever wants it, in exact terms.</p>
	</li>
</ol>

<style>
	.depth {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.step {
		display: grid;
		grid-template-rows: auto 1fr auto;
		gap: 12px;
		margin: 0;
		padding: 16px;
	}

	p {
		margin: 0;
	}

	.n {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.n.first {
		color: var(--art-cinder);
	}

	.stage {
		display: grid;
		gap: 8px;
		align-content: center;
		min-height: 150px;
	}

	.bubble {
		font-size: 13px;
		line-height: 1.45;
	}

	.you {
		justify-self: end;
		max-width: 90%;
		padding: 8px 10px;
		background: var(--surface-lifted, color-mix(in oklab, var(--fg-default) 6%, transparent));
	}

	.reply {
		color: var(--figure-muted);
	}

	.option {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.exact {
		grid-template-columns: auto minmax(0, 1fr);
		gap: 6px 10px;
		margin: 0;
		font: 500 11px var(--mono);
	}

	dt {
		color: var(--figure-muted);
	}

	dd {
		margin: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.note {
		color: var(--figure-muted);
		font-size: 13px;
	}

	@media (width < 720px) {
		.depth {
			grid-template-columns: minmax(0, 1fr);
		}

		.stage {
			min-height: 0;
		}
	}
</style>
