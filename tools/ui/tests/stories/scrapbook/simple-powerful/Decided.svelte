<!--
	Proposal C · Decided for you: an everyday conversation on top (a photo of a letter and a question about it), and
	under it, with the lid lifted, what the app decided so it could just work: each decision in plain words, with the
	exact setting a power user can change beside it.
-->
<script lang="ts">
	import { FileImage } from '@lucide/svelte';

	const decisions = [
		['A model that fits this laptop', 'model.Q4_K_M.gguf · 4.6 GB'],
		['It can read the photo', '--mmproj mmproj-F16.gguf'],
		['It remembers a long letter', '--ctx-size 32768'],
		['Balanced, not too creative', '--temp 0.7'],
		['As fast as this laptop allows', '--n-gpu-layers 99']
	];
</script>

<div class="decided">
	<div class="chat card-surface rounded-card" inert>
		<div class="you">
			<span class="file rounded-control"><FileImage aria-hidden="true" size={14} /> letter.jpg</span
			>

			<p class="bubble rounded-panel">What does this letter from the council want me to do?</p>
		</div>

		<p class="reply">
			It's a reminder that your parking permit ends on 30 November. To renew it, fill in the form on
			the back and…
		</p>
	</div>

	<div class="lid rounded-card">
		<div class="cols">
			<p class="head first">Decided for you</p>

			<p class="head">Yours to change</p>
		</div>

		<ul>
			{#each decisions as [plain, exact] (plain)}
				<li>
					<span>{plain}</span>

					<span class="exact">{exact}</span>
				</li>
			{/each}
		</ul>
	</div>
</div>

<style>
	.decided {
		display: grid;
		gap: 8px;
	}

	.chat {
		display: grid;
		gap: 12px;
		padding: 16px;
	}

	.you {
		display: grid;
		justify-items: end;
		gap: 6px;
	}

	.file {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 8px;
		border: 1px solid var(--line-default);
		font-size: 12px;
	}

	p {
		margin: 0;
	}

	.bubble {
		max-width: 80%;
		padding: 8px 12px;
		background: color-mix(in oklab, var(--fg-default) 6%, transparent);
		font-size: 14px;
	}

	.reply {
		max-width: 90%;
		font-size: 14px;
		line-height: 1.5;
	}

	.lid {
		display: grid;
		gap: 8px;
		padding: 14px 16px;
		border: 1.5px dashed color-mix(in oklab, var(--fg-muted) 50%, transparent);
	}

	.cols,
	li {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		gap: 12px;
	}

	.head {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.head.first {
		color: var(--art-cinder);
	}

	ul {
		display: grid;
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	li {
		align-items: baseline;
		margin: 0;
		font-size: 14px;
	}

	.exact {
		overflow: hidden;
		color: var(--figure-muted);
		font: 500 12px var(--mono);
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	@media (width < 640px) {
		.cols {
			display: none;
		}

		li {
			grid-template-columns: minmax(0, 1fr);
			gap: 2px;
		}
	}
</style>
