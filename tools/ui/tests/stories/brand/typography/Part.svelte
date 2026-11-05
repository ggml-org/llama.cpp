<!--
	One use of Llama Sans as a part of its overview page, pinned to that end of the optical size axis (opsz), so
	everything in it, large or small, shows that setting: a rule with the part's name and opsz, a big statement set in
	it, what it's for and its specs, then its sections (children).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		intro,
		name,
		opsz,
		specs,
		statement
	}: {
		children: Snippet;
		intro: string;
		name: string;
		opsz: number;
		specs: string;
		statement: string;
	} = $props();
</script>

<section style:--opsz-pin={`'opsz' ${opsz}`} aria-label="Llama Sans {name}" class="part">
	<header>
		<div class="rule">
			<h2>Llama Sans {name}</h2>

			<span>opsz {opsz}</span>
		</div>

		<p class="statement">{statement}</p>

		<p class="intro">{intro}</p>

		<p class="specs">{specs}</p>
	</header>

	{@render children()}
</section>

<style>
	.part {
		display: grid;
		gap: clamp(48px, 7vw, 96px);
		min-width: 0;
	}
	.part > :global(*) {
		min-width: 0;
	}
	header {
		display: grid;
		gap: clamp(16px, 2vw, 24px);
		width: var(--column);
		margin-inline: auto;
	}
	header > * {
		margin: 0;
	}
	.rule {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 16px;
		padding-top: 20px;
		border-top: 2px solid var(--ink);
	}
	h2 {
		margin: 0;
		font: 700 clamp(28px, 3.4vw, 44px) / 1 var(--headline);
		letter-spacing: -0.01em;
		text-transform: uppercase;
	}
	.rule span {
		font: 500 12px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	.statement {
		max-width: 16ch;
		font: 600 clamp(40px, 6.4vw, 88px) / 1.05 var(--headline);
		letter-spacing: -0.01em;
	}
	.intro {
		max-width: 62ch;
		font: 400 17px / 1.5 var(--ui);
	}
	.specs {
		font: 500 12px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
</style>
