<!--
	The overview's opening: the Llama icon (typed from the font, U+E000) over the family's name at full size, on
	llama.app's navy in both themes, with a kicker above and the tagline under it.
-->
<script lang="ts">
	let { kicker, tagline }: { kicker: string; tagline: string[] } = $props();
</script>

<header class="hero">
	<p class="kicker">{kicker}</p>

	<p aria-hidden="true" class="icon"></p>

	<h1>Llama<br />Sans</h1>

	<p class="tagline">
		{#each tagline as line (line)}<span>{line}</span>{/each}
	</p>
</header>

<style>
	.hero {
		display: grid;
		justify-items: center;
		gap: clamp(20px, 3vw, 36px);
		padding: clamp(56px, 9vw, 120px) 16px clamp(64px, 10vw, 136px);
		background: var(--llama-navy);
		color: #fff;
		text-align: center;
	}
	.hero > * {
		margin: 0;
	}
	.kicker {
		font: 500 12px var(--mono);
		letter-spacing: 0.12em;
		text-transform: uppercase;
		opacity: 0.7;
	}
	.icon {
		font: 700 clamp(40px, 6vw, 72px) / 1 var(--headline);
		color: var(--llama-sky);
	}
	h1 {
		font: 700 clamp(88px, 19vw, 264px) / 0.84 var(--headline);
		letter-spacing: -0.02em;
	}
	.tagline {
		display: grid;
		gap: 2px;
		font: 400 clamp(20px, 2.6vw, 30px) / 1.25 var(--ui);
		opacity: 0.85;
	}
</style>
