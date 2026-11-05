<!--
	The overview's opening: the Llama icon (typed from the font, U+E000) over the font's name at full size, set in that
	font (face), on a quiet neutral gradient that follows the theme, with a kicker above and the tagline under it.
-->
<script lang="ts">
	let {
		face = 'var(--headline)',
		kicker,
		tagline,
		title
	}: { face?: string; kicker: string; tagline: string[]; title: string[] } = $props();
</script>

<header style:--face={face} class="hero">
	<p class="kicker">{kicker}</p>

	<p aria-hidden="true" class="icon"></p>

	<h1>
		{#each title as line, i (line)}{#if i}<br />{/if}{line}{/each}
	</h1>

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
		background: linear-gradient(
			160deg,
			var(--sheet) 0%,
			var(--paper) 100%
		); /* neutral, one hue, follows the theme */
		color: var(--ink);
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
		font: 700 clamp(40px, 6vw, 72px) / 1 var(--face);
		opacity: 0.85;
	}
	h1 {
		font: 700 clamp(64px, 14vw, 208px) / 0.9 var(--face);
		letter-spacing: -0.02em;
	}
	.tagline {
		display: grid;
		gap: 2px;
		font: 400 clamp(20px, 2.6vw, 30px) / 1.25 var(--ui);
		opacity: 0.85;
	}
</style>
