<!-- Llama Sans Display in its element: large, tight, stacked type. Everything uses the font's own line spacing. -->
<script lang="ts">
	import DisplayGallery from './DisplayGallery.svelte';
</script>

<div class="display">
	<section aria-label="Banner" class="hero">
		<span class="kicker">Llama Sans Display</span>

		<h2 class="hero-title">Quiet letters for loud ideas.</h2>

		<p class="hero-sub">
			Tight tracking, firm kerning and 1.15 em lines, made for headlines that hold together.
		</p>
	</section>

	<section aria-label="Headings" class="stack">
		<p class="eyebrow">Editorial headings</p>

		<h2 class="h1">The shape of a sentence</h2>

		<h3 class="h2">Why round bowls read calmer</h3>

		<h4 class="h3">Terminals, apertures and the space between</h4>

		<p class="dek">
			Three heading sizes, one set of outlines. Each line sits on a 1.15 em line-height, so stacked
			headings close up without extra CSS.
		</p>
	</section>

	<div class="pair">
		<section aria-label="Figures" class="figures">
			<span class="big">400</span>

			<span class="to">to</span>

			<span class="big">700</span>

			<p class="cap">One weight axis, every step in between.</p>
		</section>

		<blockquote class="quote">
			<p>“Type is a voice with the volume turned down. Display turns it back up.”</p>
		</blockquote>
	</div>

	<section aria-label="Poster" class="poster">
		<span class="poster-word">AVATAR</span>

		<span class="poster-word outline">Wave Typo</span>

		<span class="poster-meta">Kerning at 115% · pairs like AV, Wa and Ty close up</span>
	</section>

	<DisplayGallery />
</div>

<style>
	.display {
		display: grid;
		gap: 28px;
		font-family: var(--display);
	}
	h2,
	h3,
	h4,
	p,
	blockquote {
		margin: 0;
	}
	.hero {
		display: grid;
		gap: 14px;
		padding: clamp(28px, 5vw, 64px);
		border-radius: 14px;
		background: var(--ink);
		color: var(--paper);
	}
	.kicker {
		font: 500 13px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		opacity: 0.7;
	}
	.hero-title {
		font: clamp(44px, 9vw, 120px) / 1.15 var(--display);
		font-weight: max(var(--w), 600);
		text-wrap: balance;
		max-width: 12ch;
	}
	.hero-sub {
		max-width: 46ch;
		font: 400 clamp(17px, 2vw, 22px) / 1.15 var(--display);
		opacity: 0.8;
	}
	.stack {
		display: grid;
		gap: 6px;
		padding-block: 8px;
	}
	.eyebrow {
		font: 500 13px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
		margin-bottom: 6px;
	}
	.h1 {
		font: var(--w) clamp(36px, 6vw, 72px) / 1.15 var(--display);
		text-wrap: balance;
	}
	.h2 {
		font: var(--w) clamp(26px, 4vw, 44px) / 1.15 var(--display);
		text-wrap: balance;
	}
	.h3 {
		font: var(--w) clamp(20px, 2.6vw, 28px) / 1.15 var(--display);
		color: var(--muted);
		text-wrap: balance;
	}
	.dek {
		max-width: 60ch;
		margin-top: 8px;
		font: 400 16px / 1.5 var(--body);
		color: var(--muted);
	}
	.pair {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 20px;
	}
	.figures,
	.quote {
		padding: 24px;
		border: 1px solid var(--rule);
		border-radius: 14px;
		background: var(--sheet);
		min-width: 0;
	}
	.figures {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 14px;
	}
	.big {
		font: var(--w) clamp(64px, 10vw, 128px) / 1.15 var(--display);
		font-variant-numeric: tabular-nums;
	}
	.to {
		font: 400 18px var(--mono);
		color: var(--muted);
	}
	.cap {
		flex-basis: 100%;
		font: 400 14px / 1.5 var(--body);
		color: var(--muted);
	}
	.quote p {
		font: var(--w) clamp(26px, 3.4vw, 40px) / 1.15 var(--display);
		text-wrap: balance;
	}
	.poster {
		display: grid;
		gap: 0;
		padding: clamp(24px, 4vw, 48px);
		border-radius: 14px;
		background: var(--llama-hero); /* llama.app's hero: the day sky, the night sky in dark mode */
		color: var(--llama-hero-ink);
		overflow: hidden;
	}
	.poster-word {
		font: 700 clamp(56px, 13vw, 180px) / 1.15 var(--display);
		overflow-wrap: anywhere;
	}
	.poster-word.outline {
		color: transparent;
		-webkit-text-stroke: 2px var(--llama-hero-ink); /* not currentColor: the fill is transparent */
	}
	.poster-meta {
		margin-top: 10px;
		font: 500 13px var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
</style>
