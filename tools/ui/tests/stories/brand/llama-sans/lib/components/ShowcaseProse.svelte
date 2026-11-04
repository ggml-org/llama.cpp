<!-- Llama Sans UI for reading: running text with its line-height set to 1.5 in CSS (the font's own is 1.25, for
     controls). The line box stays centred on the cap height, so the extra leading falls evenly above and below. -->
<script lang="ts">
	import ProseGallery from './ProseGallery.svelte';
</script>

<div class="prose-showcase">
	<article class="article">
		<header class="article-head">
			<p class="kicker">Notes on type · 4 min read</p>

			<h2 class="title">One face for the interface and the page</h2>

			<p class="standfirst">
				The labels on a button and the paragraphs of an article can share a font. Spacing does most
				of the work; line height does the rest.
			</p>
		</header>

		<div class="body">
			<p>
				At reading sizes the eye moves along a line in small jumps, taking in a few letters at a
				time. If those letters sit too close, words start to blur into each other; too far apart,
				and they stop reading as words at all. UI is spaced like the system fonts at their text
				sizes, and its kerning follows them too, so pairs like “To” and “AV” stay tidy without
				pinching.
			</p>

			<p>
				Line spacing comes from the page, not the font. Set paragraphs to <code
					>line-height: 1.5</code
				> and long passages keep an even grey texture instead of dark bands, while buttons and chips keep
				the font’s own tight, centred line.
			</p>

			<blockquote>
				<p>Good text type disappears. You notice the words, not the letters.</p>
			</blockquote>

			<p>A few things the same font covers:</p>

			<ul>
				<li>Articles, documentation and long-form writing on screens.</li>

				<li>Small sizes, from captions at 13 px to body copy at 16–20 px.</li>

				<li>Mixed figures and letters, like 4 November 2026 or 974 glyphs.</li>
			</ul>

			<figure class="figure">
				<div aria-hidden="true" class="figure-art">Aa</div>

				<figcaption>Figure 1. Display’s outlines, spaced for reading and interfaces.</figcaption>
			</figure>

			<p class="footnote">
				¹ UI and Display share their outlines, but for UI’s deeper descenders; spacing, kerning and
				line spacing set them apart.
			</p>
		</div>
	</article>

	<ProseGallery />
</div>

<style>
	.prose-showcase {
		display: grid;
		gap: 40px;
	}
	.article {
		display: grid;
		gap: 28px;
		max-width: 68ch;
		font-family: var(--display);
	}
	p,
	h2,
	blockquote,
	ul,
	figure {
		margin: 0;
	}
	.article-head {
		display: grid;
		gap: 10px;
	}
	.kicker {
		font: 500 13px / normal var(--display);
		color: var(--muted);
		letter-spacing: 0.02em;
	}
	.title {
		font: 600 clamp(28px, 4vw, 40px) / normal var(--display);
		text-wrap: balance;
	}
	.standfirst {
		font: 400 clamp(18px, 2.2vw, 21px) / 1.4 var(--display);
		color: var(--muted);
	}
	.body {
		display: grid;
		gap: 1em;
		font: var(--w) 17px / 1.5 var(--display);
	}
	code {
		font: 400 0.88em var(--mono);
		padding: 0.05em 0.35em;
		border-radius: 4px;
		background: var(--sheet);
		border: 1px solid var(--rule);
	}
	blockquote {
		padding: 4px 0 4px 18px;
		border-left: 3px solid var(--guide);
	}
	blockquote p {
		font: 500 20px / 1.4 var(--display); /* no italic yet: never let the browser fake one */
	}
	ul {
		display: grid;
		gap: 0.35em;
		padding-left: 1.2em;
	}
	.figure {
		display: grid;
		gap: 8px;
	}
	.figure-art {
		display: grid;
		place-items: center;
		aspect-ratio: 16 / 6;
		max-width: 100%;
		border-radius: 10px;
		background: var(--sheet);
		border: 1px solid var(--rule);
		font: var(--w) clamp(64px, 12vw, 140px) / 1 var(--display);
	}
	figcaption {
		font: 400 13px / 1.5 var(--display);
		color: var(--muted);
	}
	.footnote {
		font: 400 13px / 1.5 var(--display);
		color: var(--muted);
	}
</style>
