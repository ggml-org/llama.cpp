<!-- Llama Sans Display in a stack of headings, each on a 1.15 em line. -->
<div class="display-sample">
	<p class="h1">The shape of a sentence</p>

	<p class="h2">Why round bowls read calmer</p>

	<p class="h3">Terminals, apertures and the space between</p>
</div>

<style>
	.display-sample {
		display: grid;
		gap: 6px;
		font-family: var(--headline);
		line-height: 1.15;
	}
	p {
		margin: 0;
	}
	.h1 {
		font-size: clamp(28px, 3.4vw, 40px);
	}
	.h2 {
		font-size: clamp(22px, 2.4vw, 28px);
	}
	.h3 {
		font-size: 20px;
		color: var(--muted);
	}
</style>
