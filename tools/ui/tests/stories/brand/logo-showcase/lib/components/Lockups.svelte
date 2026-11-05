<!-- The icon as a glyph of Llama Sans (U+E000): a display lockup, two buttons and a line of text. -->
<script lang="ts">
	let { name = 'llama.cpp' }: { name?: string } = $props();
</script>

<section aria-label="The icon in Llama Sans" class="type">
	<span class="kicker">In Llama Sans · U+E000</span>

	<!-- the capital icon, U+E001: the lockup's proportion, typed -->
	<p class="lockup display"> {name}</p>

	<p class="lockup ui">
		<span class="pill"> Open in Llama</span><span class="pill ghost"> llama.app</span>
	</p>

	<p class="lockup text">
		In small text and buttons the icon takes its capital form (U+E001), centred on the capitals, so
		it holds its own in a line of text:  Llama,  LLAMA.CPP.
	</p>
</section>

<style>
	.type {
		display: grid;
		gap: 18px;
	}
	.kicker {
		font: 500 12px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--l-muted);
	}
	.lockup {
		margin: 0;
	}
	.display {
		font:
			700 clamp(48px, 9vw, 112px) / 1.15 'Llama Sans',
			var(--body);
	}
	.ui {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
	}
	.pill {
		display: inline-flex;
		align-items: center;
		padding: 0.5em 1em;
		border-radius: 8px;
		background: var(--l-ink);
		color: var(--l-bg);
		font:
			500 15px / normal 'Llama Sans',
			var(--body);
	}
	.pill.ghost {
		background: none;
		color: var(--l-ink);
		box-shadow: inset 0 0 0 1px var(--l-ink);
	}
	.text {
		max-width: 62ch;
		font:
			400 17px / 1.5 'Llama Sans',
			var(--body);
	}
</style>
