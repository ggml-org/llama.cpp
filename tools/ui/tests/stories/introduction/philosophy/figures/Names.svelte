<!--
	The two lockups, typed rather than placed: U+E002 and the name, Llama in Llama Sans Bold on llama.app's hero blue,
	llama.cpp in Llama Mono Bold on the neutral light and dark grounds
-->
<script lang="ts">
	const HERO =
		'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';
</script>

<div class="names">
	<div style:background={HERO} class="tile llama">
		<span aria-label="Llama" class="lockup" role="img">Llama</span>

		<span class="note">Llama Sans Bold · on colour</span>
	</div>

	<div class="tile light">
		<span aria-label="llama.cpp" class="lockup mono" role="img"
			>llama<span class="cpp">.cpp</span></span
		>

		<span class="note">Llama Mono Bold · on light</span>
	</div>

	<div class="tile dark">
		<span aria-label="llama.cpp" class="lockup mono" role="img"
			>llama<span class="cpp">.cpp</span></span
		>

		<span class="note">Llama Mono Bold · on dark</span>
	</div>
</div>

<style>
	.names {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 12px;
	}

	.tile {
		display: grid;
		grid-template-rows: 1fr auto;
		gap: 24px;
		min-height: 13rem;
		padding: 24px;
		border-radius: 10px;
		container-type: inline-size;
	}

	.llama {
		color: #0f1f33;
	}

	.light {
		background: #fafafa;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.08);
		color: #111111;
	}

	.dark {
		background: #111111;
		box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.08);
		color: #fafafa;
	}

	.lockup {
		align-self: center;
		justify-self: center;
		font-size: min(48px, 15cqi);
		font-weight: 700;
		line-height: 1;
	}

	.lockup.mono {
		font-family: var(--mono);
		font-size: min(40px, 10.5cqi);
	}

	.cpp {
		color: var(--art-cinder);
	}

	.note {
		font: 500 11px var(--mono);
		letter-spacing: 0.04em;
		opacity: 0.6;
	}
</style>
