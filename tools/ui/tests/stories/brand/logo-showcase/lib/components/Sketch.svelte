<!-- The original pencil sketch (src/lib/assets/sketch.webp) as a small card: a thumbnail and a button that opens it
     full size. Without the file nothing renders. -->
<script lang="ts">
	let { label = 'View the original sketch' }: { label?: string } = $props();

	const sketches = import.meta.glob('../assets/sketch.{jpg,jpeg,png,webp,avif}', {
		eager: true,
		import: 'default',
		query: '?url'
	}) as Record<string, string>;
	const sketch = Object.values(sketches)[0];
	let dialog = $state<HTMLDialogElement>();
</script>

{#if sketch}
	<aside class="sketch-card">
		<img alt="" src={sketch} />

		<div class="sketch-text">
			<strong>Where it started</strong>

			<span>A pencil sketch of a plus drawn into an L, before the bolt.</span>

			<button class="sketch-open" onclick={() => dialog?.showModal()} type="button">{label}</button>
		</div>
	</aside>

	<dialog bind:this={dialog} aria-label="The original pencil sketch" class="sketch" closedby="any">
		<img
			alt="A sketchbook page in pencil: a slanted plus, the same plus with its top and right arms shaded, the two-blade icon in a rounded square, an upright plus with two arms shaded, L shapes built on a grid and an L drawn in perspective"
			src={sketch}
		/>

		<form method="dialog"><button class="sketch-close">Close</button></form>
	</dialog>
{/if}

<style>
	.sketch-card {
		display: flex;
		gap: 16px;
		align-items: center;
		padding: 14px;
		border-radius: 12px;
		background: var(--l-panel);
	}
	.sketch-card img {
		width: 96px;
		height: 72px;
		object-fit: cover;
		border-radius: 8px;
	}
	.sketch-text {
		display: grid;
		gap: 6px;
		justify-items: start;
		font:
			400 14px / 1.5 'Llama Sans',
			var(--body);
		color: var(--l-muted);
	}
	.sketch-text strong {
		color: var(--l-ink);
		font:
			600 15px / normal 'Llama Sans',
			var(--body);
	}
	.sketch-open,
	.sketch-close {
		margin-top: 4px;
		padding: 0.4em 0.9em;
		border: 0;
		border-radius: 8px;
		background: none;
		box-shadow: inset 0 0 0 1px var(--l-ink);
		color: var(--l-ink);
		font:
			500 15px / normal 'Llama Sans',
			var(--body);
		cursor: pointer;
	}
	.sketch-open:focus-visible,
	.sketch-close:focus-visible {
		outline: 2px solid var(--l-ink);
		outline-offset: 2px;
	}
	.sketch {
		max-width: min(92vw, 900px);
		max-height: 92vh;
		padding: 12px;
		border: 0;
		border-radius: 12px;
		background: var(--l-bg);
		color: var(--l-ink);
	}
	.sketch::backdrop {
		background: rgba(0, 0, 0, 0.6);
	}
	.sketch img {
		display: block;
		max-width: 100%;
		max-height: calc(92vh - 80px);
		margin: 0 auto;
		border-radius: 6px;
	}
	.sketch form {
		display: flex;
		justify-content: flex-end;
		margin-top: 10px;
	}
</style>
