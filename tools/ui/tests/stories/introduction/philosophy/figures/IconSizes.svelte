<!-- The Llama icon from favicon to poster size, then typed as a letter: U+E002 in Llama Sans, inside a real button -->
<script lang="ts">
	import { PATHS, SIZE } from '../../../brand/logo-showcase/lib/logo';
	import { Button } from '$lib/components/ui/button';

	const sizes = [14, 16, 24, 32, 48, 96, 160];
</script>

<div class="icons">
	<div class="row">
		{#each sizes as size (size)}
			<div class="size">
				<svg aria-hidden="true" height={size} viewBox="0 0 {SIZE} {SIZE}" width={size}>
					{#each PATHS as d (d)}<path {d} />{/each}
				</svg>

				<span>{size}</span>
			</div>
		{/each}
	</div>

	<div class="typed">
		<p class="line">Download <span class="name">Llama</span> for your Mac.</p>

		<Button>Llama for macOS</Button>
	</div>
</div>

<style>
	.icons {
		display: grid;
		gap: 40px;
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		gap: clamp(20px, 4vw, 40px);
		align-items: flex-end;
	}

	.size {
		display: grid;
		gap: 12px;
		justify-items: center;
	}

	svg {
		fill: currentColor;
	}

	.size span {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
	}

	.typed {
		display: flex;
		flex-wrap: wrap;
		gap: 16px 32px;
		align-items: center;
		padding-top: 28px;
		border-top: 1px solid var(--line-default);
	}

	.line {
		margin: 0;
		font-size: 24px;
	}

	.name {
		font-weight: 700;
	}
</style>
