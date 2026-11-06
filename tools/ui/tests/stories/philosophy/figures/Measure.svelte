<!-- One real button on the 4px grid, with the roles that set each of its measurements -->
<script lang="ts">
	import { Play } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	const specs = [
		{ role: 'h-control-md', value: '36', what: 'control height' },
		{ role: 'rounded-control', value: '8', what: 'corner' },
		{ role: 'size-icon-md', value: '16', what: 'icon' },
		{ role: 'text-label', value: '14', what: 'label, on a 20px line, weight 500' },
		{ role: 'spacing', value: '4', what: 'the grid it all sits on' }
	];
</script>

<div class="measure">
	<div class="stage rounded-card shadow-(--elevation-inset)">
		<div class="button">
			<Button><Play />Run model</Button>
		</div>
	</div>

	<dl>
		{#each specs as s (s.role)}
			<div>
				<dt>{s.value}<span>px</span></dt>

				<dd>{s.what} <code>{s.role}</code></dd>
			</div>
		{/each}
	</dl>
</div>

<style>
	.measure {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
		gap: clamp(24px, 4vw, 48px);
		align-items: center;
	}

	/* the button at twice its size on the 4px grid, also doubled */
	.stage {
		display: grid;
		place-items: center;
		min-height: 14rem;
		background-color: var(--surface-muted);
		background-image:
			linear-gradient(
				to right,
				color-mix(in oklab, var(--fg-default) 7%, transparent) 1px,
				transparent 1px
			),
			linear-gradient(
				to bottom,
				color-mix(in oklab, var(--fg-default) 7%, transparent) 1px,
				transparent 1px
			);
		background-position: center;
		background-size: 8px 8px;
	}

	.button {
		zoom: 2;
	}

	dl {
		display: grid;
		gap: 12px;
		margin: 0;
	}

	dl div {
		display: grid;
		grid-template-columns: 5rem 1fr;
		align-items: baseline;
		padding-bottom: 12px;
		border-bottom: 1px solid var(--line-default);
	}

	dt {
		font-size: 28px;
		font-weight: 600;
		line-height: 1;
	}

	dt span {
		margin-left: 2px;
		color: var(--figure-muted);
		font-size: 13px;
		font-weight: 400;
	}

	dd {
		margin: 0;
		color: var(--figure-muted);
		font-size: 14px;
	}

	code {
		margin-left: 4px;
		color: var(--fg-default);
		font: 12px var(--mono);
	}
</style>
