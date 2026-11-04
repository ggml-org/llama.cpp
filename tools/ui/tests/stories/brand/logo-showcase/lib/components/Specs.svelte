<!-- The icon's numbers at a glance. -->
<script lang="ts">
	import { specs } from '../blueprint';
</script>

<dl class="specs">
	{#each specs as [k, v] (k)}
		<div>
			<dt>{k}</dt>

			<dd>{v}</dd>
		</div>
	{/each}
</dl>

<style>
	.specs {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
		gap: 14px 24px;
		margin: 0;
	}
	.specs div {
		display: grid;
		gap: 3px;
		min-width: 0;
		padding-top: 10px;
		border-top: 1px solid var(--l-rule);
	}
	dt {
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--l-muted);
	}
	dd {
		margin: 0;
		font:
			400 15px / normal 'Llama Sans UI',
			var(--body);
	}
</style>
