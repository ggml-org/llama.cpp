<!-- A post's bottom row: who it is from on the left, the promise on the right. -->
<script lang="ts">
	interface Props {
		left: string;
		right?: string;
		size?: number;
	}

	let { left, right, size = 0.58 }: Props = $props();
</script>

<footer style:font-size="calc(var(--u) * {size})" class="row">
	<span>{left}</span>

	{#if right}<span>{right}</span>{/if}
</footer>

<style>
	.row {
		display: flex;
		justify-content: space-between;
		gap: var(--u);
		margin-top: auto;
		color: var(--p-muted);
		line-height: 1;
		white-space: nowrap;
	}
</style>
