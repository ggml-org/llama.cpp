<!--
	The footer as a signature: the icon and a name in small type over a hairline that spans the piece, as the
	references sign off. It replaces the two-sided footer where a piece has one voice.
-->
<script lang="ts">
	import { ICON } from '../../brand/communications/posts/post';

	interface Props {
		/** the name after the icon */
		name?: string;
		/** set to the right, at least 2u from the name */
		right?: string;
	}

	let { name = 'Llama', right }: Props = $props();
</script>

<footer class="signature">
	<span class="name"><span aria-hidden="true" class="icon">{ICON}</span>{name}</span>

	{#if right}<span class="right">{right}</span>{/if}
</footer>

<style>
	.signature {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: calc(var(--u) * 2);
		margin-top: auto;
		padding-bottom: calc(var(--u) * 0.4);
		border-bottom: 2px solid var(--p-line);
		color: var(--p-ink);
		font: 500 calc(var(--u) * 0.58) / 1 var(--sans);
		white-space: nowrap;
	}
	.icon {
		margin-right: calc(var(--u) * 0.3);
		font-weight: 700;
	}
	.right {
		color: var(--p-muted);
		font-family: var(--mono);
		font-weight: 400;
	}
</style>
