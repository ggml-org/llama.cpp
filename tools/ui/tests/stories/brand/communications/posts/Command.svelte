<!--
	A command in a terminal: the prompt in cinder, the command in Llama Mono, on the ground's inset. The one place a
	piece shows code.
-->
<script lang="ts">
	interface Props {
		command: string;
		/** the type size in units */
		size?: number;
	}

	let { command, size = 0.6 }: Props = $props();
</script>

<pre style:--s="calc(var(--u) * {size})" class="terminal"><span class="prompt">$</span
	> {command}</pre>

<style>
	.terminal {
		margin: 0;
		padding: calc(var(--s) * 0.85) calc(var(--s) * 1);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--s) * 0.55);
		background: var(--p-inset);
		color: var(--p-ink);
		font: 400 var(--s) / 1 var(--mono);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.prompt {
		color: var(--art-cinder);
	}
</style>
