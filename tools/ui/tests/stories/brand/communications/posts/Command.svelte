<!--
	A command in a terminal: the prompt in cinder, the command in Llama Mono, on the ground's inset. The one place a
	piece shows code. It stays on one line, as typed, so the piece is laid out wide enough to hold it; the leading is
	the terminal's, not the display type's. A line of config or code goes without the prompt, and so without cinder.
-->
<script lang="ts">
	interface Props {
		command: string;
		/** the `$` in cinder; off for a line of config or code */
		prompt?: boolean;
		/** the type size in units */
		size?: number;
	}

	let { command, prompt = true, size = 0.66 }: Props = $props();
</script>

<pre style:--s="calc(var(--u) * {size})" class="terminal">{#if prompt}<span class="prompt"
			>$ </span>{/if}{command}</pre>

<style>
	.terminal {
		margin: 0;
		padding: calc(var(--s) * 0.55) calc(var(--s) * 1);
		border: 2px solid var(--p-line);
		border-radius: calc(var(--s) * 0.55);
		background: var(--p-inset);
		color: var(--p-ink);
		font: 400 var(--s) / 1.4 var(--mono);
		white-space: pre;
		/* tracked a little tighter, so a line of code holds at the larger sizes */
		letter-spacing: -0.02em;
	}
	.prompt {
		color: var(--art-cinder);
	}
</style>
