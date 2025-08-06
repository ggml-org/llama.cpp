<script lang="ts">
	interface Props {
		show?: boolean;
		class?: string;
	}

	let {
		show = true,
		class: className = ''
	}: Props = $props();
</script>

{#if show}
	<div class="mt-4 flex items-center justify-center {className}">
		<p class="text-muted-foreground text-xs">
			Press <kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Enter</kbd> to send,
			<kbd class="bg-muted rounded px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new line
		</p>
	</div>
{/if}
