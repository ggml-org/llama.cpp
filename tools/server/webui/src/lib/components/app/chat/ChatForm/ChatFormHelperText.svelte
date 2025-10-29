<script lang="ts">
	interface Props {
		class?: string;
		show?: boolean;
	}

	let { class: className = '', show = true }: Props = $props();
</script>

{#if show}
	<div class="mt-4 flex items-center justify-center {className}">
		<p class="text-xs text-muted-foreground">
			Press <kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">Enter</kbd> to send,
			<kbd class="rounded bg-muted px-1 py-0.5 font-mono text-xs">Shift + Enter</kbd> for new line
		</p>
	</div>
{/if}
