<script lang="ts">
	interface Props {
		isLoading: boolean;
	}

	let { isLoading }: Props = $props();
</script>

<div class="py-4 text-center text-sm text-muted-foreground">
	{#if isLoading}
		Loading resources...
	{:else}
		No resources available
	{/if}
</div>
