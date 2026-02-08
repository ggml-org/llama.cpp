<script lang="ts">
	import { RefreshCw, Loader2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		isLoading: boolean;
		onRefresh: () => void;
	}

	let { isLoading, onRefresh }: Props = $props();
</script>

<div class="flex items-center justify-between">
	<h3 class="text-sm font-medium">Available resources</h3>

	<Button
		variant="ghost"
		size="sm"
		class="h-7 w-7 p-0"
		onclick={onRefresh}
		disabled={isLoading}
		title="Refresh resources"
	>
		{#if isLoading}
			<Loader2 class="h-3.5 w-3.5 animate-spin" />
		{:else}
			<RefreshCw class="h-3.5 w-3.5" />
		{/if}
	</Button>
</div>
