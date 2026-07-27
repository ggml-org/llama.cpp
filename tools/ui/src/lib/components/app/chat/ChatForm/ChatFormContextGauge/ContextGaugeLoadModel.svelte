<script lang="ts">
	import { Loader2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		modelId: string | null;
		isLoading: boolean;
		onLoad: () => void;
	}

	let { modelId, isLoading, onLoad }: Props = $props();
</script>

{#if modelId !== null && !isLoading}
	<div class="flex flex-col gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground">
		<span>Available context size is only visible once the model is loaded.</span>
		<Button size="sm" variant="secondary" class="self-start" onclick={onLoad}>Load model</Button>
	</div>
{:else if isLoading}
	<div class="flex items-center gap-2 border-t border-border/50 pt-2 text-xs text-muted-foreground">
		<Loader2 class="h-3.5 w-3.5 animate-spin" />
		<span>Loading model...</span>
	</div>
{/if}
