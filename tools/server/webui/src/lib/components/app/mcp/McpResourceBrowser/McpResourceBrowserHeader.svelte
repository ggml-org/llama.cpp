<script lang="ts">
	import { RefreshCw, Loader2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import SearchInput from '../../forms/SearchInput.svelte';

	interface Props {
		isLoading: boolean;
		onRefresh: () => void;
		onSearch?: (query: string) => void;
		searchQuery?: string;
	}

	let { isLoading, onRefresh, onSearch, searchQuery = '' }: Props = $props();
</script>

<div class="flex flex-col gap-2">
	<div class="mb-2 flex items-center gap-4">
		<SearchInput
			placeholder="Search resources..."
			value={searchQuery}
			onInput={(value) => onSearch?.(value)}
		/>

		<Button
			variant="ghost"
			size="sm"
			class="h-8 w-8 p-0"
			onclick={onRefresh}
			disabled={isLoading}
			title="Refresh resources"
		>
			{#if isLoading}
				<Loader2 class="h-4 w-4 animate-spin" />
			{:else}
				<RefreshCw class="h-4 w-4" />
			{/if}
		</Button>
	</div>

	<h3 class="text-sm font-medium">Available resources</h3>
</div>
