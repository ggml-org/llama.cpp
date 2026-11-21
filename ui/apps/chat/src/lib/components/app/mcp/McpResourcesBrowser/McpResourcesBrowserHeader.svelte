<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Loader2, RefreshCw } from '@lucide/svelte';
	import { SearchInput } from '#lib/components/app/forms/index.ts';
	import { ICON_CLASS_DEFAULT } from '#lib/constants/index.ts';

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
			onInput={(value) => onSearch?.(value)}
			placeholder="Search resources..."
			value={searchQuery}
		/>

		<Button
			class="size-8 p-0"
			disabled={isLoading}
			onclick={onRefresh}
			size="sm"
			title="Refresh resources"
			variant="ghost"
		>
			{#if isLoading}
				<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin" />
			{:else}
				<RefreshCw class={ICON_CLASS_DEFAULT} />
			{/if}
		</Button>
	</div>

	<h3 class="text-label">Available resources</h3>
</div>
