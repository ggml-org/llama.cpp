<script lang="ts">
	import type { ModelsProviderGroup } from '../utils';
	import ModelsManagerModelsListItem from './ModelsManagerModelsListItem.svelte';

	interface Props {
		onSelect: (backendId: string | null) => void;
		providers: ModelsProviderGroup[];
		selectedBackendId: string | null;
		totalCount: number;
	}

	let { onSelect, providers, selectedBackendId, totalCount }: Props = $props();
</script>

<div class="flex h-full min-h-0 flex-col gap-1">
	<p class="px-2 py-1 text-[13px] font-semibold text-muted-foreground/70 select-none">Providers</p>

	<div class="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
		<ModelsManagerModelsListItem
			backendId={null}
			count={totalCount}
			isActive={selectedBackendId === null}
			isLocal={false}
			label="All providers"
			onSelect={() => onSelect(null)}
		/>

		{#each providers as provider (provider.backendId)}
			<ModelsManagerModelsListItem
				backendId={provider.backendId}
				count={provider.count}
				isActive={selectedBackendId === provider.backendId}
				isLocal={provider.isLocal}
				label={provider.label}
				onSelect={() => onSelect(provider.backendId)}
			/>
		{/each}
	</div>
</div>
