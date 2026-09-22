<script lang="ts">
	import type { ModelsTableGroup } from './utils';
	import { Heart, Power } from '@lucide/svelte';
	import {
		Logo,
		ModelAvatar,
		ModelId,
		ModelLoadControl,
		ModelRowActions,
		ModelsSection
	} from '$lib/components/app';
	import { Input } from '$lib/components/ui/input';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';

	interface Props {
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		selectedId: string | null;
		summary: string;
	}

	let {
		filter = $bindable(''),
		groups,
		isFavorite,
		onSelect,
		selectedId,
		summary
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	const rowGrid = 'grid grid-cols-[minmax(0,1fr)_3rem_4.5rem] items-center gap-3';

	function stateOf(option: ModelOption): ServerModelStatus | null {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	}
</script>

{#snippet row(option: ModelOption)}
	{@const status = stateOf(option)}
	{@const isOperationInProgress = modelsStore.status.isOperationInProgress(option.model)}
	{@const isLoading = status === ServerModelStatus.LOADING || isOperationInProgress}
	{@const isLoaded =
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
		!isOperationInProgress}
	{@const isFailed = status === ServerModelStatus.FAILED}
	{@const isSleeping = status === ServerModelStatus.SLEEPING}
	{@const favorite = isFavorite(option)}

	<div>
		<div
			class={[
				rowGrid,
				'cursor-pointer rounded-md px-2 py-2.5 transition',
				selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
			]}
			onclick={() => onSelect(option)}
			onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
			role="button"
			tabindex="0"
		>
			<span class="flex min-w-0 items-center gap-3">
				<ModelAvatar {option} showBaseModelAvatar size="size-9" />

				<ModelId
					aliases={option.aliases}
					class="min-w-0 flex-1"
					modalities={option.modalities}
					modelId={option.model}
					supportsThinking={option.capabilities.includes(ModelCapability.REASONING)}
					supportsToolUse={option.capabilities.includes(ModelCapability.TOOL_USE)}
					tags={option.tags}
					title={option.model}
				/>
			</span>

			<ModelLoadControl
				canLoad={getBackendCapabilities(getBackend(option.backendId)).loadUnload}
				{isFailed}
				{isLoaded}
				{isLoading}
				{isSleeping}
				{option}
				revealOnHover={false}
			/>

			<span class="flex items-center justify-end">
				<ModelRowActions isFav={favorite} {isLoaded} {option} revealOnHover={false} />
			</span>
		</div>
	</div>
{/snippet}

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 py-4">
		<Input bind:value={filter} class="h-8 max-w-64 text-sm" placeholder="Filter models..." />

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-6 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>Model</span>

		<span class="text-center">State</span>

		<span class="text-right">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#each groups as group (group.key)}
			{#if group.items.length > 0}
				{#snippet groupIcon()}
					{#if group.kind === 'favorites'}
						<Heart class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'loaded'}
						<Power class="h-3.5 w-3.5 shrink-0" />
					{:else if group.kind === 'local'}
						<Logo class="shrink-0" style="--size: 0.875rem" />
					{/if}
				{/snippet}

				<ModelsSection
					backendId={group.kind === 'provider' ? (group.backendId ?? undefined) : undefined}
					count={group.items.length}
					icon={group.kind === 'provider' ? undefined : groupIcon}
					label={group.label}
					sticky
				>
					{#each group.items as option (option.id)}
						{@render row(option)}
					{/each}
				</ModelsSection>
			{/if}
		{/each}

		{#if isEmpty}
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
