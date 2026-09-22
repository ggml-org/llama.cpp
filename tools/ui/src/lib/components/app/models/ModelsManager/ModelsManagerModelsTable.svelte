<script lang="ts">
	import type { ModelsTableGroup } from './utils';
	import { formatLastUsed } from './utils';
	import { EyeOff, Heart, MoreHorizontal, Power } from '@lucide/svelte';
	import {
		Logo,
		ModelAvatar,
		ModelContext,
		ModelId,
		ModelLoadControl,
		ModelRowActions,
		ModelsSection
	} from '$lib/components/app';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Input } from '$lib/components/ui/input';
	import { ModelCapability, ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';

	interface Props {
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onSelect: (option: ModelOption) => void;
		onToggleLoad: (option: ModelOption) => void;
		selectedId: string | null;
		summary: string;
	}

	let {
		filter = $bindable(''),
		groups,
		isFavorite,
		onSelect,
		onToggleLoad,
		selectedId,
		summary
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	let pendingDelete = $state('');
	let deleteOpen = $state(false);

	function requestDelete(option: ModelOption): void {
		pendingDelete = option.model;
		deleteOpen = true;
	}
	const rowGrid = 'grid grid-cols-[minmax(0,1fr)_7rem_5rem_3rem_4.5rem] items-center gap-3';

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
	{@const canLoad = getBackendCapabilities(getBackend(option.backendId)).loadUnload}
	{@const isHidden = modelsStore.isHidden(option.id)}

	<div>
		<div
			class={[
				rowGrid,
				'cursor-pointer rounded-md px-2 py-2.5 transition',
				isHidden && 'opacity-60',
				selectedId === option.id ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
			]}
			onclick={() => onSelect(option)}
			onkeydown={(event) => event.key === 'Enter' && onSelect(option)}
			role="button"
			tabindex="0"
		>
			<span class="flex min-w-0 items-center gap-3">
				<ModelAvatar {option} showBaseModelAvatar size="size-9" />

				{#if isHidden}
					<EyeOff class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
				{/if}

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

			<ModelContext {option} />

			<span class="text-sm text-muted-foreground">
				{formatLastUsed(modelsStore.recentModelUsage[option.id])}
			</span>

			<ModelLoadControl
				{canLoad}
				{isFailed}
				{isLoaded}
				{isLoading}
				{isSleeping}
				{option}
				revealOnHover={false}
				showRemoteMark
			/>

			<span class="flex items-center justify-end gap-1">
				<ModelRowActions isFav={favorite} {isLoaded} {option} revealOnHover={false} />

				<DropdownMenu.Root>
					<DropdownMenu.Trigger>
						{#snippet child({ props })}
							<Button
								{...props}
								aria-label="Model actions"
								class="h-7 w-7 text-muted-foreground"
								onclick={(event) => event.stopPropagation()}
								size="icon"
								variant="ghost"
							>
								<MoreHorizontal class="h-3.5 w-3.5" />
							</Button>
						{/snippet}
					</DropdownMenu.Trigger>

					<DropdownMenu.Content align="end">
						{#if canLoad}
							<DropdownMenu.Item onclick={() => onToggleLoad(option)}>
								{isLoaded ? 'Unload model' : 'Load model'}
							</DropdownMenu.Item>

							<DropdownMenu.Item onclick={() => requestDelete(option)}>
								Delete from disk
							</DropdownMenu.Item>
						{/if}

						<DropdownMenu.Item onclick={() => modelsStore.toggleHidden(option.id)}>
							{isHidden ? 'Show in selector' : 'Hide from selector'}
						</DropdownMenu.Item>
					</DropdownMenu.Content>
				</DropdownMenu.Root>
			</span>
		</div>
	</div>
{/snippet}

<DialogConfirmDownload
	action={ModelDownloadConfirmAction.DELETE}
	onClose={() => (deleteOpen = false)}
	open={deleteOpen}
	repoWithTag={pendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 py-4">
		<Input bind:value={filter} class="h-8 max-w-64 text-sm" placeholder="Filter models..." />

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-6 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>Model</span>

		<span>Context</span>

		<span>Last used</span>

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
