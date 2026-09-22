<script lang="ts">
	import type { ModelsTableGroup } from './utils';
	import { modelParamsLabel, modelSizeLabel } from './utils';
	import { Heart, Loader2, MoreHorizontal, Power } from '@lucide/svelte';
	import { Logo, ModelAvatar, ModelId, ModelsSection } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Input } from '$lib/components/ui/input';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		filter?: string;
		groups: ModelsTableGroup[];
		isFavorite: (option: ModelOption) => boolean;
		onCopyId: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		onToggleFavorite: (option: ModelOption) => void;
		onToggleLoad: (option: ModelOption) => void;
		onUseInNewChat: (option: ModelOption) => void;
		selectedId: string | null;
		summary: string;
	}

	let {
		filter = $bindable(''),
		groups,
		isFavorite,
		onCopyId,
		onSelect,
		onToggleFavorite,
		onToggleLoad,
		onUseInNewChat,
		selectedId,
		summary
	}: Props = $props();

	let isEmpty = $derived(groups.every((group) => group.items.length === 0));
	const rowGrid = 'grid grid-cols-[3.5rem_minmax(0,1fr)_5.5rem_3rem_4.5rem] items-center gap-3';

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
	{@const params = modelParamsLabel(option)}
	{@const size = modelSizeLabel(option)}
	{@const favorite = isFavorite(option)}

	<div class="px-2">
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
			<span
				class="inline-flex h-6 w-6 items-center justify-center rounded-md bg-muted text-[10px] font-medium text-muted-foreground"
			>
				{params ?? '—'}
			</span>

			<span class="flex min-w-0 items-center gap-3">
				<ModelAvatar {option} showBaseModelAvatar size="size-9" />

				<ModelId
					aliases={option.aliases}
					class="min-w-0 flex-1"
					hideParameters
					hideQuantization
					hideTags
					modalities={option.modalities}
					modelId={option.model}
					supportsThinking={option.capabilities.includes(ModelCapability.REASONING)}
					supportsToolUse={option.capabilities.includes(ModelCapability.TOOL_USE)}
					tags={option.tags}
					title={option.model}
				/>
			</span>

			<span class="text-sm text-muted-foreground">{size ?? '—'}</span>

			<span class="flex justify-center">
				{#if isLoading}
					<Loader2 class="h-3.5 w-3.5 animate-spin text-amber-500" />
				{:else if isFailed}
					<span class="block h-2.5 w-2.5 rounded-full bg-destructive"></span>
				{:else}
					<span
						class="block h-2.5 w-2.5 rounded-full {isLoaded
							? 'bg-emerald-500'
							: 'border border-muted-foreground/50'}"
					></span>
				{/if}
			</span>

			<span class="flex items-center justify-end gap-1">
				<Button
					aria-label={favorite ? 'Remove from favorites' : 'Add to favorites'}
					class={['h-7 w-7', favorite ? 'text-foreground' : 'text-muted-foreground']}
					onclick={(event) => {
						event.stopPropagation();
						onToggleFavorite(option);
					}}
					size="icon"
					variant="ghost"
				>
					<Heart class={favorite ? 'h-3.5 w-3.5 fill-current' : 'h-3.5 w-3.5'} />
				</Button>

				<DropdownMenu.Root>
					<DropdownMenu.Trigger>
						{#snippet child({ props })}
							<Button
								{...props}
								aria-label="More actions"
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
						<DropdownMenu.Item onclick={() => onUseInNewChat(option)}>
							Use in New Chat
						</DropdownMenu.Item>

						<DropdownMenu.Item onclick={() => onToggleLoad(option)}>
							{isLoaded ? 'Eject Model' : 'Load Model'}
						</DropdownMenu.Item>

						<DropdownMenu.Item onclick={() => onCopyId(option)}>Copy model id</DropdownMenu.Item>
					</DropdownMenu.Content>
				</DropdownMenu.Root>
			</span>
		</div>
	</div>
{/snippet}

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 px-4 py-3">
		<Input bind:value={filter} class="h-8 max-w-64 text-sm" placeholder="Filter models..." />

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-6 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>Params</span>

		<span>Model</span>

		<span>Size</span>

		<span class="text-center">State</span>

		<span class="text-right">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto py-2">
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
