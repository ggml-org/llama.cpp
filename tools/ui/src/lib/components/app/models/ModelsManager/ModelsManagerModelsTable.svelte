<script lang="ts">
	import { modelParamsLabel, modelSizeLabel, servedByLabel } from './utils';
	import { ArrowUpDown, ChevronDown, Heart, Loader2, MoreHorizontal, Plus } from '@lucide/svelte';
	import { ModelCapabilityIcons } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isFavorite: (option: ModelOption) => boolean;
		models: ModelOption[];
		onCopyId: (option: ModelOption) => void;
		onGetModels: () => void;
		onSelect: (option: ModelOption) => void;
		onToggleFavorite: (option: ModelOption) => void;
		onToggleLoad: (option: ModelOption) => void;
		onUseInNewChat: (option: ModelOption) => void;
		selectedId: string | null;
		summary: string;
		title: string;
	}

	let {
		isFavorite,
		models,
		onCopyId,
		onGetModels,
		onSelect,
		onToggleFavorite,
		onToggleLoad,
		onUseInNewChat,
		selectedId,
		summary,
		title
	}: Props = $props();

	let descending = $state(false);
	let sorted = $derived.by(() => {
		const list = [...models].sort((a, b) => a.name.localeCompare(b.name));

		return descending ? list.reverse() : list;
	});

	const rowGrid =
		'grid grid-cols-[3.5rem_minmax(0,1fr)_5.5rem_8.5rem_3rem_4.5rem] items-center gap-3';

	function stateOf(option: ModelOption): ServerModelStatus | null {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	}
</script>

<div class="flex h-full min-h-0 flex-col">
	<div class="flex shrink-0 items-center gap-2 px-4 py-3">
		<h3 class="text-sm font-medium">{title}</h3>

		<span class="ml-auto text-xs text-muted-foreground">{summary}</span>

		<Button class="h-7 gap-1 px-2 text-xs" onclick={onGetModels} variant="ghost">
			<Plus class="h-3.5 w-3.5" />

			Get models
		</Button>

		<Button
			aria-label="Sort by name"
			class="h-7 w-7"
			onclick={() => (descending = !descending)}
			size="icon"
			variant="ghost"
		>
			<ArrowUpDown class="h-3.5 w-3.5" />
		</Button>
	</div>

	<div
		class="{rowGrid} shrink-0 border-y border-border/40 px-4 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase"
	>
		<span>Params</span>

		<span>Model</span>

		<span>Size</span>

		<span>Served by</span>

		<span class="text-center">State</span>

		<span class="text-right">Actions</span>
	</div>

	<div class="min-h-0 flex-1 overflow-y-auto">
		{#each sorted as option (option.id)}
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

			<div
				class={[
					rowGrid,
					'cursor-pointer border-b border-border/30 px-4 py-2.5 transition',
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

				<span class="min-w-0">
					<span class="flex items-center gap-1.5">
						<span class="truncate text-sm font-medium">{option.name}</span>

						<ModelCapabilityIcons
							modalities={option.modalities}
							supportsThinking={option.capabilities.includes(ModelCapability.REASONING)}
							supportsToolUse={option.capabilities.includes(ModelCapability.TOOL_USE)}
						/>

						<ChevronDown class="h-3 w-3 shrink-0 text-muted-foreground" />
					</span>

					<span class="block truncate text-xs text-muted-foreground">
						{option.parsedId?.orgName ?? servedByLabel(option)}
					</span>
				</span>

				<span class="text-sm text-muted-foreground">{size ?? '—'}</span>

				<span>
					<Badge class="h-5 px-1.5 text-[10px]" variant="outline">{servedByLabel(option)}</Badge>
				</span>

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
		{/each}

		{#if sorted.length === 0}
			<p class="px-4 py-10 text-center text-sm text-muted-foreground">No models found.</p>
		{/if}
	</div>
</div>
