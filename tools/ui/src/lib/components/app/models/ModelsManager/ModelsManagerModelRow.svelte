<script lang="ts">
	import ModelAvatar from '../ModelAvatar.svelte';
	import ModelCapabilities from '../ModelCapabilities.svelte';
	import ModelContext from '../ModelContext.svelte';
	import ModelId from '../ModelId.svelte';
	import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
	import { modelRowActions, type ModelRowDraftTarget } from './row-actions';
	import { canLoadOption, configuredContext, modelDraftsFor, type ModelOverride } from './utils';
	import { MoreHorizontal } from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import { MODEL_ROW_GRID_CLASS } from '$lib/constants';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		/** Model the pane has open, when this row can be set as its draft. */
		draftTarget?: ModelRowDraftTarget | null;
		isFavorite: (option: ModelOption) => boolean;
		/** Stored per-model overrides, for the drafts and context the row reports. */
		overrides?: Record<string, ModelOverride>;
		/** Model this row stands for. */
		option: ModelOption;
		onDelete: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		onUseAsDraft?: (draft: ModelOption, targetId: string) => void;
		selected: boolean;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let {
		draftTarget = null,
		indent = 0,
		isFavorite,
		onDelete,
		onSelect,
		onUseAsDraft,
		option,
		overrides,
		selected
	}: Props = $props();

	let favorite = $derived(isFavorite(option));
	let isHidden = $derived(modelsStore.isHidden(option.id));
	let drafts = $derived(modelDraftsFor(option, overrides?.[option.id]?.load?.speculativeDecoding));

	function handleKeydown(event: KeyboardEvent): void {
		if (event.key === ' ') event.preventDefault();

		if (event.key === 'Enter' || event.key === ' ') onSelect(option);
	}
</script>

<div
	class={[
		MODEL_ROW_GRID_CLASS,
		'group cursor-pointer rounded-md px-2 py-3 transition',
		isHidden && 'opacity-60',
		selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
	]}
	onclick={() => onSelect(option)}
	onkeydown={handleKeydown}
	role="button"
	tabindex="0"
>
	<span class="flex min-w-0 items-center gap-3" style="padding-left: {indent}px">
		<ModelAvatar
			{option}
			showBaseModelAvatar={!settingsStore.config.groupModelsByFamily}
			showRepoOrgAvatar={settingsStore.config.groupModelsByFamily}
			size="size-9"
		/>

		<span class="flex min-w-0 items-center gap-1.25">
			<ModelId
				aliases={option.aliases}
				class="min-w-0 flex-1"
				draftSidecars={option.draftSidecars}
				{drafts}
				hideCapabilities
				hideModalities
				modalities={option.modalities}
				modelId={option.model}
				tags={option.tags}
				title={option.model}
			/>

			<ModelCapabilities {option} />
		</span>
	</span>

	<ModelContext
		class="justify-self-end"
		configured={configuredContext(option, overrides)}
		{option}
	/>

	<ModelsManagerStatusCell {option} />

	<div class="flex items-center justify-center justify-self-center">
		<DropdownMenuActions
			actions={modelRowActions(
				option,
				{ canLoad: canLoadOption(option), draftTarget, favorite, isHidden },
				{ onDelete, onUseAsDraft }
			)}
			align="end"
			triggerIcon={MoreHorizontal}
			triggerTooltip="Model actions"
		/>
	</div>
</div>
