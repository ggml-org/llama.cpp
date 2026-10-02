<script lang="ts">
	import ModelContext from '../ModelContext.svelte';
	import ModelDraftSidecars from '../ModelDraftSidecars.svelte';
	import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
	import { modelRowActions } from './row-actions';
	import { configuredContext } from './utils';
	import { MoreHorizontal } from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { MODEL_ROW_GRID_CLASS } from '$lib/constants';
	import { modelsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isFavorite: (option: ModelOption) => boolean;
		option: ModelOption;
		onDelete: (option: ModelOption) => void;
		onSelect: (option: ModelOption) => void;
		selected: boolean;
		/** Left padding in px, from the nesting depth. */
		indent?: number;
	}

	let { indent = 0, isFavorite, onDelete, onSelect, option, selected }: Props = $props();

	let favorite = $derived(isFavorite(option));
	let isHidden = $derived(modelsStore.isHidden(option.id));
	let quant = $derived(option.parsedId?.quantization ?? option.model);
</script>

<!-- the row itself takes no focus: the quant cell is the control that selects it -->
<div
	class={[
		MODEL_ROW_GRID_CLASS,
		'group rounded-md px-2 py-2 transition',
		isHidden && 'opacity-60',
		selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
	]}
>
	<button
		aria-current={selected ? 'true' : undefined}
		class="flex min-w-0 cursor-pointer items-center gap-3 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
		onclick={() => onSelect(option)}
		style="padding-left: {indent}px"
		type="button"
	>
		<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">{quant}</Badge>

		<ModelDraftSidecars draftSidecars={option.draftSidecars} />

		<span class="truncate text-sm text-muted-foreground">{option.model}</span>
	</button>

	<ModelContext class="justify-self-end" configured={configuredContext(option)} {option} />

	<ModelsManagerStatusCell {option} />

	<div class="flex items-center justify-center justify-self-center">
		<DropdownMenuActions
			actions={modelRowActions(option, favorite, isHidden, onDelete)}
			align="end"
			triggerIcon={MoreHorizontal}
			triggerTooltip="Model actions"
		/>
	</div>
</div>
