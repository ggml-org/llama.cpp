<script lang="ts">
	import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
	import {
		ModelAvatar,
		ModelCapabilities,
		ModelId,
		ModelLoadControl,
		ModelRowActions
	} from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { ModelOption } from '$lib/types/models';
	import { modelLoadFraction, modelLoadProgressText } from '$lib/utils';

	interface Props {
		option: ModelOption;
		isSelected: boolean;
		isHighlighted: boolean;
		isFav: boolean;
		hideOrgName?: boolean;
		onSelect: (modelId: string) => void;
		onMouseEnter: () => void;
		onKeyDown: (e: KeyboardEvent) => void;
	}

	let {
		hideOrgName = false,
		isFav,
		isHighlighted,
		isSelected,
		onKeyDown,
		onMouseEnter,
		onSelect,
		option
	}: Props = $props();

	let serverStatus = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isFailed = $derived(serverStatus === ServerModelStatus.FAILED);
	let isSleeping = $derived(serverStatus === ServerModelStatus.SLEEPING);
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));
	let isLoading = $derived(serverStatus === ServerModelStatus.LOADING || isOperationInProgress);

	let loadProgress = $derived(isLoading ? modelsStore.status.getLoadProgress(option.model) : null);
	let loadPercent = $derived(Math.round(modelLoadFraction(loadProgress) * 100));
	let loadTitle = $derived(modelLoadProgressText(loadProgress));
	let modalities = $derived(option.modalities);
	let showCapabilities = $derived(
		settingsStore.config[SETTINGS_KEYS.SHOW_MODEL_CAPABILITIES_IN_SELECTOR] ?? false
	);
</script>

<div
	aria-selected={isSelected || isHighlighted}
	class={[
		'group relative flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
		'cursor-pointer',
		// skip layout and paint for rows scrolled out of the long lists
		'[content-visibility:auto] [contain-intrinsic-size:auto_2.25rem]',
		isSelected && !isHighlighted && 'bg-accent/50',
		isHighlighted && 'bg-accent',
		(isSelected || isHighlighted) && 'text-accent-foreground',
		'hover:bg-accent',
		'focus:bg-accent',
		isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
	]}
	onclick={() => onSelect(option.id)}
	onkeydown={onKeyDown}
	onmouseenter={onMouseEnter}
	role="option"
	tabindex="0"
	title={loadTitle}
>
	<ModelAvatar {option} />

	<ModelId
		aliases={option.aliases}
		class="min-w-0 flex-1"
		draftSidecars={option.draftSidecars}
		hideCapabilities
		hideModalities
		{hideOrgName}
		{modalities}
		modelId={option.model}
		tags={option.tags}
		title={option.model}
	/>

	{#if showCapabilities}
		<ModelCapabilities {option} />
	{/if}

	<div class="flex shrink-0 items-center gap-1">
		<ModelRowActions {isFav} {option} />

		<ModelLoadControl {isFailed} {isLoaded} {isLoading} {isSleeping} {option} />
	</div>

	{#if isLoading}
		<ModelLoadHighlight percent={loadPercent} />
	{/if}
</div>
