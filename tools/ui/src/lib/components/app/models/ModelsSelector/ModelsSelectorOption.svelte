<script lang="ts">
	import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
	import { modelDraftsFor } from '../ModelsManager/utils';
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
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';

	interface Props {
		option: ModelOption;
		isSelected: boolean;
		isHighlighted: boolean;
		isFav: boolean;
		hideOrgName?: boolean;
		onSelect: (modelId: string) => void;
		onMouseEnter: () => void;
		onKeyDown: (e: KeyboardEvent) => void;
		/** Show the base model's org as the main avatar and the repo (quant) org as the corner badge; resolves the base org lazily via HF. */
		showBaseModelAvatar?: boolean;
		/** Show the repo's own org instead of the base model's. */
		showRepoOrgAvatar?: boolean;
	}

	let {
		hideOrgName = false,
		isFav,
		isHighlighted,
		isSelected,
		onKeyDown,
		onMouseEnter,
		onSelect,
		option,
		showBaseModelAvatar = false,
		showRepoOrgAvatar = false
	}: Props = $props();

	// row actions follow the backend that serves the row, not the selected one
	let rowBackend = $derived(getBackend(option.backendId));
	let canLoad = $derived(rowBackend ? getBackendCapabilities(rowBackend).loadUnload : false);
	let currentRouterModels = $derived(modelsStore.routerModels);
	let serverStatus = $derived.by(() => {
		const model = currentRouterModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isFailed = $derived(serverStatus === ServerModelStatus.FAILED);
	let isSleeping = $derived(serverStatus === ServerModelStatus.SLEEPING);
	let isLoaded = $derived(
		(serverStatus === ServerModelStatus.LOADED || isSleeping) && !isOperationInProgress
	);
	let isLoading = $derived(serverStatus === ServerModelStatus.LOADING || isOperationInProgress);

	let loadProgress = $derived(isLoading ? modelsStore.status.getLoadProgress(option.model) : null);
	let loadPercent = $derived(Math.round(modelLoadFraction(loadProgress) * 100));
	let loadTitle = $derived(modelLoadProgressText(loadProgress));
	let modalities = $derived(option.modalities);
	let showCapabilities = $derived(
		settingsStore.config[SETTINGS_KEYS.SHOW_MODEL_CAPABILITIES_IN_SELECTOR] ?? false
	);
	// Avatar: with showBaseModelAvatar the original base model's org is the main
	// image and the repo (quantizer) org the corner badge, as in the discover
	// list. Loaded models usually carry the `base_model` tag on the option; GGUF
	// repos only known to HF are resolved lazily via the cached getBaseModel
	// lookup.
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
	<ModelAvatar {option} {showBaseModelAvatar} {showRepoOrgAvatar} />

	<ModelId
		aliases={option.aliases}
		class="min-w-0 flex-1"
		drafts={modelDraftsFor(option)}
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

		<ModelLoadControl
			{canLoad}
			{isFailed}
			{isLoaded}
			{isLoading}
			{isSleeping}
			{option}
			showBackendMark
		/>
	</div>

	{#if isLoading}
		<ModelLoadHighlight percent={loadPercent} />
	{/if}
</div>
