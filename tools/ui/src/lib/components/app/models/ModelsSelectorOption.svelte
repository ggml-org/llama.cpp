<script lang="ts">
	import ModelsDiscoverAvatar from './discover/ModelsDiscoverAvatar.svelte';
	import ModelLoadHighlight from './ModelLoadHighlight.svelte';
	import {
		CircleAlert,
		Heart,
		HeartOff,
		Info,
		Loader2,
		Power,
		PowerOff,
		RotateCw
	} from '@lucide/svelte';
	import { ActionIcon, ModelId } from '$lib/components/app';
	import { HF_BASE_MODEL_TAG_REGEX, ICON_CLASS_DEFAULT, PATH_SEPARATOR } from '$lib/constants';
	import { ModelCapability, ServerModelStatus } from '$lib/enums';
	import { useModelParamsFallback } from '$lib/hooks/use-model-params-fallback.svelte';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
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
		onInfoClick?: (modelName: string) => void;
		/** Show the base model's org as the main avatar and the repo (quant) org as the corner badge; resolves the base org lazily via HF. */
		showBaseModelAvatar?: boolean;
	}

	let {
		hideOrgName = false,
		isFav,
		isHighlighted,
		isSelected,
		onInfoClick,
		onKeyDown,
		onMouseEnter,
		onSelect,
		option,
		showBaseModelAvatar = false
	}: Props = $props();

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
	// Avatar: with showBaseModelAvatar the original base model's org is the main
	// image and the repo (quantizer) org the corner badge, as in the models hub
	// list. Loaded models usually carry the `base_model` tag on the option; GGUF
	// repos only known to HF are resolved lazily via the cached getBaseModel
	// lookup.
	let parsedId = $derived(ModelsService.parseModelId(option.model));
	let orgName = $derived(parsedId.orgName);
	let tagBaseModel = $derived(
		(option.tags ?? [])
			.find((t) => HF_BASE_MODEL_TAG_REGEX.test(t))
			?.match(HF_BASE_MODEL_TAG_REGEX)?.[1] ?? null
	);
	let fetchedBaseModelOrg = $state<string | null>(null);
	let baseModelOrg = $derived(tagBaseModel?.split(PATH_SEPARATOR)[0] ?? fetchedBaseModelOrg);

	$effect(() => {
		fetchedBaseModelOrg = null;

		if (!showBaseModelAvatar || !orgName || tagBaseModel) return;

		let cancelled = false;

		void HuggingFaceService.getBaseModel(option.model).then((base) => {
			if (!cancelled && base?.org) fetchedBaseModelOrg = base.org;
		});

		return () => {
			cancelled = true;
		};
	});
	let capabilities = $derived.by(() => ({
		reasoning: modelsStore.props.checkModelSupportsThinking(option.model),
		tools: option.capabilities.includes(ModelCapability.TOOL_USE)
	}));

	const { paramsFallback } = useModelParamsFallback({
		metaParams: () => option.meta?.n_params,
		modelId: () => option.model
	});
</script>

<div
	aria-selected={isSelected || isHighlighted}
	class={[
		'group relative flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
		'cursor-pointer',
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
	{#if orgName}
		<ModelsDiscoverAvatar
			class="mt-0"
			org={baseModelOrg ?? orgName}
			quantOrg={showBaseModelAvatar ? orgName : undefined}
			quantPositionClass="-bottom-1 -right-1"
			quantSize="h-3 w-3"
			size="size-5"
		/>
	{/if}

	<ModelId
		aliases={option.aliases}
		class="flex-1"
		{hideOrgName}
		{modalities}
		modelId={option.model}
		params={paramsFallback}
		showRawTooltip
		supportsThinking={capabilities.reasoning}
		supportsToolUse={capabilities.tools}
		tags={option.tags}
	/>

	<div class="flex shrink-0 items-center gap-1">
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="pointer-events-none flex items-center justify-center gap-0.75 pl-2 opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
			onclick={(e) => e.stopPropagation()}
		>
			{#if isFav}
				<ActionIcon
					class="h-3 w-3 hover:text-foreground"
					icon={HeartOff}
					iconSize="h-2.5 w-2.5"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Remove from favorites"
				/>
			{:else}
				<ActionIcon
					class="h-3 w-3 hover:text-foreground"
					icon={Heart}
					iconSize="h-2.5 w-2.5"
					onclick={() => modelsStore.toggleFavorite(option.model)}
					tooltip="Add to favorites"
				/>
			{/if}

			<!-- info button: only shown when model is loaded and callback is provided -->
			{#if isLoaded && onInfoClick}
				<ActionIcon
					class="h-3 w-3 hover:text-foreground"
					icon={Info}
					iconSize="h-2.5 w-2.5"
					onclick={() => onInfoClick(option.model)}
					tooltip="Model information"
				/>
			{/if}
		</div>

		{#if isLoading}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-5">
				<Loader2 class="{ICON_CLASS_DEFAULT} animate-spin text-muted-foreground" />
			</div>
		{:else if isFailed}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<CircleAlert
					class="h-3.5 w-3.5 text-red-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
				/>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						class="h-3 w-3 text-red-500 hover:text-foreground"
						icon={RotateCw}
						iconSize="h-2.5 w-2.5"
						onclick={() => modelsStore.status.load(option.model)}
						stopPropagationOnClick
						tooltip="Retry loading model"
					/>
				</div>
			</div>
		{:else if isSleeping}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-orange-400 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-amber-500 [@media(pointer:coarse)]:hover:text-amber-600"
						icon={PowerOff}
						iconSize="h-2.5 w-2.5"
						onclick={(e) => {
							e?.stopPropagation();
							modelsStore.status.unload(option.model);
						}}
						tooltip="Unload model"
					/>
				</div>
			</div>
		{:else if isLoaded}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-green-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						class="h-3 w-3 text-red-500 hover:text-red-600 [@media(pointer:coarse)]:text-green-500 [@media(pointer:coarse)]:hover:text-green-600"
						icon={PowerOff}
						iconSize="h-2.5 w-2.5"
						onclick={() => modelsStore.status.unload(option.model)}
						stopPropagationOnClick
						tooltip="Unload model"
					/>
				</div>
			</div>
		{:else}
			<div class="flex w-4 items-center justify-center [@media(pointer:coarse)]:w-auto">
				<span
					class="h-2 w-2 rounded-full bg-muted-foreground/50 group-hover:hidden [@media(pointer:coarse)]:hidden"
				></span>

				<div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
					<ActionIcon
						class="h-3 w-3 [@media(pointer:coarse)]:text-muted-foreground"
						icon={Power}
						iconSize="h-2.5 w-2.5"
						onclick={() => modelsStore.status.load(option.model)}
						stopPropagationOnClick
						tooltip="Load model"
					/>
				</div>
			</div>
		{/if}
	</div>

	{#if isLoading}
		<ModelLoadHighlight percent={loadPercent} />
	{/if}
</div>
