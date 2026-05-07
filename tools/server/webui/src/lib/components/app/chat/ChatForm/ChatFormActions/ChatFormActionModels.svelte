<script lang="ts">
	import { chatStore } from '$lib/stores/chat.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode, serverError } from '$lib/stores/server.svelte';
	import { ModelsSelectorDropdown, ModelsSelectorSheet } from '$lib/components/app';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { activeMessages } from '$lib/stores/conversations.svelte';

	interface Props {
		currentModel?: string;
		disabled?: boolean;
		forceForegroundText?: boolean;
		hasAudioModality?: boolean;
		hasVisionModality?: boolean;
		hasModelSelected?: boolean;
		isSelectedModelInCache?: boolean;
		submitTooltip?: string;
		useGlobalSelection?: boolean;
	}

	let {
		currentModel,
		disabled = false,
		forceForegroundText = false,
		hasAudioModality = $bindable(false),
		hasVisionModality = $bindable(false),
		hasModelSelected = $bindable(false),
		isSelectedModelInCache = $bindable(true),
		submitTooltip = $bindable(''),
		useGlobalSelection = false
	}: Props = $props();

	let isRouter = $derived(isRouterMode());
	let isOffline = $derived(!!serverError());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let lastSyncedConversationModel: string | null = null;

	$effect(() => {
		if (conversationModel && conversationModel !== lastSyncedConversationModel) {
			lastSyncedConversationModel = conversationModel;

			modelsStore.selectModelByName(conversationModel);
		} else if (isRouter && !modelsStore.selectedModelId && modelsStore.loadedModelIds.length > 0) {
			lastSyncedConversationModel = null;
			// auto-select the first loaded model only when nothing is selected yet
			const first = modelOptions().find((m) => modelsStore.loadedModelIds.includes(m.model));

			if (first) modelsStore.selectModelById(first.id);
		}
	});

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();

		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);

			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);

			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);

			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	$effect(() => {
		hasAudioModality = activeModelId ? modelsStore.modelSupportsAudio(activeModelId) : false;
	});

	$effect(() => {
		void modelPropsVersion;

		hasVisionModality = activeModelId ? modelsStore.modelSupportsVision(activeModelId) : false;
	});

	$effect(() => {
		hasModelSelected = !isRouter || !!conversationModel || !!selectedModelId();
	});

	$effect(() => {
		if (!isRouter) {
			isSelectedModelInCache = true;
		} else if (conversationModel) {
			isSelectedModelInCache = modelOptions().some((option) => option.model === conversationModel);
		} else {
			const currentModelId = selectedModelId();

			if (!currentModelId) {
				isSelectedModelInCache = false;
			} else {
				isSelectedModelInCache = modelOptions().some((option) => option.id === currentModelId);
			}
		}
	});

	$effect(() => {
		if (!hasModelSelected) {
			submitTooltip = 'Please select a model first';
		} else if (!isSelectedModelInCache) {
			submitTooltip = 'Selected model is not available, please select another';
		} else {
			submitTooltip = '';
		}
	});

	let selectorModelRef: ModelsSelectorDropdown | ModelsSelectorSheet | undefined =
		$state(undefined);

	let isMobile = new IsMobile();

	export function open() {
		selectorModelRef?.open();
	}
</script>

{#if isMobile.current}
	<ModelsSelectorSheet
		disabled={disabled || isOffline}
		bind:this={selectorModelRef}
		{currentModel}
		{forceForegroundText}
		{useGlobalSelection}
	/>
{:else}
	<ModelsSelectorDropdown
		disabled={disabled || isOffline}
		bind:this={selectorModelRef}
		{currentModel}
		{forceForegroundText}
		{useGlobalSelection}
	/>
{/if}
