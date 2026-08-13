<script lang="ts">
	import { ModelsSelectorDropdown, ModelsSelectorSheet } from '$lib/components/app';
	import { chatStore, conversationsStore, isMobile, modelsStore, serverStore } from '$lib/stores';

	interface Props {
		disabled?: boolean;
		forceForegroundText?: boolean;
		hasAudioModality?: boolean;
		hasVideoModality?: boolean;
		hasVisionModality?: boolean;
		hasModelSelected?: boolean;
		isSelectedModelInCache?: boolean;
		submitTooltip?: string;
		useGlobalSelection?: boolean;
	}

	let {
		disabled = false,
		forceForegroundText = false,
		hasAudioModality = $bindable(false),
		hasModelSelected = $bindable(false),
		hasVideoModality = $bindable(false),
		hasVisionModality = $bindable(false),
		isSelectedModelInCache = $bindable(true),
		submitTooltip = $bindable(''),
		useGlobalSelection = false
	}: Props = $props();

	let isRouter = $derived(serverStore.isRouterMode);
	let isOffline = $derived(!!serverStore.error);

	let conversationModel = $derived(
		chatStore.getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);

	let lastSyncedConversationModel: string | null = null;

	let selectorModel = $derived.by(() => {
		const storeModel = modelsStore.selectedModelName;

		if (storeModel && storeModel !== conversationModel) {
			return storeModel;
		}

		if (conversationModel) {
			return conversationModel;
		}

		return null;
	});

	$effect(() => {
		if (conversationModel && conversationModel !== lastSyncedConversationModel) {
			if (modelsStore.models.some((m) => m.model === conversationModel)) {
				modelsStore.selectedModelName = conversationModel;
				modelsStore.selectModelByName(conversationModel);
			} else {
				modelsStore.selectedModelName = null;
				modelsStore.clearSelection();
			}

			lastSyncedConversationModel = conversationModel;
		} else if (
			isRouter &&
			!modelsStore.selectedModelId &&
			modelsStore.loadedModelIds.length > 0 &&
			conversationsStore.activeMessages.length > 0 &&
			!conversationModel
		) {
			lastSyncedConversationModel = null;
			const first = modelsStore.models.find((m) => modelsStore.loadedModelIds.includes(m.model));

			if (first) modelsStore.selectModelById(first.id);
		}
	});

	let activeModelId = $derived.by(() => {
		const options = modelsStore.models;

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = modelsStore.selectedModelId;

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
		void modelPropsVersion;

		hasAudioModality = activeModelId ? modelsStore.modelSupportsAudio(activeModelId) : false;
	});

	$effect(() => {
		void modelPropsVersion;

		hasVideoModality = activeModelId ? modelsStore.modelSupportsVideo(activeModelId) : false;
	});

	$effect(() => {
		void modelPropsVersion;

		hasVisionModality = activeModelId ? modelsStore.modelSupportsVision(activeModelId) : false;
	});

	$effect(() => {
		hasModelSelected = !isRouter || !!conversationModel || !!modelsStore.selectedModelId;
	});

	$effect(() => {
		if (!isRouter) {
			isSelectedModelInCache = true;
		} else if (conversationModel) {
			isSelectedModelInCache = modelsStore.models.some(
				(option) => option.model === conversationModel
			);
		} else {
			const currentModelId = modelsStore.selectedModelId;

			if (!currentModelId) {
				isSelectedModelInCache = false;
			} else {
				isSelectedModelInCache = modelsStore.models.some((option) => option.id === currentModelId);
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

	export function open() {
		selectorModelRef?.open();
	}
</script>

{#if isMobile.current}
	<ModelsSelectorSheet
		disabled={disabled || isOffline}
		bind:this={selectorModelRef}
		currentModel={selectorModel}
		{forceForegroundText}
		{useGlobalSelection}
	/>
{:else}
	<ModelsSelectorDropdown
		disabled={disabled || isOffline}
		bind:this={selectorModelRef}
		currentModel={selectorModel}
		{forceForegroundText}
		{useGlobalSelection}
	/>
{/if}
