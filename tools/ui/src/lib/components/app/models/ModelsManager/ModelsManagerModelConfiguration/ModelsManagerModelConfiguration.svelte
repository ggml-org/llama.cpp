<script lang="ts">
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import { MODEL_ID, SETTINGS_KEYS } from '$lib/constants';
	import { ServerModelStatus } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
	}

	let { onClose, onToggleLoad, onUseInNewChat, option }: Props = $props();

	// the props cache is a plain Map, so the read has to name its version to stay reactive
	let serverProps = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelProps(option.model);
	});
	let status = $derived.by(() => modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
			!isOperationInProgress
	);

	// The server reports the full metadata only once a model is loaded, which loads it.
	// With the Hub enabled, it fills those gaps instead.
	let hubDetails = $state<HfModelDetailInfo | null>(null);

	$effect(() => {
		const repo = option.model.split(MODEL_ID.QUANTIZATION_SEPARATOR)[0] ?? option.model;

		let cancelled = false;

		hubDetails = null;

		if (!settingsStore.config[SETTINGS_KEYS.USE_HUGGING_FACE_HUB] || !repo.includes('/')) {
			return;
		}

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) hubDetails = details;
			})
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});
</script>

<div class="flex h-full min-h-0 flex-col">
	<ModelsManagerModelConfigurationHeader
		{isLoaded}
		{onClose}
		{onToggleLoad}
		{onUseInNewChat}
		{option}
		{status}
	/>

	<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4">
		<ModelsManagerModelConfigurationInformation hub={hubDetails} {option} {serverProps} />
	</div>
</div>
