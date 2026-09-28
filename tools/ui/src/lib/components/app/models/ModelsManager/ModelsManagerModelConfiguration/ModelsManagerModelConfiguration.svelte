<script lang="ts">
	import { LOAD_DEFAULTS, type ModelOverride } from '../utils';
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInference from './ModelsManagerModelConfigurationInference.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import ModelsManagerModelConfigurationLoad from './ModelsManagerModelConfigurationLoad.svelte';
	import * as Tabs from '$lib/components/ui/tabs';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { ServerModelStatus } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore, settingsStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isCustomized: boolean;
		onClose: () => void;
		onSave: (override: ModelOverride) => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		override?: ModelOverride;
	}

	let { isCustomized, onClose, onSave, onToggleLoad, onUseInNewChat, option, override }: Props =
		$props();

	let tab = $state('information');
	// edits in this pane, cleared on reset; the stored override is the baseline
	let edits = $state<ModelOverride | null>(null);
	let draft = $derived(edits ?? override ?? {});

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
	let loadProgress = $derived(
		isOperationInProgress ? modelsStore.status.getLoadProgress(option.model) : null
	);

	// The server only reports the full metadata once a model is loaded, which loads
	// it. When discovery is on, the Hub fills those gaps instead.
	let hubDetails = $state<HfModelDetailInfo | null>(null);
	// the window the model can take, from the listing or the Hub; a loaded model's runtime
	// context only bounds it when nothing else says otherwise
	let contextMax = $derived(
		option.contextLength ??
			hubDetails?.gguf?.context_length ??
			serverProps?.default_generation_settings?.n_ctx ??
			LOAD_DEFAULTS.contextLength
	);

	$effect(() => {
		const repo = option.model.split(':')[0] ?? option.model;

		let cancelled = false;

		hubDetails = null;

		if (!settingsStore.config[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS] || !repo.includes('/')) {
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

	function resetDraft(): void {
		edits = null;
	}
</script>

<div class="flex h-full min-h-0 flex-col">
	<ModelsManagerModelConfigurationHeader
		{isCustomized}
		{isLoaded}
		{onClose}
		{onToggleLoad}
		{onUseInNewChat}
		{option}
		{status}
	/>

	<Tabs.Root class="mt-3 min-h-0 flex-1 gap-0" onValueChange={(value) => (tab = value)} value={tab}>
		<div class="pl-4">
			<Tabs.List class="w-full">
				<Tabs.Trigger value="information">Information</Tabs.Trigger>

				<Tabs.Trigger value="load">Load</Tabs.Trigger>

				<Tabs.Trigger value="inference">Inference</Tabs.Trigger>
			</Tabs.List>
		</div>

		<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4">
			<Tabs.Content value="information">
				<ModelsManagerModelConfigurationInformation
					draftSetting={override?.load?.speculativeDecoding ?? null}
					hub={hubDetails}
					{option}
					{serverProps}
				/>
			</Tabs.Content>

			<Tabs.Content value="load">
				<ModelsManagerModelConfigurationLoad
					{contextMax}
					{draft}
					{loadProgress}
					onChange={(next) => (edits = next)}
					onReset={resetDraft}
					onSave={() => onSave(draft)}
				/>
			</Tabs.Content>

			<Tabs.Content value="inference">
				<ModelsManagerModelConfigurationInference
					{draft}
					onChange={(next) => (edits = next)}
					onReset={resetDraft}
					onSave={() => onSave(draft)}
				/>
			</Tabs.Content>
		</div>
	</Tabs.Root>
</div>
