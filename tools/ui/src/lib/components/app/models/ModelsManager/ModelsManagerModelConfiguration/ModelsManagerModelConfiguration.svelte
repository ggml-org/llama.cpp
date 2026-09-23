<script lang="ts">
	import { LOAD_DEFAULTS, type ModelOverride } from '../utils';
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInference from './ModelsManagerModelConfigurationInference.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import ModelsManagerModelConfigurationLoad from './ModelsManagerModelConfigurationLoad.svelte';
	import * as Tabs from '$lib/components/ui/tabs';
	import { ServerModelStatus } from '$lib/enums';
	import { modelsStore } from '$lib/stores';
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

	let serverProps = $derived(modelsStore.props.getModelProps(option.model));
	let status = $derived.by(() => {
		const model = modelsStore.routerModels.find((m) => m.id === option.model);

		return (model?.status?.value as ServerModelStatus) ?? null;
	});
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
			!isOperationInProgress
	);
	let loadProgress = $derived(
		isOperationInProgress ? modelsStore.status.getLoadProgress(option.model) : null
	);
	let contextMax = $derived(
		option.contextLength ??
			serverProps?.default_generation_settings?.n_ctx ??
			LOAD_DEFAULTS.contextLength
	);

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
		{serverProps}
	/>

	<Tabs.Root class="mt-3 min-h-0 flex-1 gap-0" onValueChange={(value) => (tab = value)} value={tab}>
		<Tabs.List class="px-4">
			<Tabs.Trigger value="information">Information</Tabs.Trigger>

			<Tabs.Trigger value="load">Load</Tabs.Trigger>

			<Tabs.Trigger value="inference">Inference</Tabs.Trigger>
		</Tabs.List>

		<div class="min-h-0 flex-1 overflow-y-auto px-4 py-4">
			<Tabs.Content value="information">
				<ModelsManagerModelConfigurationInformation {option} {serverProps} />
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
