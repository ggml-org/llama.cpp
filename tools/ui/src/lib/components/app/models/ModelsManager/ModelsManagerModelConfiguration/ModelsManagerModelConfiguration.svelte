<script lang="ts">
	import type { ModelOverride } from '../utils';
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInference from './ModelsManagerModelConfigurationInference.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import ModelsManagerModelConfigurationLoad from './ModelsManagerModelConfigurationLoad.svelte';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import * as Tabs from '$lib/components/ui/tabs';
	import { LOAD_DEFAULTS, MODEL_ID } from '$lib/constants';
	import { ModelDownloadConfirmAction } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';
	import { repoOf } from '$lib/utils';

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

	// the same removal the table row offers, confirmed by the same dialog
	let deleteOpen = $state(false);

	// the props cache is a plain Map, so the read has to name its version to stay reactive
	let serverProps = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelProps(option.model);
	});
	let status = $derived(modelsStore.getModelStatus(option.model));
	let isOperationInProgress = $derived(modelsStore.status.isOperationInProgress(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));
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
		const repo = repoOf(option.model);

		let cancelled = false;

		hubDetails = null;

		if (!HuggingFaceService.isEnabled() || !repo.includes(MODEL_ID.ORG_SEPARATOR)) {
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
		<div class="pl-4 max-md:pr-4">
			<Tabs.List class="w-full">
				<Tabs.Trigger value="information">Information</Tabs.Trigger>

				<Tabs.Trigger value="load">Load</Tabs.Trigger>

				<Tabs.Trigger value="inference">Inference</Tabs.Trigger>
			</Tabs.List>
		</div>

		<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4 max-md:pr-4">
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

	<div class="border-t border-border/30 px-4 pt-3 pb-4">
		<Button
			class="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive dark:hover:bg-destructive/20"
			onclick={() => (deleteOpen = true)}
			size="sm"
			variant="ghost"
		>
			<Trash2 class="h-4 w-4" />
			Delete this model from disk
		</Button>
	</div>

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.DELETE}
		onClose={() => (deleteOpen = false)}
		open={deleteOpen}
		repoWithTag={option.model}
	/>
</div>
