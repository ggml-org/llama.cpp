<script lang="ts">
	import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
	import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { Button } from '$lib/components/ui/button';
	import { ModelDownloadConfirmAction } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';
	import { repoOf } from '$lib/utils';

	interface Props {
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
	}

	let { onClose, onToggleLoad, onUseInNewChat, option }: Props = $props();

	// the same removal the table row offers, confirmed by the same dialog
	let deleteOpen = $state(false);

	// the props cache is a plain Map, so the read has to name its version to stay reactive
	let serverProps = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelProps(option.model);
	});
	let status = $derived(modelsStore.getModelStatus(option.model));
	let isLoaded = $derived(modelsStore.isModelRunning(option.model));

	// The server reports the full metadata only once a model is loaded, which loads it.
	// With the Hub enabled, it fills those gaps instead.
	let hubDetails = $state<HfModelDetailInfo | null>(null);

	$effect(() => {
		const repo = repoOf(option.model);

		let cancelled = false;

		hubDetails = null;

		if (!HuggingFaceService.isEnabled() || !repo.includes('/')) {
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

	<div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4 max-md:pr-4">
		<ModelsManagerModelConfigurationInformation hub={hubDetails} {option} {serverProps} />

		<div class="mt-4 border-t border-border/30 pt-3 pb-2">
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
	</div>

	<DialogConfirmDownload
		action={ModelDownloadConfirmAction.DELETE}
		onClose={() => (deleteOpen = false)}
		open={deleteOpen}
		repoWithTag={option.model}
	/>
</div>
