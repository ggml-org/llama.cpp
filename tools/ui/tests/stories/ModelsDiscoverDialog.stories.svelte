<script lang="ts" module>
	import { mockListModels } from './fixtures/models-discover';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import DialogModelsDiscover from '$lib/components/app/dialogs/DialogModelsDiscover.svelte';
	import DialogModelDownload from '$lib/components/app/models/discover/DialogModelDownload.svelte';
	import DownloadProgressBar from '$lib/components/app/models/discover/DownloadProgressBar.svelte';
	import ModelsDiscover from '$lib/components/app/models/discover/ModelsDiscover.svelte';
	import { ModelDraftSidecar } from '$lib/enums';
	import { modelsHubStore, modelsStore } from '$lib/stores';
	import { SvelteSet } from 'svelte/reactivity';

	const { Story } = defineMeta({
		tags: ['autodocs'],
		title: 'Models/Discover/Dialog'
	});

	// Wire the hub store singleton with fixtures so the container renders data.
	modelsHubStore.models = mockListModels;
	modelsHubStore.loading = false;
	modelsHubStore.error = null;

	// Mark the Q4_K_M tag as failed for the previous-failure story. The status
	// manager keeps this in a private SvelteSet keyed by `<repo>:<tag>`.
	const FAILED_TAG = 'ggml-org/gemma-4-12b-it-GGUF:Q4_K_M';

	(
		modelsStore.status as unknown as { failedDownloads: Set<string> }
	).failedDownloads = new SvelteSet([FAILED_TAG]);


</script>

<Story name="Progress bar">
	<div class="space-y-4 p-4">
		<div class="space-y-1">
			<p class="text-xs text-muted-foreground">0%</p>

			<DownloadProgressBar downloadedBytes={0} totalBytes={7300000000} />
		</div>

		<div class="space-y-1">
			<p class="text-xs text-muted-foreground">49%</p>

			<DownloadProgressBar downloadedBytes={3600000000} totalBytes={7300000000} />
		</div>

		<div class="space-y-1">
			<p class="text-xs text-muted-foreground">100%</p>

			<DownloadProgressBar downloadedBytes={7300000000} totalBytes={7300000000} />
		</div>

		<div class="relative h-8 overflow-hidden rounded border">
			<p class="p-1 text-xs text-muted-foreground">overlay</p>

			<DownloadProgressBar downloadedBytes={3600000000} overlay totalBytes={7300000000} />
		</div>
	</div>
</Story>

<Story name="Download dialog (confirm)">
	<div class="p-4">
		<DialogModelDownload
			filePath="Q4_K_M/gemma-4-12b-it-Q4_K_M.gguf"
			formattedSize="7.3 GB"
			onClose={() => {}}
			open
			quant="Q4_K_M"
			repoId="ggml-org/gemma-4-12b-it-GGUF"
			sidecar={ModelDraftSidecar.MTP}
		/>
	</div>
</Story>

<Story name="Download dialog (previous failure)">
	<div class="p-4">
		<DialogModelDownload
			filePath="Q4_K_M/gemma-4-12b-it-Q4_K_M.gguf"
			formattedSize="7.3 GB"
			onClose={() => {}}
			open
			quant="Q4_K_M"
			repoId="ggml-org/gemma-4-12b-it-GGUF"
			sidecar={null}
		/>
	</div>
</Story>

<Story name="Discover (dialog shell)">
	<div class="h-160 w-full overflow-hidden border">
		<DialogModelsDiscover open />
	</div>
</Story>

<Story name="Discover (panes)">
	<div class="flex h-160 w-full border">
		<ModelsDiscover />
	</div>
</Story>
