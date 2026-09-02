<script lang="ts">
	import {
		type BitDepthRow,
		classify,
		type DownloadEntryState,
		labelFor,
		type QuantOption,
		type SelectableFile
	} from './download-options.utils';
	import ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand from './ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand.svelte';
	import ModelsDiscoverModelDetailsDownloadOptionsRow from './ModelsDiscoverModelDetailsDownloadOptionsRow.svelte';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** GGUF files grouped by bit depth. */
		bitDepthRows: BitDepthRow[];
		/** Download state lookup; defaults to the models store status feed. */
		getDownloadState?: (
			repoWithTag: string,
			filePath: string,
			isSidecar: boolean
		) => DownloadEntryState;
	}

	let { bitDepthRows, getDownloadState, modelId }: Props = $props();

	function stateFor(repoWithTag: string, filePath: string, isSidecar: boolean): DownloadEntryState {
		if (getDownloadState) return getDownloadState(repoWithTag, filePath, isSidecar);

		const isDownloading = modelsStore.status.isDownloadInProgress(repoWithTag);
		const isPaused = modelsStore.status.isDownloadPaused(repoWithTag);

		return {
			// solo downloads register in /v1/models under the tag (args stay empty),
			// while drafts pulled by a loaded model only show up as its --model-draft
			isDownloaded:
				!isDownloading &&
				(modelsStore.status.isModelDownloaded(repoWithTag) ||
					(isSidecar && modelsStore.status.isSidecarDownloaded(modelId, filePath))),
			isDownloading,
			isFailed: modelsStore.status.hasFailedDownload(repoWithTag),
			isPaused,
			// live progress while downloading, else the frozen snapshot of the pause
			progress:
				modelsStore.status.getDownloadProgress(repoWithTag) ??
				modelsStore.status.getPausedDownloadProgress(repoWithTag),
			repoWithTag
		};
	}

	/**
	 * Every selectable file with its kind and download state, in row order.
	 * Single source of truth for the chip rows and the command selects.
	 */
	let selectableFiles = $derived.by(() => {
		const files: (SelectableFile & { state: DownloadEntryState })[] = [];

		for (const row of bitDepthRows) {
			for (const file of row.files) {
				const meta = HuggingFaceService.extractQuantMeta(file.path);
				const tag = ModelsService.buildDownloadTag(
					modelId,
					meta?.quant ?? null,
					meta?.sidecar ?? null
				);

				files.push({
					...file,
					kind: classify(file.path),
					state: stateFor(tag, file.path, Boolean(meta?.sidecar))
				});
			}
		}

		return files;
	});

	/** Rows for the row component: per-bit-depth files with state attached. */
	let rows = $derived.by(() =>
		bitDepthRows.map((row) => {
			const paths = new Set(row.files.map((f) => f.path));

			return {
				bitDepth: row.bitDepth,
				files: selectableFiles.filter((f) => paths.has(f.path))
			};
		})
	);

	function optionFor(file: SelectableFile): QuantOption {
		return {
			label: labelFor(file.path),
			path: file.path
		};
	}

	/** Non-draft quants for the command's base select, in row order. */
	let mainOptions = $derived(selectableFiles.filter((f) => f.kind === 'main').map(optionFor));

	/**
	 * Draft options for the command's draft select, with their sidecar type
	 * (MTP, DFLASH...) since a repo can ship more than one draft flavour.
	 */
	let draftOptions = $derived(
		selectableFiles
			.filter((f) => f.kind === 'draft')
			.map((f) => ({
				...optionFor(f),
				badge: HuggingFaceService.extractQuantMeta(f.path)?.sidecar ?? null
			}))
	);
</script>

{#if bitDepthRows.length}
	<section class="rounded-3xl border border-border/30 bg-muted/60 shadow-xs dark:border-border/20">
		<!-- One chip per file, each an independent download action with its own
			 lifecycle state; nothing here selects anything. -->
		<div class="flex w-full flex-col divide-y divide-border/50 px-4 pb-1 dark:divide-border/35">
			{#each rows as row (row.bitDepth)}
				<ModelsDiscoverModelDetailsDownloadOptionsRow bitDepth={row.bitDepth} files={row.files} />
			{/each}
		</div>

		<!-- Terminal command preview, standalone: its picks are not bound to the chips. -->
		<div class="border-t border-border/50 px-4 pt-3.5 pb-4 dark:border-border/35">
			<ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand
				{draftOptions}
				{mainOptions}
				{modelId}
			/>
		</div>
	</section>
{/if}
