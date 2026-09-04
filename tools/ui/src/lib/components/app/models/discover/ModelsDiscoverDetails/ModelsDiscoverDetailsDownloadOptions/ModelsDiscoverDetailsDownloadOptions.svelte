<script lang="ts">
	import { classify, labelFor } from './download-options.utils';
	import ModelsDiscoverDetailsDownloadOptionsDownloadCommand from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
	import ModelsDiscoverDetailsDownloadOptionsRow from './ModelsDiscoverDetailsDownloadOptionsRow.svelte';
	import { DialogConfirmDownload } from '$lib/components/app/dialogs';
	import { ModelDownloadConfirmAction, ModelSelectableFileKind } from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type {
		ModelBitDepthRow,
		ModelDownloadEntryState,
		ModelQuantOption,
		ModelSelectableFile
	} from '$lib/types';

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** GGUF files grouped by bit depth. */
		bitDepthRows: ModelBitDepthRow[];
		/** Download state lookup; defaults to the models store status feed. */
		getDownloadState?: (
			repoWithTag: string,
			filePath: string,
			isSidecar: boolean
		) => ModelDownloadEntryState;
	}

	let { bitDepthRows, getDownloadState, modelId }: Props = $props();

	// Destructive chip actions (delete a downloaded model, cancel a download) are
	// confirmed here rather than inside each chip: a single shared dialog owned by
	// the options panel, keyed by the repo+tag the user acted on, so one dialog is
	// mounted for the whole panel instead of one per chip.
	// The acted-on target is kept after closing so the copy stays rendered through
	// the dialog's close transition.
	let pending: { action: ModelDownloadConfirmAction; repoWithTag: string } = $state({
		action: ModelDownloadConfirmAction.CANCEL,
		repoWithTag: ''
	});
	let confirmOpen = $state(false);

	function requestCancel(repoWithTag: string) {
		pending = { action: ModelDownloadConfirmAction.CANCEL, repoWithTag };
		confirmOpen = true;
	}

	function requestDelete(repoWithTag: string) {
		pending = { action: ModelDownloadConfirmAction.DELETE, repoWithTag };
		confirmOpen = true;
	}

	function stateFor(
		repoWithTag: string,
		filePath: string,
		isSidecar: boolean
	): ModelDownloadEntryState {
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
		const files: (ModelSelectableFile & { state: ModelDownloadEntryState })[] = [];

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

	function optionFor(file: ModelSelectableFile): ModelQuantOption {
		return {
			label: labelFor(file.path),
			path: file.path
		};
	}

	/** Non-draft quants for the command's base select, in row order. */
	let mainOptions = $derived(
		selectableFiles.filter((f) => f.kind === ModelSelectableFileKind.MAIN).map(optionFor)
	);

	/**
	 * Draft options for the command's draft select, with their sidecar type
	 * (MTP, DFLASH...) since a repo can ship more than one draft flavour.
	 */
	let draftOptions = $derived(
		selectableFiles
			.filter((f) => f.kind === ModelSelectableFileKind.DRAFT)
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
				<ModelsDiscoverDetailsDownloadOptionsRow
					bitDepth={row.bitDepth}
					files={row.files}
					onRequestCancel={requestCancel}
					onRequestDelete={requestDelete}
				/>
			{/each}
		</div>

		<!-- Terminal command preview, standalone: its picks are not bound to the chips. -->
		<div class="border-t border-border/50 px-4 pt-3.5 pb-4 dark:border-border/35">
			<ModelsDiscoverDetailsDownloadOptionsDownloadCommand {draftOptions} {mainOptions} {modelId} />
		</div>
	</section>
{/if}

<DialogConfirmDownload
	action={pending.action}
	onClose={() => (confirmOpen = false)}
	open={confirmOpen}
	repoWithTag={pending.repoWithTag}
/>
