<script lang="ts">
	import {
		type BitDepthRow,
		classify,
		type DownloadEntryState,
		labelFor,
		type QuantOption,
		type SelectableFile
	} from './download-options.utils';
	import ModelsDiscoverModelDetailsDownloadOptionsDownloadButton from './ModelsDiscoverModelDetailsDownloadOptionsDownloadButton.svelte';
	import ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand from './ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand.svelte';
	import ModelsDiscoverModelDetailsDownloadOptionsRow from './ModelsDiscoverModelDetailsDownloadOptionsRow.svelte';
	import ModelsDownloadManagerDownloadStatusToast from '$lib/components/app/models/download-manager/ModelsDownloadManagerDownloadStatusToast.svelte';
	import { ToggleGroup } from '$lib/components/ui/toggle-group';
	import { type ModelSidecar } from '$lib/constants';
	import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import { toast } from 'svelte-sonner';

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

	let selectedPaths = $state<string[]>([]);

	// Quant picks shown inline in the command. `basePick` is preselected so the
	// command is readable before the user touches anything.
	let basePick = $state('');
	let draftPick = $state('');

	/** Bit depth to preselect for the base model; falls back to the closest one. */
	const DEFAULT_BASE_BIT_DEPTH = 4;

	function stateFor(repoWithTag: string, filePath: string, isSidecar: boolean): DownloadEntryState {
		if (getDownloadState) return getDownloadState(repoWithTag, filePath, isSidecar);

		return {
			isDownloaded: isSidecar
				? modelsStore.status.isDraftDownloaded(modelId, filePath)
				: modelsStore.status.isModelDownloaded(repoWithTag),
			isDownloading: modelsStore.status.isDownloadInProgress(repoWithTag),
			isFailed: modelsStore.status.hasFailedDownload(repoWithTag),
			progress: modelsStore.status.getDownloadProgress(repoWithTag)
		};
	}

	/**
	 * Every selectable file with its kind and download state, in row order.
	 * Single source of truth for the toggle group rows, the selects and the
	 * command.
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

	let mainFiles = $derived(selectableFiles.filter((f) => f.kind === 'main'));
	let draftFiles = $derived(selectableFiles.filter((f) => f.kind === 'draft'));

	/** Paths already downloaded on the server; static chips, disabled options. */
	let downloadedPaths = $derived(
		new Set(selectableFiles.filter((f) => f.state.isDownloaded).map((f) => f.path))
	);

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

	/** Bit depth of a file; `99` (Other) when it carries no quant token. */
	function bitDepthOf(path: string): number {
		const quant = HuggingFaceService.extractQuantMeta(path)?.quant ?? '';
		const match = quant.match(/^UD-(?=.)/i) ? quant.slice(3) : quant;
		const bit = match.match(/(?:I?Q|F)(\d+)/i)?.[1];

		return bit ? Number(bit) : 99;
	}

	function optionFor(file: SelectableFile & { state: DownloadEntryState }): QuantOption {
		return {
			disabled: file.state.isDownloaded,
			label: labelFor(file.path),
			path: file.path,
			size: file.size ?? 0
		};
	}

	/**
	 * Options per kind, in row (bit depth) order, so like quants line up
	 * between the two selects. Draft options carry their sidecar type (MTP,
	 * DFLASH...) since a repo can ship more than one draft flavour.
	 */
	let baseOptions = $derived(mainFiles.map(optionFor));
	let draftOptions = $derived(
		draftFiles.map((f) => ({
			...optionFor(f),
			badge: HuggingFaceService.extractQuantMeta(f.path)?.sidecar ?? null
		}))
	);

	/**
	 * Default base quant: the untouched selection falls back to the 4-bit file,
	 * or the lowest bit depth available when there is none. Downloaded files
	 * are skipped, so the command points at something there is still to fetch.
	 */
	function defaultBasePath(): string {
		const candidates = mainFiles.filter((f) => !downloadedPaths.has(f.path));
		const pool = candidates.length ? candidates : mainFiles;
		const preferred = pool.find((f) => bitDepthOf(f.path) === DEFAULT_BASE_BIT_DEPTH);

		if (preferred) return preferred.path;

		const ranked = [...pool].sort((a, b) => bitDepthOf(a.path) - bitDepthOf(b.path));

		return ranked[0]?.path ?? '';
	}

	/**
	 * Selection source of truth: the toggle group, the selects and the command
	 * all read it; the selects also mirror it. A pick replaces the previous one
	 * of its kind, an empty value drops it, aux sidecars are left alone.
	 */
	function setPick(kind: 'main' | 'draft', path: string) {
		const rest = selectedPaths.filter((p) => classify(p) !== kind);

		selectedPaths = path ? [...rest, path] : rest;
	}

	/**
	 * Mirror the selection into the selects. With no main chip on, the default
	 * quant (4-bit, or the nearest one still to fetch) is preselected in both
	 * the command and the chips, so the command is always complete; the dashed
	 * border tells a default-only pick from a deliberate one.
	 */
	$effect(() => {
		const paths = selectedPaths;
		const mainPath = mainFiles.find((f) => paths.includes(f.path))?.path ?? '';

		if (mainPath) {
			basePick = mainPath;
		} else {
			const fallback = defaultBasePath();

			basePick = fallback;

			// Still fetchable: also toggle the chip on. When everything is
			// downloaded the chips stay off and only the command previews it.
			if (fallback && !downloadedPaths.has(fallback)) {
				selectedPaths = [...paths.filter((p) => classify(p) === 'aux'), fallback];
			}
		}

		draftPick = draftFiles.find((f) => paths.includes(f.path))?.path ?? '';
	});

	/**
	 * Constrained selection: at most one base model and one draft sidecar.
	 * Aux sidecars (mmproj) are unconstrained.
	 */
	function handleSelection(next: string[]) {
		const added = next.find((p) => !selectedPaths.includes(p));

		if (added && downloadedPaths.has(added)) return;

		if (!added) {
			selectedPaths = next;

			return;
		}

		const kind = classify(added);
		const base = kind === 'aux' ? selectedPaths : selectedPaths.filter((p) => classify(p) !== kind);

		selectedPaths = [...base, added];
	}

	/** Selection ordered main-quant-first, so the command reads naturally. */
	let selected = $derived.by(() => {
		const mains: SelectableFile[] = [];
		const drafts: SelectableFile[] = [];

		for (const file of selectableFiles) {
			if (!selectedPaths.includes(file.path)) continue;

			if (file.kind === 'draft') drafts.push(file);
			else mains.push(file);
		}

		return [...mains, ...drafts];
	});

	/** Selected main weights, drive the `-hf <repo>:<quant>` tag. */
	let mainEntry = $derived(selected.find((f) => f.kind === 'main') ?? null);

	/** Selected draft sidecar; its type drives the `--spec-type` flag. */
	let draftEntry = $derived(selected.find((f) => f.kind === 'draft') ?? null);

	let draftSidecar = $derived(
		draftEntry ? (HuggingFaceService.extractQuantMeta(draftEntry.path)?.sidecar ?? null) : null
	);

	/** True when the picked draft is the shared (target-borrowing) variant. */
	let draftShared = $derived(
		draftEntry ? (HuggingFaceService.extractQuantMeta(draftEntry.path)?.shared ?? false) : false
	);

	// LLAMA-APP-REUSE: --spec-type value for each draft sidecar
	const SPEC_TYPE: Record<ModelSidecar, string> = {
		[ModelAuxSidecar.MMPROJ]: '',
		[ModelDraftSidecar.DFLASH]: 'draft-dflash',
		[ModelDraftSidecar.DSPARK]: 'draft-dspark',
		[ModelDraftSidecar.EAGLE3]: 'eagle3',
		[ModelDraftSidecar.MTP]: 'draft-mtp'
	};

	/** Base file the command shows: the picked one, else whatever the base select holds. */
	let commandMain = $derived(mainEntry ?? mainFiles.find((f) => f.path === basePick) ?? null);

	/** Quant of the file the `-hf` tag points at; null when the file carries no quant. */
	let commandMainQuant = $derived(
		commandMain ? (HuggingFaceService.extractQuantMeta(commandMain.path)?.quant ?? null) : null
	);

	/** Quant of the file the `-hfd` tag points at. */
	let commandDraftQuant = $derived(
		draftEntry ? (HuggingFaceService.extractQuantMeta(draftEntry.path)?.quant ?? null) : null
	);

	/**
	 * Queue one download and surface a live progress toast keyed by the tag,
	 * so a retry updates the same toast instead of stacking a new one.
	 */
	async function queueDownload(tag: string) {
		try {
			await modelsStore.status.downloadModel(tag);
		} catch {
			// the store already toasted the failure
			return;
		}

		toast.custom(ModelsDownloadManagerDownloadStatusToast, {
			componentProps: { repoWithTag: tag },
			duration: Infinity,
			id: tag
		});
	}

	/**
	 * Files the CTA would download: the selection, else the command's base
	 * pick. The same set feeds the total size on the button label.
	 */
	let downloadQueue = $derived.by(() =>
		mainEntry
			? selected
			: [
					...selected.filter((f) => f.kind === 'main' || f.kind === 'aux'),
					...(commandMain ? [commandMain] : [])
				]
	);

	/** Total bytes the CTA would fetch; drives the size in the button label. */
	let downloadTotalBytes = $derived(downloadQueue.reduce((sum, file) => sum + (file.size ?? 0), 0));

	/**
	 * Fire downloads for the selection. With no main chip on, the command shows
	 * the default base quant, so the CTA queues exactly that file too.
	 */
	function downloadSelected() {
		for (const file of downloadQueue) {
			const meta = HuggingFaceService.extractQuantMeta(file.path);
			const tag = ModelsService.buildDownloadTag(
				modelId,
				meta?.quant ?? null,
				meta?.sidecar ?? null
			);

			if (modelsStore.status.hasFailedDownload(tag)) {
				void modelsStore.status.cancelDownload(tag).then(() => queueDownload(tag));
			} else {
				void queueDownload(tag);
			}
		}

		selectedPaths = [];
	}

	/** The llama serve command; always readable, driven by the inline selects. */
	// LLAMA-APP-REUSE: serve command shape (-hf / -hfd / --spec-type)
	let serveCommand = $derived.by(() => {
		const mainQuant = commandMain
			? HuggingFaceService.extractQuantMeta(commandMain.path)?.quant
			: null;
		const mainTag = mainQuant ? `${modelId}:${mainQuant}` : modelId;
		const parts = ['llama', 'serve', '-hf', mainTag];

		if (draftEntry && draftSidecar) {
			const draftQuant = HuggingFaceService.extractQuantMeta(draftEntry.path)?.quant;
			const draftTag = draftQuant ? `${modelId}:${draftQuant}` : modelId;

			parts.push('-hfd', draftTag, '--spec-type', SPEC_TYPE[draftSidecar]);
		}

		return parts.join(' ');
	});

	/**
	 * CTA label for the command shown: the selection, else the preview base
	 * quant, plus the total download size.
	 */
	let downloadLabel = $derived.by(() => {
		const main = mainEntry ?? commandMain;
		const draftLabel = draftSidecar
			? `${draftSidecar.toUpperCase()}${draftShared ? '-SHARED' : ''}`
			: undefined;

		let label = 'Download';

		if (main && draftLabel) label = `Download ${labelFor(main.path)} + ${draftLabel}`;
		else if (draftLabel) label = `Download ${draftLabel} draft`;
		else if (main) label = `Download ${labelFor(main.path)}`;

		if (downloadTotalBytes > 0) {
			label += ` · ${HuggingFaceService.formatFileSize(downloadTotalBytes)}`;
		}

		return label;
	});
</script>

{#if bitDepthRows.length}
	<section
		class="rounded-3xl border border-border/30 bg-muted/40 shadow-xs transition-[box-shadow,border-color] focus-within:border-border focus-within:shadow-sm dark:border-border/20 dark:bg-muted/50"
	>
		<!-- header row is intentionally hidden for now
		<div class="flex flex-wrap items-center justify-between gap-2 px-4 pt-3 pb-1">
			<h2 class="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
				Downloadable options
			</h2>

			{#if selectedPaths.length}
				<span class="text-xs text-muted-foreground">{selected.length} selected</span>
			{/if}
		</div>
		-->

		<ToggleGroup
			class="flex w-full flex-col items-stretch divide-y divide-border/50 px-4 pb-1 dark:divide-border/35"
			onValueChange={handleSelection}
			type="multiple"
			value={selectedPaths}
		>
			{#each rows as row (row.bitDepth)}
				<ModelsDiscoverModelDetailsDownloadOptionsRow bitDepth={row.bitDepth} files={row.files} />
			{/each}
		</ToggleGroup>

		<!-- Download CTA + terminal command with inline quant selects -->
		<div class="space-y-2.5 border-t border-border/50 px-4 pt-3.5 pb-4 dark:border-border/35">
			<ModelsDiscoverModelDetailsDownloadOptionsDownloadButton
				disabled={selected.length === 0 && !commandMain}
				label={downloadLabel}
				onclick={downloadSelected}
			/>

			<div aria-hidden="true" class="flex items-center gap-3">
				<span class="h-px flex-1 bg-border/50"></span>

				<span class="text-xs whitespace-nowrap text-muted-foreground">
					or run in your terminal
				</span>

				<span class="h-px flex-1 bg-border/50"></span>
			</div>

			<ModelsDiscoverModelDetailsDownloadOptionsDownloadCommand
				{baseOptions}
				{basePick}
				command={serveCommand}
				{draftOptions}
				{draftPick}
				draftQuant={commandDraftQuant}
				mainQuant={commandMainQuant}
				mainSelected={Boolean(mainEntry)}
				{modelId}
				onBasePick={(path) => setPick('main', path)}
				onDraftPick={(path) => setPick('draft', path)}
				specType={draftEntry && draftSidecar ? SPEC_TYPE[draftSidecar] : null}
			/>
		</div>
	</section>
{/if}
