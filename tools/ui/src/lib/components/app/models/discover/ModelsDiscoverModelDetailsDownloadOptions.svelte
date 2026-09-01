<script lang="ts">
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Check, Copy, Download } from '@lucide/svelte';
	import ModelsDownloadManagerDownloadStatusToast from '$lib/components/app/models/download-manager/ModelsDownloadManagerDownloadStatusToast.svelte';
	import { ToggleGroup, ToggleGroupItem } from '$lib/components/ui/toggle-group';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelSibling } from '$lib/types/huggingface';
	import { copyToClipboard, minMemoryTierGb } from '$lib/utils';
	import { toast } from 'svelte-sonner';

	/** Download state of a single repo entry, injected by the integration layer. */
	export interface DownloadEntryState {
		isDownloading: boolean;
		progress: ModelDownloadProgress | null;
		isDownloaded: boolean;
		isFailed: boolean;
	}

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

	/** A selectable GGUF, tagged with its kind: main weights, draft, or aux (mmproj). */
	type SelectableFile = HfModelSibling & { kind: 'main' | 'draft' | 'aux' };

	/** Option of a quant `<select>`; already-downloaded files stay non-selectable. */
	interface QuantOption {
		disabled: boolean;
		/** Quant token, or the file name when the file carries no quant (e.g. BF16). */
		label: string;
		path: string;
		size: number;
	}

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

	// min-w keeps the value clear of the native chevron: Safari sizes a select
	// to its widest option, so an exactly-as-wide value would otherwise let the
	// chevron overlap the text (draft selects are all same-width quants).
	const selectClass =
		'h-6 min-w-18 max-w-40 shrink-0 cursor-pointer rounded-md border border-input bg-transparent py-0 pr-3 pl-2 font-mono text-xs outline-none transition-colors hover:bg-accent/40 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]';

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

	/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
	function classify(path: string): 'main' | 'draft' | 'aux' {
		const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

		if (!sidecar) return 'main';

		return isAuxSidecar(sidecar) ? 'aux' : 'draft';
	}

	/** Display label of a file: its quant, else the file name without the extension. */
	function labelFor(path: string): string {
		const quant = HuggingFaceService.extractQuantMeta(path)?.quant;

		if (quant) return quant;

		const basename = path.split('/').pop() ?? path;

		return basename.replace(/\.gguf$/i, '');
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
			badge: HuggingFaceService.extractQuantMeta(f.path)?.sidecar
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

	let copied = $state(false);

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

	async function copyCommand() {
		await copyToClipboard(serveCommand);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

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
		const draftLabel = draftSidecar?.toUpperCase();

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
	<section class="rounded-xl border">
		<!-- header row is intentionally hidden for now
		<div class="flex flex-wrap items-center justify-between gap-2 px-4 pt-3 pb-1">
			<h2 class="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
				<Download class="h-4 w-4" />
				Downloadable options
			</h2>

			{#if selectedPaths.length}
				<span class="text-xs text-muted-foreground">{selected.length} selected</span>
			{/if}
		</div>
		-->

		<ToggleGroup
			class="flex w-full flex-col items-stretch divide-y px-4 pb-1"
			onValueChange={handleSelection}
			type="multiple"
			value={selectedPaths}
		>
			{#each bitDepthRows as row (row.bitDepth)}
				{@const mainFile = row.files.find(
					(f) => !HuggingFaceService.extractQuantMeta(f.path)?.sidecar
				)}
				{@const draftFile = row.files.find((f) => {
					const sidecar = HuggingFaceService.extractQuantMeta(f.path)?.sidecar;

					return sidecar && !isAuxSidecar(sidecar);
				})}
				{@const mainMemGb = mainFile ? minMemoryTierGb(mainFile.size ?? 0) : null}
				{@const draftMemGb = draftFile ? minMemoryTierGb(draftFile.size ?? 0) : null}
				<div class="grid grid-cols-[5rem_1fr] items-start gap-3 py-3">
					<div class="pt-1 text-sm tabular-nums text-muted-foreground">
						{#if row.bitDepth === 99}
							Other
						{:else}
							{row.bitDepth}-bit
						{/if}

						{#if mainMemGb}
							<span class="block text-[10px] whitespace-nowrap text-muted-foreground/60">
								needs at least {mainMemGb}GB{draftMemGb ? ` + ${draftMemGb}GB` : ''}+ memory
							</span>
						{/if}
					</div>

					<div class="flex flex-wrap justify-end gap-1.5">
						{#each row.files as file (file.path)}
							{@const meta = HuggingFaceService.extractQuantMeta(file.path)}
							{@const label = labelFor(file.path)}
							{@const hfRepoWithTag = ModelsService.buildDownloadTag(
								modelId,
								meta?.quant ?? null,
								meta?.sidecar ?? null
							)}
							{@const state = stateFor(hfRepoWithTag, file.path, Boolean(meta?.sidecar))}
							{@const isDownloading = state.isDownloading}
							{@const progress = state.progress}
							{@const isDownloaded = state.isDownloaded}
							{@const isFailed = state.isFailed}
							{@const tooltipText = isDownloading
								? `Downloading ${file.path}`
								: isDownloaded
									? `Already downloaded: ${file.path}`
									: isFailed
										? `Last attempt failed: ${file.path}`
										: `Download ${file.path}`}
							<Tooltip.Root>
								<Tooltip.Trigger>
									{#if isDownloaded}
										<!-- downloaded files are not selectable, just marked as done -->
										<div
											aria-disabled="true"
											aria-label={tooltipText}
											class="inline-flex cursor-default items-center gap-1 rounded-md border bg-muted px-2 py-1 font-mono text-xs opacity-70"
										>
											{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
												<span
													class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
												>
													{meta.sidecar}
												</span>
											{/if}

											<span class="font-medium">{label}</span>

											<span class="-my-1 w-px self-stretch bg-border"></span>

											<span>{HuggingFaceService.formatFileSize(file.size ?? 0)}</span>

											<Check class="h-3.5 w-3.5 shrink-0 text-green-500" />
										</div>
									{:else}
										<ToggleGroupItem
											aria-label={tooltipText}
											class="relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md! border bg-muted px-2 py-1 text-left font-mono text-xs transition-colors data-[state=on]:border-primary data-[state=on]:bg-primary/10 {isFailed
												? 'border-destructive'
												: ''}"
											value={file.path}
										>
											{#if isFailed && !isDownloading}
												<span
													class="rounded-md bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
												>
													Failed
												</span>
											{/if}

											{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
												<span
													class="rounded-md bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
												>
													{meta.sidecar}
												</span>
											{/if}

											<span class="font-medium">{label}</span>

											<span class="-my-1 w-px self-stretch bg-border"></span>

											<span>
												{#if isDownloading && progress && progress.totalBytes > 0}
													{Math.round((progress.downloadedBytes / progress.totalBytes) * 100)}%
												{:else}
													{HuggingFaceService.formatFileSize(file.size ?? 0)}
												{/if}
											</span>

											{#if isDownloading && progress}
												<DownloadProgressBar
													downloadedBytes={progress.downloadedBytes}
													overlay
													totalBytes={progress.totalBytes}
												/>
											{/if}
										</ToggleGroupItem>
									{/if}
								</Tooltip.Trigger>

								<Tooltip.Content>
									<p>{tooltipText}</p>
								</Tooltip.Content>
							</Tooltip.Root>
						{/each}
					</div>
				</div>
			{/each}
		</ToggleGroup>

		<!-- Terminal command with inline quant selects + download CTA -->
		<div class="space-y-2 border-t px-4 pt-3 pb-4">
			<div
				class="flex flex-wrap items-center gap-2 rounded-md px-3 py-2"
				style="background: var(--code-background); border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
			>
				<div
					class="flex min-w-0 flex-1 flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-foreground/90"
				>
					<span>llama</span>

					<span>serve</span>

					<span>-hf</span>

					<span class="truncate">{modelId}{commandMainQuant ? ':' : ''}</span>

					<!-- Base quant: always part of the command, the 8-bit file by default. -->
					{#if baseOptions.length}
						<select
							aria-label="Base model quantization"
							class="{selectClass} {mainEntry ? '' : 'border-dashed'} -ml-2"
							onchange={(e) => setPick('main', e.currentTarget.value)}
							title={mainEntry
								? undefined
								: 'Default quant - pick a file above or another quant here'}
							value={basePick}
						>
							{#each baseOptions as option (option.path)}
								<option disabled={option.disabled} value={option.path}>
									{option.label}
								</option>
							{/each}
						</select>
					{/if}

					<!-- Draft segment: appears once a draft is picked, quant inline too. -->
					{#if draftEntry && draftSidecar}
						<span>-hfd</span>

						<span class="truncate">{modelId}{commandDraftQuant ? ':' : ''}</span>

						<select
							aria-label="Draft model quantization"
							class={selectClass}
							onchange={(e) => setPick('draft', e.currentTarget.value)}
							value={draftPick}
						>
							{#each draftOptions as option (option.path)}
								<option disabled={option.disabled} value={option.path}>
									<!-- {option.badge ? `${option.badge.toUpperCase()} ` : ''} -->
									{option.label}
								</option>
							{/each}
						</select>

						<span>--spec-type</span>

						<span>{SPEC_TYPE[draftSidecar]}</span>
					{/if}
				</div>

				<button
					aria-label="Copy command"
					class="ml-auto shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
					onclick={copyCommand}
					type="button"
				>
					{#if copied}
						<Check class="h-3.5 w-3.5 text-green-500" />
					{:else}
						<Copy class="h-3.5 w-3.5" />
					{/if}
				</button>
			</div>

			<button
				class="inline-flex w-full cursor-pointer items-center justify-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
				disabled={selected.length === 0 && !commandMain}
				onclick={downloadSelected}
				type="button"
			>
				<Download class="h-4 w-4" />

				{downloadLabel}
			</button>
		</div>
	</section>
{/if}
