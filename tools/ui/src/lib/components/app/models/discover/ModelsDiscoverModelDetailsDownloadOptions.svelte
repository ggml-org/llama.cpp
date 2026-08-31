<script lang="ts">
	import DownloadProgressBar from './DownloadProgressBar.svelte';
	import { Check, Copy, Download } from '@lucide/svelte';
	import { ToggleGroup, ToggleGroupItem } from '$lib/components/ui/toggle-group';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';
	import { HuggingFaceService, ModelsService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { HfModelSibling } from '$lib/types/huggingface';
	import { copyToClipboard, estimateModelMemoryBytes } from '$lib/utils';

	/** Download state of a single repo entry, injected by the integration layer. */
	export interface DownloadEntryState {
		isDownloading: boolean;
		progress: ModelDownloadProgress | null;
		isDownloaded: boolean;
		isFailed: boolean;
	}

	type BitDepthRow = { bitDepth: number; files: HfModelSibling[] };

	interface SelectedDownload {
		filePath: string;
		quant: string | null;
		sidecar: ModelSidecar | null;
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

	/** Kind of a file path: the main weights, a draft sidecar, or an aux sidecar (mmproj). */
	function classify(path: string): 'main' | 'draft' | 'aux' {
		const sidecar = HuggingFaceService.extractQuantMeta(path)?.sidecar;

		if (!sidecar) return 'main';

		return isAuxSidecar(sidecar) ? 'aux' : 'draft';
	}

	/**
	 * Constrained selection: at most one base model and one draft sidecar.
	 * Aux sidecars (mmproj) are unconstrained.
	 */
	function handleSelection(next: string[]) {
		const added = next.find((p) => !selectedPaths.includes(p));

		if (!added) {
			selectedPaths = next;

			return;
		}

		const kind = classify(added);
		const base = kind === 'aux' ? selectedPaths : selectedPaths.filter((p) => classify(p) !== kind);

		selectedPaths = [...base, added];
	}
	let copied = $state(false);

	/** All repo files in one lookup by path. */
	let fileByPath = $derived(new Map(bitDepthRows.flatMap((r) => r.files).map((f) => [f.path, f])));

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

	/** Selection ordered main-quant-first, so the command reads naturally. */
	let selected: SelectedDownload[] = $derived.by(() => {
		const mains: SelectedDownload[] = [];
		const sidecars: SelectedDownload[] = [];

		for (const path of selectedPaths) {
			const file = fileByPath.get(path);

			if (!file) continue;

			const meta = HuggingFaceService.extractQuantMeta(file.path);
			const entry = {
				filePath: file.path,
				quant: meta?.quant ?? null,
				sidecar: meta?.sidecar ?? null
			};

			if (entry.sidecar && !isAuxSidecar(entry.sidecar)) sidecars.push(entry);
			else mains.push(entry);
		}

		return [...mains, ...sidecars];
	});

	/** First selected main quant, drives the `-hf <repo>:<quant>` tag. */
	let primaryQuant = $derived(selected.find((s) => !s.sidecar)?.quant ?? null);

	/** First selected draft sidecar, drives the `--spec-type` flag. */
	let draft = $derived(
		selected.find((s) => s.sidecar && !isAuxSidecar(s.sidecar))?.sidecar ?? null
	);

	// llama.cpp --spec-type value for each draft sidecar.
	const SPEC_TYPE: Record<ModelSidecar, string> = {
		[ModelAuxSidecar.MMPROJ]: '',
		[ModelDraftSidecar.DFLASH]: 'draft-dflash',
		[ModelDraftSidecar.DSPARK]: 'draft-dspark',
		[ModelDraftSidecar.EAGLE3]: 'eagle3',
		[ModelDraftSidecar.MTP]: 'draft-mtp'
	};

	function buttonClass(parts: { isDownloaded: boolean; isFailed: boolean }): string {
		const classes = [
			'relative inline-flex items-center gap-1 overflow-hidden rounded-md border bg-muted px-2 py-1 text-left font-mono text-xs transition-colors'
		];

		if (parts.isDownloaded && !parts.isFailed) {
			classes.push('border-foreground bg-muted');
		} else if (parts.isFailed) {
			classes.push('border-destructive');
		}

		return classes.join(' ');
	}

	async function copyCommand() {
		await copyToClipboard(serveCommand);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}

	/** Fire downloads for every selected entry. */
	function downloadSelected() {
		for (const sel of selected) {
			const tag = ModelsService.buildDownloadTag(modelId, sel.quant, sel.sidecar);

			if (modelsStore.status.hasFailedDownload(tag)) {
				void modelsStore.status
					.cancelDownload(tag)
					.then(() => modelsStore.status.downloadModel(tag, sel.filePath));
			} else {
				void modelsStore.status.downloadModel(tag, sel.filePath);
			}
		}

		selectedPaths = [];
	}

	/** The llama serve command for the current selection. */
	let serveCommand = $derived.by(() => {
		const quantTag = primaryQuant ? `${modelId}:${primaryQuant}` : modelId;
		const parts = ['llama', 'serve', '-hf', quantTag];

		if (draft) parts.push('-hfd', modelId, '--spec-type', SPEC_TYPE[draft]);

		return parts.join(' ');
	});
</script>

{#if bitDepthRows.length}
	<section class="space-y-3 rounded-xl border p-4">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<h2 class="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
				<Download class="h-4 w-4" />
				Downloadable options
			</h2>

			{#if selectedPaths.length}
				<span class="text-xs text-muted-foreground">{selected.length} selected</span>
			{/if}
		</div>

		<ToggleGroup
			class="flex flex-col"
			onValueChange={handleSelection}
			type="multiple"
			value={selectedPaths}
		>
			{#each bitDepthRows as row (row.bitDepth)}
				<div class="grid grid-cols-[5rem_1fr] items-start gap-3 py-2">
					<div class="pt-1 text-sm tabular-nums text-muted-foreground">
						{#if row.bitDepth === 99}
							Other
						{:else}
							{row.bitDepth}-bit
						{/if}
					</div>

					<div class="flex flex-wrap justify-end gap-1.5">
						{#each row.files as file (file.path)}
							{@const meta = HuggingFaceService.extractQuantMeta(file.path)}
							{@const basename = file.path.split('/').pop() ?? file.path}
							{@const label = meta?.quant ?? basename.replace(/\.gguf$/i, '')}
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
							{@const memoryGb = Math.ceil(estimateModelMemoryBytes(file.size ?? 0) / 1024 ** 3)}
							{@const tooltipText = isDownloading
								? `Downloading ${file.path}`
								: isDownloaded
									? `Already downloaded: ${file.path}`
									: isFailed
										? `Last attempt failed: ${file.path}`
										: `Download ${file.path} (requires ~${memoryGb} GB of memory)`}
							<Tooltip.Root>
								<Tooltip.Trigger>
									<ToggleGroupItem
										aria-label={tooltipText}
										class="relative inline-flex h-auto items-center gap-1 overflow-hidden rounded-md border bg-muted px-2 py-1 text-left font-mono text-xs transition-colors data-[state=on]:border-primary data-[state=on]:bg-primary/10 {buttonClass(
											{ isDownloaded, isFailed }
										)}"
										value={file.path}
									>
										{#if isFailed && !isDownloading && !isDownloaded}
											<span
												class="rounded bg-destructive px-1 py-0.5 text-[10px] font-semibold tracking-wide text-destructive-foreground uppercase"
											>
												Failed
											</span>
										{/if}

										{#if meta?.sidecar && !isAuxSidecar(meta.sidecar)}
											<span
												class="rounded bg-primary px-1 py-0.5 text-[10px] font-semibold tracking-wide text-primary-foreground uppercase"
											>
												{meta.sidecar}
											</span>
										{/if}

										<span class="font-medium {isDownloaded ? '' : 'text-muted-foreground/80'}"
											>{label}</span
										>

										<span class="-my-1 w-px self-stretch bg-border"></span>

										<span class={isDownloaded ? '' : 'text-muted-foreground/80'}>
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

		<!-- Terminal command + download CTA for the current selection -->
		<div class="space-y-2">
			<div
				class="flex items-center justify-between gap-2 rounded-md px-3 py-2"
				style="background: var(--code-background); border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
			>
				<span class="truncate font-mono text-xs text-foreground/90">{serveCommand}</span>

				<button
					aria-label="Copy command"
					class="shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
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
				disabled={selected.length === 0}
				onclick={downloadSelected}
				type="button"
			>
				<Download class="h-4 w-4" />

				{#if primaryQuant && draft}
					Download {primaryQuant} + {draft}
				{:else}
					Download
				{/if}
			</button>
		</div>
	</section>
{/if}
