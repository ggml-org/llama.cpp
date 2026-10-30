<script lang="ts">
	import ModelsDiscoverDetailsDownloadOptionsDownloadCommand from './ModelsDiscoverDetailsDownloadOptionsDownloadCommand.svelte';
	import ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton from './ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton.svelte';
	import ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';
	import { ChevronDown } from '@lucide/svelte';
	import {
		BIT_DEPTH_LABEL_SUFFIX,
		GIGABYTE_LABEL,
		OTHER_BIT_DEPTH,
		OTHER_BIT_DEPTH_LABEL
	} from '$lib/constants';
	import type { ModelDraftSidecar } from '$lib/enums';
	import type { ModelQuantChip } from '$lib/types';
	import { minMemoryTierGb } from '$lib/utils';
	import { cubicOut } from 'svelte/easing';
	import { slide } from 'svelte/transition';

	interface Props {
		/** Bit depth of the row; `99` renders as "Other". */
		bitDepth: number;
		/** The row's main quant chips. */
		files: ModelQuantChip[];
		/** Full HuggingFace repo id, for the row's command preview. */
		modelId: string;
		/** Draft kind the panel pulls with the model, null when the sidecars are off. */
		draftKind?: ModelDraftSidecar | null;
		/** Draft kinds the repo ships, for the command's own pick. */
		draftKinds?: ModelDraftSidecar[];
		/** Forwarded to the command: the panel follows its draft pick. */
		onDraftKindChange?: (kind: ModelDraftSidecar) => void;
		/** Forwarded to each chip: ask the parent to confirm a cancel. */
		onRequestCancel?: (repoWithTag: string) => void;
		/** Forwarded to each chip: ask the parent to confirm a delete. */
		onRequestDelete?: (repoWithTag: string) => void;
	}

	let {
		bitDepth,
		draftKind = null,
		draftKinds = [],
		files,
		modelId,
		onDraftKindChange,
		onRequestCancel,
		onRequestDelete
	}: Props = $props();

	// collapsed until asked for; the panel remounts per model
	let commandOpen = $state(false);

	let mainFile = $derived(files[0] ?? null);
	let mainMemGb = $derived(mainFile ? minMemoryTierGb(mainFile.size ?? 0) : null);
	// the row's weights plus, when one rides along, the draft's own tier
	let totalMemGb = $derived.by(() => {
		if (mainMemGb === null) return null;

		const draftGb = mainFile?.companion?.size ? minMemoryTierGb(mainFile.companion.size) : 0;

		return mainMemGb + (draftGb ?? 0);
	});

	/** Drafts a downloaded quant is missing, each getting a chip of its own to add. */
	let missingDrafts = $derived(
		files.flatMap((file) =>
			file.state.isDownloaded && file.companion && !file.companion.onDisk
				? [{ companion: file.companion, repoWithTag: file.state.repoWithTag }]
				: []
		)
	);
</script>

<div class="py-3">
	<div class="grid grid-cols-[5rem_1fr] items-center gap-3 @max-lg:grid-cols-1 @max-lg:gap-1.5">
		<!-- the row's own label opens its command: it pins a quant of this row, so
		     the copy matches what the chips download -->
		<button
			aria-expanded={commandOpen}
			class="flex w-fit cursor-pointer flex-col items-start gap-0.5 text-left text-sm tabular-nums text-muted-foreground transition-colors hover:text-foreground @max-lg:flex-row @max-lg:items-baseline @max-lg:gap-2"
			onclick={() => (commandOpen = !commandOpen)}
			type="button"
		>
			<span class="flex items-center gap-1">
				{#if bitDepth === OTHER_BIT_DEPTH}
					{OTHER_BIT_DEPTH_LABEL}
				{:else}
					{bitDepth}{BIT_DEPTH_LABEL_SUFFIX}
				{/if}

				<ChevronDown
					class="size-icon-xs shrink-0 transition-transform duration-200 ease-[cubic-bezier(0.23,1,0.32,1)] {commandOpen
						? 'rotate-180'
						: ''}"
				/>
			</span>

			{#if totalMemGb}
				<span
					class="text-[10px] whitespace-nowrap text-muted-foreground/60 @max-lg:whitespace-normal"
				>
					needs at least {totalMemGb}{GIGABYTE_LABEL} memory
				</span>
			{/if}
		</button>

		<div class="flex flex-wrap justify-end gap-1.5 @max-lg:justify-start">
			{#each files as file (file.path)}
				<ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton
					{file}
					{onRequestCancel}
					{onRequestDelete}
				/>
			{/each}

			{#each missingDrafts as draft (draft.repoWithTag + ':' + draft.companion.kind)}
				<ModelsDiscoverDetailsDownloadOptionsDraftDownloadButton
					companion={draft.companion}
					repoWithTag={draft.repoWithTag}
				/>
			{/each}
		</div>
	</div>

	{#if commandOpen}
		<div transition:slide={{ duration: 200, easing: cubicOut }} class="pt-2">
			<ModelsDiscoverDetailsDownloadOptionsDownloadCommand
				{draftKind}
				{draftKinds}
				{modelId}
				{onDraftKindChange}
				quant={mainFile?.quant ?? null}
			/>
		</div>
	{/if}
</div>
