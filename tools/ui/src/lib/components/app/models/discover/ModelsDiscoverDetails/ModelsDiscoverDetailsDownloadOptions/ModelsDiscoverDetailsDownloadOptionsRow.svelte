<script lang="ts">
	import ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';
	import {
		BIT_DEPTH_LABEL_SUFFIX,
		GIGABYTE_LABEL,
		OTHER_BIT_DEPTH,
		OTHER_BIT_DEPTH_LABEL
	} from '$lib/constants';
	import { ModelSelectableFileKind } from '$lib/enums';
	import type { ModelDownloadEntryState, ModelSelectableFile } from '$lib/types';
	import { minMemoryTierGb } from '$lib/utils';

	interface Props {
		/** Bit depth of the row; `99` renders as "Other". */
		bitDepth: number;
		/** Every GGUF of this bit depth, with download state attached. */
		files: (ModelSelectableFile & { state: ModelDownloadEntryState })[];
		/** Forwarded to each chip: ask the parent to confirm a cancel. */
		onRequestCancel?: (repoWithTag: string) => void;
		/** Forwarded to each chip: ask the parent to confirm a delete. */
		onRequestDelete?: (repoWithTag: string) => void;
	}

	let { bitDepth, files, onRequestCancel, onRequestDelete }: Props = $props();

	let mainFile = $derived(files.find((f) => f.kind === ModelSelectableFileKind.MAIN) ?? null);
	let draftFile = $derived(files.find((f) => f.kind === ModelSelectableFileKind.DRAFT) ?? null);

	let mainMemGb = $derived(mainFile ? minMemoryTierGb(mainFile.size ?? 0) : null);
	let draftMemGb = $derived(draftFile ? minMemoryTierGb(draftFile.size ?? 0) : null);
</script>

<div class="grid grid-cols-[5rem_1fr] items-center gap-3 py-3">
	<div class="pt-1 text-sm tabular-nums text-muted-foreground">
		{#if bitDepth === OTHER_BIT_DEPTH}
			{OTHER_BIT_DEPTH_LABEL}
		{:else}
			{bitDepth}{BIT_DEPTH_LABEL_SUFFIX}
		{/if}

		{#if mainMemGb}
			<span class="block text-[10px] whitespace-nowrap text-muted-foreground/60">
				needs at least {mainMemGb}{GIGABYTE_LABEL}{draftMemGb
					? ` + ${draftMemGb}${GIGABYTE_LABEL}`
					: ''} memory
			</span>
		{/if}
	</div>

	<div class="flex flex-wrap justify-end gap-1.5">
		{#each files as file (file.path)}
			<ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton
				entry={file.state}
				{file}
				{onRequestCancel}
				{onRequestDelete}
			/>
		{/each}
	</div>
</div>
