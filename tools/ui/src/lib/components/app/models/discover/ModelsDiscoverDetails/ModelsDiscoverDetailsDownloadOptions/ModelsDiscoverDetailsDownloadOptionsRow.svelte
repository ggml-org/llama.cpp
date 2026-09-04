<script lang="ts">
	import ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton from './ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton.svelte';
	import { SelectableFileKind } from '$lib/enums';
	import type { DownloadEntryState, SelectableFile } from '$lib/types';
	import { minMemoryTierGb } from '$lib/utils';

	interface Props {
		/** Bit depth of the row; `99` renders as "Other". */
		bitDepth: number;
		/** Every GGUF of this bit depth, with download state attached. */
		files: (SelectableFile & { state: DownloadEntryState })[];
	}

	let { bitDepth, files }: Props = $props();

	let mainFile = $derived(files.find((f) => f.kind === SelectableFileKind.MAIN) ?? null);
	let draftFile = $derived(files.find((f) => f.kind === SelectableFileKind.DRAFT) ?? null);

	let mainMemGb = $derived(mainFile ? minMemoryTierGb(mainFile.size ?? 0) : null);
	let draftMemGb = $derived(draftFile ? minMemoryTierGb(draftFile.size ?? 0) : null);
</script>

<div class="grid grid-cols-[5rem_1fr] items-center gap-3 py-3">
	<div class="pt-1 text-sm tabular-nums text-muted-foreground">
		{#if bitDepth === 99}
			Other
		{:else}
			{bitDepth}-bit
		{/if}

		{#if mainMemGb}
			<span class="block text-[10px] whitespace-nowrap text-muted-foreground/60">
				needs at least {mainMemGb}GB{draftMemGb ? ` + ${draftMemGb}GB` : ''}+ memory
			</span>
		{/if}
	</div>

	<div class="flex flex-wrap justify-end gap-1.5">
		{#each files as file (file.path)}
			<ModelsDiscoverDetailsDownloadOptionsQuantDownloadButton entry={file.state} {file} />
		{/each}
	</div>
</div>
