<script lang="ts">
	import { FileDown } from '@lucide/svelte';
	import { getDropImportContext } from '#lib/contexts/index.ts';
	import { HtmlInputType } from '#lib/enums/index.ts';
	import { toast } from 'svelte-sonner';

	const dropImport = getDropImportContext();

	let dragCounter = $state(0);
	let input = $state<HTMLInputElement>();
	let isDragOver = $derived(dragCounter > 0);

	async function importFiles(files: File[]) {
		if (!dropImport || files.length === 0) return;

		for (const file of await dropImport.importExports(files)) {
			toast.error(`${file.name} is not a Llama Chat export`);
		}
	}

	// every drag event stops here, so the app-wide import overlay stays off this area
	function handleDragEnter(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		dragCounter++;
	}

	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		dragCounter = Math.max(0, dragCounter - 1);
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		event.stopPropagation();
		dragCounter = 0;
		importFiles(Array.from(event.dataTransfer?.files ?? []));
	}

	function handleChange() {
		importFiles(Array.from(input?.files ?? []));

		if (input) input.value = '';
	}
</script>

<!-- No `accept` filter: iOS resolves each entry to a UTI and has none for `.jsonl`,
	 which greys out exported conversations in the file picker. -->
<input
	bind:this={input}
	class="hidden"
	multiple
	onchange={handleChange}
	type={HtmlInputType.FILE}
/>

<button
	class="flex w-full flex-col items-center justify-center gap-1 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none {isDragOver
		? 'border-primary bg-accent'
		: 'border-border hover:border-muted-foreground/50 hover:bg-muted/50'}"
	onclick={() => input?.click()}
	ondragenter={handleDragEnter}
	ondragleave={handleDragLeave}
	ondragover={handleDragOver}
	ondrop={handleDrop}
	type="button"
>
	<FileDown class="mb-2 size-8 text-muted-foreground" />

	<span class="text-label">Drop files here or click to browse</span>

	<span class="text-caption text-muted-foreground">
		A conversation (.jsonl), several in a .zip, settings (.json) or a full backup
	</span>
</button>
