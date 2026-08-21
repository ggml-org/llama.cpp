<script lang="ts">
	import { FileDown } from '@lucide/svelte';
</script>

<div
	class="pointer-events-none fixed inset-0 z-1000000 flex items-center justify-center bg-black/50 backdrop-blur-sm"
>
	<div
		class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border bg-background p-12 shadow-lg"
	>
		<FileDown class="mb-4 h-12 w-12 text-muted-foreground" />

		<p class="text-lg font-medium text-foreground">Import conversations or settings</p>

		<p class="text-sm text-muted-foreground">Drop your files here to import</p>
	</div>
</div>
