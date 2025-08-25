<script>
	import { Upload } from "@lucide/svelte";

</script>

	<div
		class="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
	>
		<div
			class="bg-background border-border flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-12 shadow-lg"
		>
			<Upload class="text-muted-foreground mb-4 h-12 w-12" />
			<p class="text-foreground text-lg font-medium">Attach a file</p>
			<p class="text-muted-foreground text-sm">Drop your files here to upload</p>
		</div>
	</div>