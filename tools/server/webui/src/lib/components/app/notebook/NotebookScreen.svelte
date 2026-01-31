<script lang="ts">
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Play, Square } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';

	let { content } = $state(notebookStore);
</script>

<div class="flex h-full flex-col p-4 md:p-6">
	<div class="mb-4 flex items-center justify-between">
		<h1 class="text-2xl font-semibold">Notebook</h1>
		<div class="flex gap-2">
			{#if notebookStore.isGenerating}
				<Button variant="destructive" onclick={() => notebookStore.stop()}>
					<Square class="mr-2 h-4 w-4" />
					Stop
				</Button>
			{:else}
				<Button onclick={() => notebookStore.generate()}>
					<Play class="mr-2 h-4 w-4" />
					Generate
				</Button>
			{/if}
		</div>
	</div>

	<div class="flex-1 overflow-hidden rounded-lg border bg-background shadow-sm">
		<textarea
			class="h-full w-full resize-none border-0 bg-transparent p-4 font-mono text-sm focus:ring-0 focus-visible:ring-0"
			placeholder="Enter your text here..."
			bind:value={notebookStore.content}
		></textarea>
	</div>

	<div class="mt-4 text-xs text-muted-foreground">
		<p>
			Model: {config().model || 'Default'} | Temperature: {config().temperature ?? 0.8} | Max Tokens: {config()
				.max_tokens ?? -1}
		</p>
	</div>
</div>
