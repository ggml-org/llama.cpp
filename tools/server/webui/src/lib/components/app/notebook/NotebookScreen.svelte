<script lang="ts">
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Play, Square, Settings, Info } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import ChatMessageStatistics from '$lib/components/app/chat/ChatMessages/ChatMessageStatistics.svelte';
	import DialogChatSettings from '$lib/components/app/dialogs/DialogChatSettings.svelte';
	import DialogModelInformation from '$lib/components/app/dialogs/DialogModelInformation.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';

	let { content } = $state(notebookStore);
	let settingsOpen = $state(false);
	let modelInfoOpen = $state(false);

	let inputContent = $state(content);

	// Sync local input with store content
	$effect(() => {
		inputContent = notebookStore.content;
	});

	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		notebookStore.content = target.value;
	}

	async function handleGenerate() {
		await notebookStore.generate();
	}

	function handleStop() {
		notebookStore.stop();
	}

	let currentModel = $derived(
		modelsStore.models.find((m) => m.id === config().model) || modelsStore.models[0]
	);
</script>

<div class="flex h-full flex-col">
	<header
		class="flex items-center justify-between border-b border-border/40 bg-background/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/60"
	>
		<div class="w-10"></div>
		<!-- Spacer for centering -->
		<h1 class="text-lg font-semibold">Notebook</h1>
		<Button variant="ghost" size="icon" onclick={() => (settingsOpen = true)}>
			<Settings class="h-5 w-5" />
		</Button>
	</header>

	<div class="flex-1 overflow-y-auto p-4 md:p-6">
		<Textarea
			value={inputContent}
			oninput={handleInput}
			class="h-full min-h-[500px] w-full resize-none rounded-xl border-none bg-muted p-4 text-base focus-visible:ring-0 md:p-6"
			placeholder="Enter your prompt here..."
		/>
	</div>

	<div class="border-t border-border/40 bg-background p-4 md:px-6 md:py-4">
		<div class="flex items-center justify-between gap-4">
			<div class="flex items-center gap-2">
				<Button
					onclick={notebookStore.isGenerating ? handleStop : handleGenerate}
					size="sm"
					variant={notebookStore.isGenerating ? 'destructive' : 'default'}
					class="gap-2"
				>
					{#if notebookStore.isGenerating}
						<Square class="h-4 w-4 fill-current" />
						Stop
					{:else}
						<Play class="h-4 w-4 fill-current" />
						Generate
					{/if}
				</Button>

				<Button variant="ghost" size="icon" onclick={() => (modelInfoOpen = true)}>
					<Info class="h-4 w-4" />
				</Button>
			</div>

			<ChatMessageStatistics
				predictedTokens={notebookStore.predictedTokens}
				predictedMs={notebookStore.predictedMs}
				promptTokens={notebookStore.promptTokens}
				promptMs={notebookStore.promptMs}
				isLive={notebookStore.isGenerating}
			/>
		</div>
	</div>

	<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
	<DialogModelInformation open={modelInfoOpen} onOpenChange={(open) => (modelInfoOpen = open)} />
</div>
