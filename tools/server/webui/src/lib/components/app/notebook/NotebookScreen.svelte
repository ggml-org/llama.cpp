<script lang="ts">
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Play, Square, Settings } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import DialogChatSettings from '$lib/components/app/dialogs/DialogChatSettings.svelte';
	import { ModelsSelector } from '$lib/components/app';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { content } = $state(notebookStore);
	let settingsOpen = $state(false);

	let inputContent = $state(content);

	let isRouter = $derived(isRouterMode());

	// Sync local input with store content
	$effect(() => {
		inputContent = notebookStore.content;
	});

	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		notebookStore.content = target.value;
	}

	async function handleGenerate() {
		if (notebookModel == null) {
			notebookModel = activeModelId;
		}
		await notebookStore.generate(notebookModel);
	}

	function handleStop() {
		notebookStore.stop();
	}

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		return null;
	});

	let hasModelSelected = $derived(!isRouter || !!selectedModelId());

	let isSelectedModelInCache = $derived.by(() => {
		if (!isRouter) return true;

		const currentModelId = selectedModelId();
		if (!currentModelId) return false;

		return modelOptions().some((option) => option.id === currentModelId);
	});

	let generateTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}

		if (!isSelectedModelInCache) {
			return 'Selected model is not available, please select another';
		}

		if (inputContent.length == 0) {
			return 'Input some text first';
		}

		return '';
	});

	let canGenerate = $derived(inputContent.length > 0 && hasModelSelected && isSelectedModelInCache);
	let isDisabled = $derived(!canGenerate);

	let notebookModel = $state(null);

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities: () => ({ vision: false, audio: false }), // Notebook doesn't require modalities
		onSuccess: async (modelName) => {
			notebookModel = modelName;
		}
	});
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
			placeholder="Enter your text here..."
		/>
	</div>

	<div class="border-t border-border/40 bg-background p-4 md:px-6 md:py-4">
		<div class="flex items-center justify-between gap-4">
			<div class="flex items-center gap-2">
				{#snippet generateButton(props = {})}
					<Button
						disabled={isDisabled}
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
				{/snippet}

				{#if generateTooltip}
					<Tooltip.Root>
						<Tooltip.Trigger>
							{@render generateButton()}
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>{generateTooltip}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{:else}
					{@render generateButton()}
				{/if}

				<ModelsSelector
					currentModel={notebookModel}
					onModelChange={handleModelChange}
					forceForegroundText={true}
					useGlobalSelection={true}
					disabled={notebookStore.isGenerating}
				/>
			</div>
		</div>
	</div>

	<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />
</div>
