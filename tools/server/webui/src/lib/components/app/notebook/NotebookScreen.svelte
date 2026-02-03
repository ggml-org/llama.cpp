<script lang="ts">
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Play, Square, Settings, Undo, Redo, RulerDimensionLine } from '@lucide/svelte';
	import { config } from '$lib/stores/settings.svelte';
	import DialogChatSettings from '$lib/components/app/dialogs/DialogChatSettings.svelte';
	import { ChatMessageStatistics, DialogChatError, KeyboardShortcutInfo, ModelsSelector } from '$lib/components/app';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { content } = $state(notebookStore);
	let settingsOpen = $state(false);

	let inputContent = $state(content);

	import {
		AUTO_SCROLL_AT_BOTTOM_THRESHOLD,
		AUTO_SCROLL_INTERVAL,
		INITIAL_SCROLL_DELAY
	} from '$lib/constants/auto-scroll';
	import { onMount } from 'svelte';

	let disableAutoScroll = $derived(Boolean(config().disableAutoScroll));
	let showMessageStats = $derived(config().showMessageStats);
	let autoScrollEnabled = $state(true);
	let scrollContainer: HTMLTextAreaElement | null = $state(null);
	let lastScrollTop = $state(0);
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let scrollTimeout: ReturnType<typeof setTimeout> | undefined;
	let userScrolledUp = $state(false);

	let isRouter = $derived(isRouterMode());

	let errorDialog = $derived(notebookStore.error);
	let canUndo = $derived(notebookStore.previousContent !== null && !notebookStore.isGenerating);
	let canRedo = $derived(notebookStore.undoneContent !== null && !notebookStore.isGenerating);

	// Sync local input with store content
	$effect(() => {
		inputContent = notebookStore.content;
		notebookStore.updateTokenCount();
	});

	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		notebookStore.content = target.value;
		notebookStore.resetUndoRedo();
	}

	function handleErrorDialogOpenChange(open: boolean) {
		if (!open) {
			notebookStore.dismissError();
		}
	}

	async function handleGenerate() {
		if (!disableAutoScroll) {
			userScrolledUp = false;
			autoScrollEnabled = true;
			scrollToBottom();
		}

		if (notebookModel == null) {
			notebookModel = activeModelId;
		}
		await notebookStore.generate(notebookModel);
	}

	function handleUndo() {
		notebookStore.undo();
	}

	function handleRedo() {
		notebookStore.redo();
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

	function handleScroll() {
		if (disableAutoScroll || !scrollContainer) return;

		const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
		const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
		const isAtBottom = distanceFromBottom < AUTO_SCROLL_AT_BOTTOM_THRESHOLD;

		if (scrollTop < lastScrollTop && !isAtBottom) {
			userScrolledUp = true;
			autoScrollEnabled = false;
		} else if (isAtBottom && userScrolledUp) {
			userScrolledUp = false;
			autoScrollEnabled = true;
		}

		if (scrollTimeout) {
			clearTimeout(scrollTimeout);
		}

		scrollTimeout = setTimeout(() => {
			if (isAtBottom) {
				userScrolledUp = false;
				autoScrollEnabled = true;
			}
		}, AUTO_SCROLL_INTERVAL);

		lastScrollTop = scrollTop;
	}

	function scrollToBottom(behavior: ScrollBehavior = 'smooth') {
		if (disableAutoScroll) return;

		scrollContainer?.scrollTo({
			top: scrollContainer?.scrollHeight,
			behavior
		});
	}

	onMount(() => {
		if (!disableAutoScroll) {
			setTimeout(() => scrollToBottom('instant'), INITIAL_SCROLL_DELAY);
		}
	});

	$effect(() => {
		if (disableAutoScroll) {
			autoScrollEnabled = false;
			if (scrollInterval) {
				clearInterval(scrollInterval);
				scrollInterval = undefined;
			}
			return;
		}

		if (notebookStore.isGenerating && autoScrollEnabled) {
			scrollInterval = setInterval(() => scrollToBottom(), AUTO_SCROLL_INTERVAL);
		} else if (scrollInterval) {
			clearInterval(scrollInterval);
			scrollInterval = undefined;
		}
	});

	function handleBeforeUnload(event: BeforeUnloadEvent) {
		if (inputContent.length > 0) {
			event.preventDefault();
			event.returnValue = '';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		const isCtrlOrCmd = event.ctrlKey || event.metaKey;

		if (event.shiftKey && event.key === 'Enter') {
			event.preventDefault();
			if (notebookStore.isGenerating) {
				handleStop();
			} else if (canGenerate) {
				handleGenerate();
			}
		}

		if (isCtrlOrCmd && event.key === 'z') {
			event.preventDefault();
			if (canUndo) {
				handleUndo();
			}
		}

		if (isCtrlOrCmd && event.key === 'y') {
			event.preventDefault();
			if (canRedo) {
				handleRedo();
			}
		}
	}
</script>

<svelte:window onbeforeunload={handleBeforeUnload} onkeydown={handleKeydown} />

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

	<div class="flex-1 overflow-y-auto pb-0 pt-2 md:pt-4 px-2 md:px-4">
		<Textarea
			bind:ref={scrollContainer}
			onscroll={handleScroll}
			value={inputContent}
			oninput={handleInput}
			class="h-full min-h-[100px] w-full resize-none rounded-xl border-none bg-muted p-4 text-base focus-visible:ring-0 md:p-6"
			placeholder="Enter your text here..."
		/>
	</div>

	<div class="bg-background p-2 md:p-4">
		<div class="flex flex-col-reverse gap-4 md:flex-row md:items-center md:justify-between">
			<div class="flex items-center gap-2">
				<Tooltip.Root>
					<Tooltip.Trigger>
						<Button
							variant="ghost"
							size="icon"
							disabled={!canUndo}
							onclick={handleUndo}
						>
							<Undo class="h-4 w-4" />
						</Button>
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>Undo last generation</p>
						<KeyboardShortcutInfo keys={['ctrl', 'z']} class="opacity-100 w-full justify-center" />
					</Tooltip.Content>
				</Tooltip.Root>

				<Tooltip.Root>
					<Tooltip.Trigger>
						<Button
							variant="ghost"
							size="icon"
							disabled={!canRedo}
							onclick={handleRedo}
						>
							<Redo class="h-4 w-4" />
						</Button>
					</Tooltip.Trigger>
					<Tooltip.Content>
						<p>Redo last generation</p>
						<KeyboardShortcutInfo keys={['ctrl', 'y']} class="opacity-100 w-full justify-center" />
					</Tooltip.Content>
				</Tooltip.Root>

				{#snippet generateButton(props = {})}
					<Button
						disabled={isDisabled}
						onclick={notebookStore.isGenerating ? handleStop : handleGenerate}
						size="sm"
						variant={notebookStore.isGenerating ? 'destructive' : 'default'}
						class="gap-2 min-w-[120px]"
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

				<Tooltip.Root>
					<Tooltip.Trigger>
						{@render generateButton()}
					</Tooltip.Trigger>

					<Tooltip.Content>
						{#if generateTooltip}
							<p>{generateTooltip}</p>
						{:else}
							<KeyboardShortcutInfo keys={['shift', 'enter']} class="opacity-100 w-full justify-center" />
						{/if}
					</Tooltip.Content>
				</Tooltip.Root>

				<ModelsSelector
					currentModel={notebookModel}
					onModelChange={handleModelChange}
					forceForegroundText={true}
					useGlobalSelection={true}
					disabled={notebookStore.isGenerating}
				/>
			</div>

			{#if showMessageStats}
				<div class="flex w-full flex-col items-end justify-center gap-0.5 md:w-auto">
					{#if notebookStore.totalTokens > 0}
						<Tooltip.Root>
							<Tooltip.Trigger>
								<div class="pr-3.5 flex items-center gap-1.5 text-xs text-muted-foreground">
									<RulerDimensionLine class="h-3.5 w-3.5" />
									<span>{notebookStore.totalTokens} tokens</span>
								</div>
							</Tooltip.Trigger>
							<Tooltip.Content>
								<p>Total tokens</p>
							</Tooltip.Content>
						</Tooltip.Root>
					{/if}

					{#if notebookStore.promptTokens > 0 || notebookStore.predictedTokens > 0}
						<ChatMessageStatistics
							promptTokens={notebookStore.promptTokens}
							promptMs={notebookStore.promptMs}
							predictedTokens={notebookStore.predictedTokens}
							predictedMs={notebookStore.predictedMs}
							isLive={notebookStore.isGenerating}
							isProcessingPrompt={notebookStore.isGenerating && notebookStore.predictedTokens === 0}
						/>
					{/if}
				</div>
			{/if}
		</div>
	</div>

	<DialogChatSettings open={settingsOpen} onOpenChange={(open) => (settingsOpen = open)} />

	<DialogChatError
		message={errorDialog?.message ?? ''}
		contextInfo={errorDialog?.contextInfo}
		onOpenChange={handleErrorDialogOpenChange}
		open={Boolean(errorDialog)}
		type={errorDialog?.type ?? 'server'}
	/>
</div>
