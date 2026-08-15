<script lang="ts">
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import Textarea from '$lib/components/ui/textarea/textarea.svelte';
	import { Play, Square, Undo, Redo } from '@lucide/svelte';
	import { settingsStore } from '$lib/stores/settings.svelte';
	import {
		ChatFormContextGauge,
		ChatMessageStatistics,
		DialogChatError,
		KeyboardShortcutInfo,
		ModelsSelectorDropdown
	} from '$lib/components/app';
	import ContextGaugePopup from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugePopup.svelte';
	import { ProcessingText, ProcessingInfo } from '$lib/components/app/misc';
	import { ErrorDialogType } from '$lib/enums';

	import { chatStore } from '$lib/stores/chat.svelte';
	import { contextStatsStore } from '$lib/stores/context-stats.svelte';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	import {
		AUTO_SCROLL_AT_BOTTOM_THRESHOLD,
		AUTO_SCROLL_INTERVAL
	} from '$lib/constants';

	// Note: this constant was previously in the constants/auto-scroll file above but was
	// removed in the PR #20999
	const INITIAL_SCROLL_DELAY = 50;

	import { onMount, onDestroy } from 'svelte';

	let disableAutoScroll = $derived(Boolean(settingsStore.config.disableAutoScroll));
	let showMessageStats = $derived(settingsStore.config.showMessageStats);
	let autoScrollEnabled = $state(true);
	let scrollContainer: HTMLTextAreaElement | null = $state(null);
	let lastScrollTop = $state(0);
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let scrollTimeout: ReturnType<typeof setTimeout> | undefined;
	let userScrolledUp = $state(false);

	let isRouter = $derived(serverStore.isRouterMode);
	let processingState = $derived(notebookStore.processingState);
	let hasGeneratedStats = $derived(
		notebookStore.isGenerating ||
			processingState?.promptMs !== undefined ||
			(processingState?.tokensDecoded ?? 0) > 0
	);

	let errorDialog = $derived(notebookStore.error);
	let canUndo = $derived(notebookStore.previousContent !== null && !notebookStore.isGenerating);
	let canRedo = $derived(notebookStore.undoneContent !== null && !notebookStore.isGenerating);

	function handleInput(e: Event) {
		const target = e.target as HTMLTextAreaElement;
		notebookStore.content = target.value;
		notebookStore.resetUndoRedo();
		if (activeModelId || !isRouter) {
			notebookStore.updateTokenCount(activeModelId ?? undefined);
		}
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

		if (activeModelId) {
			await notebookStore.generate(activeModelId);
		}
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
		const options = modelsStore.models;

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = modelsStore.selectedModelId;
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		return null;
	});

	let hasModelSelected = $derived(!isRouter || !!modelsStore.selectedModelId);

	let generateTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}

		if (notebookStore.content.length == 0) {
			return 'Input some text first';
		}

		return '';
	});

	let canGenerate = $derived(notebookStore.content.length > 0 && hasModelSelected);
	let isDisabled = $derived(!canGenerate);

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
		chatStore.setActiveProcessingConversation('notebook');
		contextStatsStore.notebookMode = true;
		if (notebookStore.content.length > 0) {
			notebookStore.updateTokenCount(activeModelId ?? undefined);
		}
		if (!disableAutoScroll) {
			setTimeout(() => scrollToBottom('instant'), INITIAL_SCROLL_DELAY);
		}
	});

	onDestroy(() => {
		contextStatsStore.notebookMode = false;
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
		// This should prevent the browser from closing the tab if there is content in the notebook
		if (notebookStore.content.length > 0) {
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

<div class="flex h-dvh flex-col">
	<header
		class="flex items-center justify-center border-b border-border/40 bg-background/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/60"
	>
		<h1 class="text-lg font-semibold">Notebook</h1>
	</header>

	<div class="min-h-0 flex-1 overflow-hidden px-2 pt-2 pb-0 md:px-4 md:pt-4">
		<Textarea
			bind:ref={scrollContainer}
			onscroll={handleScroll}
			value={notebookStore.content}
			oninput={handleInput}
			class="h-full min-h-[100px] w-full resize-none rounded-xl border-none bg-muted p-4 text-base focus-visible:ring-0 md:p-6"
			placeholder="Enter your text here..."
		/>
	</div>

	{#if notebookStore.processingState?.status === 'preparing'}
		<ProcessingText cls="px-4 md:px-6" processingText={notebookStore.getPromptProcessingText()} />
	{:else if showMessageStats}
		<ProcessingInfo
			visible={notebookStore.processingState?.status === 'generating'}
			processingDetails={notebookStore.getProcessingDetails()}
		/>
	{/if}

	<div class="relative bg-background p-2 md:p-4" data-gauge-container>
		{#snippet generateButton()}
			<Button
				disabled={isDisabled}
				onclick={notebookStore.isGenerating ? handleStop : handleGenerate}
				size="sm"
				variant={notebookStore.isGenerating ? 'destructive' : 'default'}
				class="min-w-[120px] gap-2"
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

		{#snippet statisticsWidget()}
			{#if processingState}
				<ChatMessageStatistics
					promptTokens={processingState.promptTokens}
					promptMs={processingState.promptMs}
					predictedTokens={processingState.tokensDecoded}
					predictedMs={processingState.predictedMs}
					isLive={notebookStore.isGenerating}
					isProcessingPrompt={notebookStore.isGenerating && processingState.tokensDecoded === 0}
				/>
			{/if}
		{/snippet}

		<div class="flex flex-col gap-2 md:gap-3">
			{#if showMessageStats && hasGeneratedStats && processingState}
				<div class="flex items-center justify-end md:hidden">
					{@render statisticsWidget()}
				</div>
			{/if}

			<div class="flex flex-wrap items-center justify-between gap-2.5">
				<div class="flex flex-wrap items-center gap-2">
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Button variant="ghost" size="icon" disabled={!canUndo} onclick={handleUndo}>
								<Undo class="h-4 w-4" />
							</Button>
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Undo last generation</p>
							<KeyboardShortcutInfo keys={['cmd', 'z']} class="w-full justify-center opacity-100" />
						</Tooltip.Content>
					</Tooltip.Root>

					<Tooltip.Root>
						<Tooltip.Trigger>
							<Button variant="ghost" size="icon" disabled={!canRedo} onclick={handleRedo}>
								<Redo class="h-4 w-4" />
							</Button>
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p>Redo last generation</p>
							<KeyboardShortcutInfo keys={['cmd', 'y']} class="w-full justify-center opacity-100" />
						</Tooltip.Content>
					</Tooltip.Root>

					<Tooltip.Root>
						<Tooltip.Trigger>
							{@render generateButton()}
						</Tooltip.Trigger>

						<Tooltip.Content>
							{#if generateTooltip}
								<p>{generateTooltip}</p>
							{:else}
								<div class="flex items-center justify-center py-1">
									<KeyboardShortcutInfo keys={['shift', 'enter']} class="opacity-100" />
								</div>
							{/if}
						</Tooltip.Content>
					</Tooltip.Root>

					<ModelsSelectorDropdown
						forceForegroundText={true}
						useGlobalSelection={true}
						disabled={notebookStore.isGenerating}
					/>
				</div>

				<div class="flex items-center gap-2.5">
					<ChatFormContextGauge notebookMode={true} />

					{#if showMessageStats && hasGeneratedStats && processingState}
						<div class="hidden md:block">
							{@render statisticsWidget()}
						</div>
					{/if}
				</div>
			</div>
		</div>

		<ContextGaugePopup />
	</div>

	<DialogChatError
		message={errorDialog?.message ?? ''}
		contextInfo={errorDialog?.contextInfo}
		onOpenChange={handleErrorDialogOpenChange}
		open={Boolean(errorDialog)}
		type={errorDialog?.type ?? ErrorDialogType.SERVER}
	/>
</div>
