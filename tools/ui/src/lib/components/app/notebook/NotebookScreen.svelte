<script lang="ts">
	import NotebookEditor from './NotebookEditor.svelte';
	import { Play, Redo, Square, Undo } from '@lucide/svelte';
	import {
		ChatFormContextGauge,
		ChatMessageStatistics,
		DialogChatError,
		KeyboardShortcutInfo,
		ModelsSelector
	} from '$lib/components/app';
	import ContextGaugePopup from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugePopup.svelte';
	import Button from '$lib/components/ui/button/button.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		AUTO_SCROLL_AT_BOTTOM_THRESHOLD,
		AUTO_SCROLL_INTERVAL,
		ICON_CLASS_DEFAULT,
		INPUT_CLASSES
	} from '$lib/constants';
	import { ErrorDialogType } from '$lib/enums';
	import { contextStatsStore } from '$lib/stores/chat/context-stats.svelte';
	import { chatStore } from '$lib/stores/chat/index.svelte';
	import { modelsStore } from '$lib/stores/models/index.svelte';
	import { notebookStore } from '$lib/stores/notebook.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { settingsStore } from '$lib/stores/settings/index.svelte';

	// Note: this constant was previously in the constants/auto-scroll file above but was
	// removed in the PR #20999
	const INITIAL_SCROLL_DELAY = 50;

	import { onDestroy, onMount } from 'svelte';

	let disableAutoScroll = $derived(Boolean(settingsStore.config.disableAutoScroll));
	let showMessageStats = $derived(settingsStore.config.showMessageStats);
	let autoScrollEnabled = $state(true);
	let scrollContainer: HTMLElement | null = $state(null);
	let editor: NotebookEditor | undefined = $state();
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

	function handleChange(value: string) {
		notebookStore.content = value;
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

		const { clientHeight, scrollHeight, scrollTop } = scrollContainer;
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
			behavior,
			top: scrollContainer?.scrollHeight
		});
	}

	onMount(() => {
		chatStore.processing.setActiveConversation('notebook');
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

		// Focused editor handled it as a text-level undo/redo
		if (event.defaultPrevented) return;

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

<!-- Clicks on the empty page below the text continue writing at its end -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div
	bind:this={scrollContainer}
	class="flex h-dvh flex-col overflow-y-auto px-4 pt-16 md:pt-8"
	onclick={(event) => {
		if (event.target === event.currentTarget) editor?.focusEnd();
	}}
	onscroll={handleScroll}
>
	<NotebookEditor
		bind:this={editor}
		class="w-full min-w-0"
		onChange={handleChange}
		placeholder="Enter your text here..."
		value={notebookStore.content}
	/>

	<!-- Bar right under the text, styled like the chat form; once the text outgrows the
	     page it sticks to the bottom edge -->
	<div class="pointer-events-none sticky bottom-0 z-10 flex justify-center pt-4 pb-4">
		<div
			class="{INPUT_CLASSES} pointer-events-auto relative flex w-full max-w-3xl items-center gap-1.5 rounded-full px-2 py-1.5 backdrop-blur-md"
			data-gauge-container
		>
			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						class="h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9"
						disabled={!canUndo}
						onclick={handleUndo}
						variant="ghost"
					>
						<span class="sr-only">Undo last generation</span>

						<Undo class={ICON_CLASS_DEFAULT} />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Undo last generation</p>

					<KeyboardShortcutInfo class="w-full justify-center opacity-100" keys={['cmd', 'z']} />
				</Tooltip.Content>
			</Tooltip.Root>

			<Tooltip.Root>
				<Tooltip.Trigger>
					<Button
						class="h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9"
						disabled={!canRedo}
						onclick={handleRedo}
						variant="ghost"
					>
						<span class="sr-only">Redo last generation</span>

						<Redo class={ICON_CLASS_DEFAULT} />
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Redo last generation</p>

					<KeyboardShortcutInfo class="w-full justify-center opacity-100" keys={['cmd', 'y']} />
				</Tooltip.Content>
			</Tooltip.Root>

			<div class="ml-auto flex min-w-0 items-center gap-1.5">
				{#if showMessageStats && hasGeneratedStats && processingState}
					<div class="hidden md:block">
						<ChatMessageStatistics
							isLive={notebookStore.isGenerating}
							isProcessingPrompt={notebookStore.isGenerating && processingState.tokensDecoded === 0}
							predictedMs={processingState.predictedMs}
							predictedTokens={processingState.tokensDecoded}
							promptMs={processingState.promptMs}
							promptTokens={processingState.promptTokens}
						/>
					</div>
				{/if}

				<ChatFormContextGauge notebookMode={true} />

				<ModelsSelector
					disabled={notebookStore.isGenerating}
					forceForegroundText={true}
					useGlobalSelection={true}
				/>

				{#if notebookStore.isGenerating}
					<Button
						class="group h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9 hover:bg-destructive/10!"
						onclick={handleStop}
						variant="secondary"
					>
						<span class="sr-only">Stop</span>

						<Square
							class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive"
						/>
					</Button>
				{:else}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<Button
								class="h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9"
								disabled={isDisabled}
								onclick={handleGenerate}
							>
								<span class="sr-only">Generate</span>

								<Play class="{ICON_CLASS_DEFAULT} fill-current" />
							</Button>
						</Tooltip.Trigger>

						<Tooltip.Content>
							{#if generateTooltip}
								<p>{generateTooltip}</p>
							{:else}
								<div class="flex items-center gap-2">
									<p>Generate</p>

									<KeyboardShortcutInfo class="opacity-100" keys={['shift', 'enter']} />
								</div>
							{/if}
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}
			</div>

			<ContextGaugePopup />
		</div>
	</div>

	<DialogChatError
		contextInfo={errorDialog?.contextInfo}
		message={errorDialog?.message ?? ''}
		onOpenChange={handleErrorDialogOpenChange}
		open={Boolean(errorDialog)}
		type={errorDialog?.type ?? ErrorDialogType.SERVER}
	/>
</div>
