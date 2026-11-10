<script lang="ts">
	import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
	import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
	import ChatScreenGreeting from './ChatScreenGreeting.svelte';
	import ChatScreenMobileHeader from './ChatScreenMobileHeader.svelte';
	import { disableScrollHandling, onNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ChatFormContextGauge,
		ChatMessages,
		ChatScreenDragOverlay,
		ChatScreenForm,
		ChatScreenServerError,
		ChatScreenStreamResumeStatus,
		ServerLoadingSplash
	} from '$lib/components/app';
	import {
		CHAT_FORM_BOTTOM_OFFSET,
		CHAT_FORM_CHROME_FADE_MS,
		CHAT_FORM_EASE_OUT,
		CHAT_FORM_MOVE_IN_EASING,
		CHAT_FORM_MOVE_IN_MS,
		CHAT_FORM_MOVE_OUT_EASING,
		CHAT_FORM_MOVE_OUT_FADE_END,
		CHAT_FORM_MOVE_OUT_MS,
		CHAT_FORM_MOVE_SHIFT_PX,
		CHAT_FORM_MOVE_TO_BOTTOM_IN_MS,
		CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS,
		ROUTES
	} from '$lib/constants';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import { useChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
	import { useChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
	import { useShowContextGauge } from '$lib/hooks/use-context-gauge.svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { createStickToBottom } from '$lib/hooks/use-stick-to-bottom.svelte';
	import {
		chatStore,
		conversationsStore,
		deviceStore,
		serverStore,
		settingsStore,
		uiStore
	} from '$lib/stores';
	import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
	import { tick, untrack } from 'svelte';
	import { fade } from 'svelte/transition';

	let { showCenteredEmpty = false } = $props();

	let emptyFileNames = $state<string[]>([]);
	let showDeleteDialog = $state(false);
	let showEmptyFileDialog = $state(false);
	let isEmpty = $derived(
		showCenteredEmpty && conversationsStore.activeMessages.length === 0 && !chatStore.isLoading
	);
	// where the form is drawn: follows `isEmpty` once the form has faded out
	let isFormCentered = $state(untrack(() => isEmpty));
	// it fades while it moves between the centered and bottom positions (see moveForm);
	// otherwise its position eases with the form expanding and collapsing
	let isFormMoving = $state(false);
	let isFormHeldAtBottom = $state(false);
	let isFormLifted = $derived(isFormCentered && !isFormHeldAtBottom);
	let formWrapper: HTMLElement | undefined = $state();
	let chatScreenForm: ChatScreenForm | undefined = $state();
	let activeErrorDialog = $derived(chatStore.errorDialogState);
	let isServerLoading = $derived(serverStore.loading);
	let hasPropsError = $derived(!!serverStore.error);
	let isCurrentConversationLoading = $derived(chatStore.isLoading || chatStore.isStreaming());
	let chatFormBottomPosition = $derived.by(() => {
		if (!deviceStore.isMobile) return CHAT_FORM_BOTTOM_OFFSET.desktop;

		if (deviceStore.isStandalone) return CHAT_FORM_BOTTOM_OFFSET.installedApp;

		if (deviceStore.isIOSSafari) return CHAT_FORM_BOTTOM_OFFSET.iosSafari;

		return CHAT_FORM_BOTTOM_OFFSET.phoneBrowser;
	});
	// how far the form floats in from the screen's sides on a phone (its wrapper's
	// left and right below); the expanded sheet insets its content by as much
	let chatFormSideInset = $derived(deviceStore.isStandalone ? '1rem' : '0.5rem');

	// The page is the scroller: it follows new content while the reader sits at
	// the bottom, and otherwise leaves their position alone.
	const stick = createStickToBottom();
	const showContextGauge = useShowContextGauge();
	let stickPin: HTMLDivElement | undefined = $state();
	const activeModel = useChatScreenActiveModel();
	const fileUpload = useChatScreenFileUpload({
		activeModelId: () => activeModel.activeModelId,
		capabilities: () => ({
			hasAudio: activeModel.hasAudioModality,
			hasVideo: activeModel.hasVideoModality,
			hasVision: activeModel.hasVisionModality
		})
	});
	const dragAndDrop = useChatScreenDragAndDrop({
		onDrop: fileUpload.handleFileUpload
	});
	const { handleKeydown } = useKeyboardShortcuts({
		deleteActiveConversation: () => {
			if (conversationsStore.activeConversation) {
				showDeleteDialog = true;
			}
		}
	});

	async function handleDeleteConfirm() {
		const conversation = conversationsStore.activeConversation;

		if (conversation) {
			await conversationsStore.deleteConversation(conversation.id);
		}

		showDeleteDialog = false;
	}

	async function handleSendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		// before the cleared form collapses: it fades into the conversation from the bottom
		isFormHeldAtBottom = isEmpty && uiStore.isComposerExpanded;

		try {
			return await sendMessage(message, files);
		} finally {
			// the send failed or was refused: the new chat stays, and so may the centering
			if (isEmpty) isFormHeldAtBottom = false;
		}
	}

	async function sendMessage(message: string, files?: ChatUploadedFile[]): Promise<boolean> {
		const plainFiles = files ? $state.snapshot(files) : undefined;
		const result = plainFiles
			? await parseFilesToMessageExtras(plainFiles, activeModel.activeModelId ?? undefined)
			: undefined;

		if (result?.emptyFiles && result.emptyFiles.length > 0) {
			emptyFileNames = result.emptyFiles;
			showEmptyFileDialog = true;

			if (files) {
				const emptyFileNamesSet = new Set(result.emptyFiles);

				fileUpload.uploadedFiles = fileUpload.uploadedFiles.filter(
					(file) => !emptyFileNamesSet.has(file.name)
				);
			}

			return false;
		}

		// Sending re-arms following: the reply streams in under the message.
		stick.scrollToBottom();

		await chatStore.sendMessage(message, result?.extras);

		return true;
	}

	let lastScrolledConversationId: string | null = null;

	// Every navigation lands anew: this screen outlives its routes, so coming
	// back to the conversation it last landed on (via the new-chat screen)
	// must not count as already landed. The landing also owns the scroll
	// position, so SvelteKit's own reset (or restore, on back/forward) would
	// only move the previous conversation, still on screen, to its top.
	onNavigate((navigation) => {
		lastScrolledConversationId = null;

		if (navigation.to?.params?.id) disableScrollHandling();
	});

	// Lands at the bottom of a conversation the first time its messages render.
	// The resize observer keeps it there while the list settles (lazy rows,
	// syntax highlighting, images) without polling for a stable height.
	async function handleMessagesReady(messageCount: number) {
		if (messageCount === 0) return;

		const id = conversationsStore.activeConversation?.id ?? null;

		if (!id || id === lastScrolledConversationId) return;

		lastScrolledConversationId = id;
		await tick();
		stick.scrollToBottom();
	}

	/**
	 * Moves the form between its centered and bottom positions: it fades out
	 * where it is, drifting toward where it goes, and fades
	 * back in from the other side once it is there. Reduced motion keeps only
	 * the fade. Re-reads `isEmpty` after each move, so a flip back mid-move
	 * ends where the screen is. A phone keeps the form at the bottom either way, so there it
	 * only switches, at once.
	 */
	async function moveForm() {
		if (isFormMoving) return;

		if (deviceStore.isMobile) {
			isFormCentered = isEmpty;
			isFormHeldAtBottom = false;

			return;
		}

		isFormMoving = true;

		while (formWrapper && isFormCentered !== isEmpty) {
			const toCenter = isEmpty;
			const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
			const shift = reduceMotion ? 0 : (toCenter ? -1 : 1) * CHAT_FORM_MOVE_SHIFT_PX;
			const shown = { opacity: 1, transform: 'translateY(0)' };
			// `transform` moves the form on top of the offset that centers it
			const leave = formWrapper.animate(
				[
					shown,
					{
						offset: CHAT_FORM_MOVE_OUT_FADE_END,
						opacity: 0,
						transform: `translateY(${shift * CHAT_FORM_MOVE_OUT_FADE_END}px)`
					},
					{ opacity: 0, transform: `translateY(${shift}px)` }
				],
				{
					duration: toCenter ? CHAT_FORM_MOVE_OUT_MS : CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS,
					easing: CHAT_FORM_MOVE_OUT_EASING,
					fill: 'forwards'
				}
			);

			await leave.finished.catch(() => {});

			isFormCentered = toCenter;
			await tick();
			// lay the new position out while the bottom transition is off, so it does not slide
			formWrapper?.getBoundingClientRect();

			// cancelled in the same frame the entry starts, so it never flashes
			leave.cancel();
			formWrapper?.animate([{ opacity: 0, transform: `translateY(${-shift}px)` }, shown], {
				duration: toCenter ? CHAT_FORM_MOVE_IN_MS : CHAT_FORM_MOVE_TO_BOTTOM_IN_MS,
				easing: CHAT_FORM_MOVE_IN_EASING
			});
		}

		isFormCentered = isEmpty;
		isFormMoving = false;
		isFormHeldAtBottom = false;
	}

	$effect(() => {
		if (isEmpty !== isFormCentered) untrack(() => void moveForm());
	});

	function handleErrorDialogOpenChange(open: boolean) {
		if (!open) {
			chatStore.dismissErrorDialog();
		}
	}

	async function handleSystemPromptAdd(draft: { message: string; files: ChatUploadedFile[] }) {
		await chatStore.addSystemPrompt(draft);
	}

	$effect(() => {
		stick.setEnabled(!settingsStore.config.disableAutoScroll);
	});

	$effect(() => {
		stick.attach(document.documentElement, document.body, stickPin);

		return () => stick.detach();
	});

	// the form growing (the editor expanding, more lines) keeps the latest message
	// in view above it while the reader sits at the bottom
	$effect(() => {
		if (formWrapper) return stick.followSizeOf(formWrapper);
	});
</script>

{#if dragAndDrop.isDragOver}
	<ChatScreenDragOverlay />
{/if}

<svelte:window onkeydown={handleKeydown} onscroll={() => stick.handleScroll()} />

{#if isServerLoading}
	<ServerLoadingSplash />
{:else}
	<div
		style:--chat-form-bottom-position={chatFormBottomPosition}
		style:--chat-form-side-inset={chatFormSideInset}
		class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem)] md:min-h-[calc(100dvh-var(--chat-tabs-offset,0px))] px-4 md:py-0 pt-(--page-top) pb-48 md:pb-4"
		ondragenter={dragAndDrop.dragHandlers.dragenter}
		ondragleave={dragAndDrop.dragHandlers.dragleave}
		ondragover={dragAndDrop.dragHandlers.dragover}
		ondrop={dragAndDrop.dragHandlers.drop}
		role="main"
	>
		<!-- a phone pins the gauge to the top right, across from the menu toggle; like the toggle,
		     it steps aside while the expanded chat form covers the screen and its collapse key -->
		{#if deviceStore.isMobile && showContextGauge.current && !uiStore.isComposerExpanded}
			<div transition:fade={{ duration: CHAT_FORM_CHROME_FADE_MS }}>
				<ChatFormContextGauge
					class="sidebar-reveal-pin-top fixed top-(--corner-control-top) right-4 z-20 size-10 rounded-full bg-muted/60 border border-border/20 shadow-sm backdrop-blur-sm"
					corner
				/>
			</div>
		{/if}

		<!-- a phone's chat shows its title and working directory between the corner keys; a new
		     chat has neither yet, and offers the directory in the + menu instead. Like the keys,
		     it steps aside while the expanded chat form covers the screen -->
		{#if deviceStore.isMobile && page.params.id && !uiStore.isComposerExpanded}
			<div transition:fade={{ duration: CHAT_FORM_CHROME_FADE_MS }}>
				<ChatScreenMobileHeader
					onWorkingDirectoryClick={() => chatScreenForm?.openWorkingDirectoryPicker()}
				/>
			</div>
		{/if}

		{#if !isEmpty}
			<ChatMessages
				messages={conversationsStore.activeMessages}
				onMessagesReady={handleMessagesReady}
			/>
		{/if}

		<!-- the scroll anchor that keeps the page at its bottom, see createStickToBottom -->
		<div bind:this={stickPin} aria-hidden="true" class="stick-to-bottom-pin"></div>

		<div
			bind:this={formWrapper}
			style:--chat-form-resize-ease={isFormLifted ? CHAT_FORM_EASE_OUT : undefined}
			style:padding-top="var(--chat-form-padding-top)"
			class={[
				// moves between the centered and bottom positions with a fade (see moveForm), at
				// the same height in the page, so the messages never shift
				'pointer-events-none md:sticky fixed mt-auto max-md:z-10',
				// a phone holds it in place on screen while the page is slid aside
				deviceStore.isMobile && 'sidebar-reveal-pin-bottom',
				// on a phone the expanded form is a sheet: flush with the screen's sides and bottom
				uiStore.isComposerExpanded
					? 'bottom-0 right-0 left-0'
					: deviceStore.isStandalone
						? 'bottom-6 right-4 left-4'
						: deviceStore.isIOSSafari
							? 'bottom-2 left-2 right-2'
							: 'bottom-3 right-2 left-2',
				// Lifted on a new chat until the resting card's top edge is on the screen's middle,
				// by its sticky bottom offset and not a translate, which a quick reversal starts a
				// frame ahead of the card's height
				isFormLifted && 'md:[--chat-form-drop:calc(50dvh-1rem-var(--chat-form-resting-height))]',
				isFormLifted && !uiStore.isComposerExpanded
					? 'md:bottom-[calc(var(--chat-form-drop)+1rem)]'
					: 'md:bottom-4',
				!isFormMoving &&
					'transition-[left,right,bottom] duration-400 ease-[var(--chat-form-resize-ease,cubic-bezier(0.3,1.12,0.5,1))] max-md:transition-none motion-reduce:transition-none'
			]}
		>
			<ChatScreenGreeting
				faded={uiStore.isComposerExpanded}
				instant={deviceStore.isMobile}
				isEmpty={isFormLifted}
			/>

			<ChatScreenServerError />

			{#if page.params.id}
				<ChatScreenStreamResumeStatus />
			{/if}

			<div class="pointer-events-none flex flex-col gap-6 items-center w-full">
				{#if stick.farFromBottom && page.url.hash.includes(ROUTES.CHAT) && page.params.id}
					<ChatScreenActionScrollDown onclick={() => stick.scrollToBottom('smooth')} />
				{/if}
			</div>

			<ChatScreenForm
				bind:this={chatScreenForm}
				bind:uploadedFiles={fileUpload.uploadedFiles}
				class="pointer-events-auto conversation-chat-form"
				disabled={hasPropsError || chatStore.isEditing()}
				isLoading={isCurrentConversationLoading}
				onFileRemove={fileUpload.handleFileRemove}
				onFileUpload={fileUpload.handleFileUpload}
				onSend={handleSendMessage}
				onStop={() => chatStore.stopGeneration()}
				onSystemPromptAdd={handleSystemPromptAdd}
			/>
		</div>
	</div>
{/if}

<ChatScreenDialogsAndAlerts
	{activeErrorDialog}
	{emptyFileNames}
	{fileUpload}
	{handleDeleteConfirm}
	{handleErrorDialogOpenChange}
	{showDeleteDialog}
	{showEmptyFileDialog}
/>
