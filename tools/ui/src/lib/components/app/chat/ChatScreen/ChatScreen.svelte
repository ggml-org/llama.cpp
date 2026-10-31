<script lang="ts">
	import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
	import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
	import ChatScreenGreeting from './ChatScreenGreeting.svelte';
	import { disableScrollHandling, onNavigate } from '$app/navigation';
	import { page } from '$app/state';
	import {
		ChatMessages,
		ChatScreenDragOverlay,
		ChatScreenForm,
		ChatScreenServerError,
		ChatScreenStreamResumeStatus,
		ServerLoadingSplash
	} from '$lib/components/app';
	import {
		CHAT_FORM_MOVE_IN_EASING,
		CHAT_FORM_MOVE_IN_MS,
		CHAT_FORM_MOVE_OUT_EASING,
		CHAT_FORM_MOVE_OUT_MS,
		CHAT_FORM_MOVE_SHIFT_PX,
		CHAT_FORM_MOVE_TO_BOTTOM_IN_MS,
		CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS,
		CHAT_FORM_RESIZE_MS,
		CHAT_GREETING_ENTER_DELAY_MS,
		ROUTES
	} from '$lib/constants';
	import { useChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
	import { useChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
	import { useChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
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

	let { showCenteredEmpty = false } = $props();

	let emptyFileNames = $state<string[]>([]);
	let showDeleteDialog = $state(false);
	let showEmptyFileDialog = $state(false);
	let isEmpty = $derived(
		showCenteredEmpty && conversationsStore.activeMessages.length === 0 && !chatStore.isLoading
	);
	// where the form is drawn: follows `isEmpty` once the form has faded out
	let isFormCentered = $state(untrack(() => isEmpty));

	// Collapsing the expanded form brings the greeting back: it waits for the form to
	// shrink back first instead of fading in over it
	let greetingEnterDelayMs = $state(CHAT_GREETING_ENTER_DELAY_MS);
	let wasComposerExpanded = untrack(() => uiStore.isComposerExpanded);

	$effect(() => {
		const expanded = uiStore.isComposerExpanded;

		if (expanded === wasComposerExpanded) return;

		wasComposerExpanded = expanded;
		greetingEnterDelayMs = expanded ? CHAT_GREETING_ENTER_DELAY_MS : CHAT_FORM_RESIZE_MS;

		const timer = setTimeout(
			() => (greetingEnterDelayMs = CHAT_GREETING_ENTER_DELAY_MS),
			CHAT_FORM_RESIZE_MS * 2
		);

		return () => clearTimeout(timer);
	});
	// While the form moves between its centered and bottom positions it fades (see
	// moveForm); otherwise its position eases, in step with the form expanding and
	// collapsing, which settles it at the bottom while expanded
	let isFormMoving = $state(false);
	let isFormHeldAtBottom = $state(false);
	let formWrapper: HTMLElement | undefined = $state();
	let activeErrorDialog = $derived(chatStore.errorDialogState);
	let isServerLoading = $derived(serverStore.loading);
	let hasPropsError = $derived(!!serverStore.error);
	let isCurrentConversationLoading = $derived(chatStore.isLoading || chatStore.isStreaming());
	let chatFormBottomPosition = $derived.by(() => {
		if (!deviceStore.isMobile) return '1rem';

		if (deviceStore.isStandalone) return '1.5rem';

		if (deviceStore.isIOSSafari) return '0.25rem';

		return '0.5rem';
	});
	// how far the form floats in from the screen's sides on a phone (its wrapper's
	// left and right below); the expanded sheet insets its content by as much
	let chatFormSideInset = $derived(deviceStore.isStandalone ? '1rem' : '0.5rem');

	// The page is the scroller: it follows new content while the reader sits at
	// the bottom, and otherwise leaves their position alone.
	const stick = createStickToBottom();
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
		// called before the cleared form collapses: a send from the expanded form keeps it
		// at the bottom, where it fades into the conversation instead of heading back up
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
	 * ends where the screen is.
	 */
	async function moveForm() {
		if (isFormMoving) return;

		isFormMoving = true;

		while (formWrapper && isFormCentered !== isEmpty) {
			const toCenter = isEmpty;
			const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
			const shift = reduceMotion ? 0 : (toCenter ? -1 : 1) * CHAT_FORM_MOVE_SHIFT_PX;
			const shown = { opacity: 1, transform: 'translateY(0)' };
			// `transform` moves the form on top of the offset that centers it
			const leave = formWrapper.animate(
				[shown, { opacity: 0, transform: `translateY(${shift}px)` }],
				{
					duration: toCenter ? CHAT_FORM_MOVE_OUT_MS : CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS,
					easing: CHAT_FORM_MOVE_OUT_EASING,
					fill: 'forwards'
				}
			);

			await leave.finished.catch(() => {});

			isFormCentered = toCenter;
			await tick();

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
		if (formWrapper) return stick.followGrowthOf(formWrapper);
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
		class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem)] md:min-h-[calc(100dvh-var(--chat-tabs-offset,0px))] px-4 md:py-0 pt-12 pb-48 md:pb-4"
		ondragenter={dragAndDrop.dragHandlers.dragenter}
		ondragleave={dragAndDrop.dragHandlers.dragleave}
		ondragover={dragAndDrop.dragHandlers.dragover}
		ondrop={dragAndDrop.dragHandlers.drop}
		role="main"
	>
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
			style:padding-top="var(--chat-form-padding-top)"
			class={[
				// moves between the centered (empty chat) and bottomed positions with
				// a fade, see moveForm. Its height in the page is the same in both
				// (the padding stays, the desktop greeting floats above the form),
				// so the messages never shift while it moves
				'pointer-events-none md:sticky fixed mt-auto',
				// on a phone the expanded form is a sheet: flush with the screen's sides and bottom
				uiStore.isComposerExpanded
					? 'bottom-0 right-0 left-0'
					: deviceStore.isStandalone
						? 'bottom-6 right-4 left-4'
						: deviceStore.isIOSSafari
							? 'bottom-1 left-2 right-2'
							: 'bottom-2 right-2 left-2',
				// Centered on a new chat. Expanded, the form settles at the bottom and its card
				// grows by exactly the distance it drops (set below, see ChatForm), so its top
				// edge stays where the collapsed card's was, through the whole transition. It is
				// lifted by its sticky bottom offset (1rem pinned plus the drop), not a
				// translate: both that and the card's height are then laid out on the main
				// thread and start on the same frame, where a composited translate starts a
				// frame ahead on a quick reversal and the top edge jumps by that frame's travel
				isFormCentered && !isFormHeldAtBottom
					? [
							'md:[--chat-form-drop:calc(50dvh-7rem)] 2xl:[--chat-form-drop:calc(50dvh-4rem)]',
							// an ease-out without the overshoot: the form can not sink below where it
							// rests, nor the collapsed card shrink below its text, so an overshoot
							// would lift the top edge by its last few pixels instead
							'[--chat-form-resize-ease:cubic-bezier(0.32,0.72,0,1)]',
							uiStore.isComposerExpanded
								? 'md:bottom-4'
								: 'md:bottom-[calc(50dvh-6rem)] 2xl:bottom-[calc(50dvh-3rem)]'
						]
					: 'md:bottom-4',
				!isFormMoving &&
					'transition-[left,right,bottom] duration-400 ease-[var(--chat-form-resize-ease,cubic-bezier(0.3,1.12,0.5,1))] max-md:transition-none'
			]}
		>
			<ChatScreenGreeting
				enterDelayMs={greetingEnterDelayMs}
				faded={uiStore.isComposerExpanded}
				isEmpty={isFormCentered && !isFormHeldAtBottom}
			/>

			<ChatScreenServerError />

			{#if page.params.id}
				<ChatScreenStreamResumeStatus />
			{/if}

			<div class="pointer-events-none flex flex-col gap-6 items-center w-full">
				{#if !stick.atBottom && page.url.hash.includes(ROUTES.CHAT) && page.params.id}
					<ChatScreenActionScrollDown onclick={() => stick.scrollToBottom('smooth')} />
				{/if}
			</div>

			<ChatScreenForm
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
