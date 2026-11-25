<script lang="ts">
	import ContextGaugeDrawer from './ChatFormContextGauge/ContextGaugeDrawer.svelte';
	import { microphoneUnavailableReason, warmUpVoiceCapture } from '@ggml-org/llama-voice';
	import { VoiceRecorder } from '@ggml-org/llama-voice/svelte';
	import { Maximize, Minimize } from '@lucide/svelte';
	import {
		ActionIcon,
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormActionsRecording,
		ChatFormCurrentWorkingDirectory,
		ChatFormInput,
		ChatFormInputFileInputInvisible,
		ChatFormMcpResourcesList,
		ChatFormPickers,
		ChatFormSystemMessageAttachment,
		ChatFormSystemMessageChip,
		DialogMcpResourcesBrowser
	} from '#lib/components/app/index.ts';
	import {
		CHAT_FORM_SINGLE_ROW_MAX_TEXT_LINES,
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		ICON_CLASS_DEFAULT,
		INITIAL_FILE_SIZE,
		OPEN_OVERLAY_SELECTOR,
		PROMPT_CONTENT_SEPARATOR,
		SETTING_CONFIG_DEFAULT,
		SYSTEM_MESSAGE_PLACEHOLDER,
		VOICE_EDITOR_SELECTOR
	} from '#lib/constants/index.ts';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MessageRole,
		MimeTypeText,
		SpecialFileType,
		ToolSource
	} from '#lib/enums/index.ts';
	import { useChatFormExpansion } from '#lib/hooks/use-chat-form-expansion.svelte.ts';
	import { useChatFormPickers } from '#lib/hooks/use-chat-form-pickers.svelte.ts';
	import {
		useChatFormVoice,
		type VoiceRecorderLike
	} from '#lib/hooks/use-chat-form-voice.svelte.ts';
	import {
		chatStore,
		conversationsStore,
		deviceStore,
		mcpStore,
		modelsStore,
		serverStore,
		settingsStore,
		toolsStore,
		uiStore,
		voiceTranscriptsStore
	} from '#lib/stores/index.ts';
	import type {
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '#lib/types/index.ts';
	import {
		buildMentionInsertion,
		findCommandToken,
		findMentionToken,
		getConversationModel,
		isIMEComposing,
		parseClipboardContent,
		uuid
	} from '#lib/utils/index.ts';
	import { onMount, tick, untrack } from 'svelte';
	import type { ClassValue } from 'svelte/elements';
	import { toast } from 'svelte-sonner';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: ClassValue;
		disabled?: boolean;
		/** Grow into a large editor once the message holds markdown or several lines. */
		expandable?: boolean;
		/** On a desktop, keep a message of one line on the actions' row, in a pill. */
		singleRow?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showAddButton?: boolean;
		showModelSelector?: boolean;
		/** Writing the system message: the form takes the system message's dashed, muted look. */
		systemMode?: boolean;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onValueChange?: (value: string) => void;

		/** Records voice input; tests and stories pass a fake, the app a VoiceRecorder. */
		voiceRecorder?: VoiceRecorderLike;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		expandable = false,
		isLoading = false,
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onUploadedFilesChange,
		onValueChange,
		placeholder = 'Type a message...',
		showAddButton = true,
		showModelSelector = true,
		singleRow = false,
		systemMode = false,
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		voiceRecorder
	}: Props = $props();

	// Component References

	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormInputFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let inputRef: ChatFormInput | undefined = $state(undefined);

	// Voice input
	const recorder: VoiceRecorderLike = untrack(() => voiceRecorder) ?? new VoiceRecorder();
	/** Why this page can't record at all (no HTTPS, no recorder), which no tap fixes. */
	let micUnavailableReason = $state<string | null>(null);
	/** Why the last start failed (denied, no microphone, busy); the next tap tries again. */
	let micError = $state<string | null>(null);

	// Invisible anchor the pickers' popovers float above: the form's top edge, or in the
	// expanded editor the line being written
	let pickerAnchor: HTMLDivElement | null = $state(null);
	let pickerAnchorBox = $state({ height: 1, top: 0 });

	let cwd = $derived(
		conversationsStore.activeConversation?.cwd ?? conversationsStore.preferences.pendingCwd
	);

	let systemMessage = $derived(
		conversationsStore.activeMessages.find((m) => m.role === MessageRole.SYSTEM)
	);
	// A new chat shows the system message written for it as an attachment, until the first
	// message sends it. A conversation with one offers it in a chip under the form.
	// A placeholder is one nobody has written yet
	let systemMessageAttachment = $derived.by(() => {
		if (conversationsStore.activeConversation) return null;

		const content = chatStore.pendingSystemMessage;

		return content && content.trim() && content !== SYSTEM_MESSAGE_PLACEHOLDER ? content : null;
	});
	let systemMessageChip = $derived(
		conversationsStore.activeConversation ? systemMessage?.content : undefined
	);

	let showWorkingDirectory = $derived(conversationsStore.preferences.hasEnabledCwdTools());
	// On a desktop the chip under the form shows a directory once one is set, and the + menu
	// sets one until then. A phone has no chip: a chat shows it in its header (see
	// ChatScreenMobileHeader), and a new chat, which has no header yet, in the + menu
	let showWorkingDirectoryChip = $derived(showWorkingDirectory && !deviceStore.isMobile && !!cwd);
	let isWorkingDirectoryInAddMenu = $derived(
		showWorkingDirectory && (deviceStore.isMobile ? !conversationsStore.activeConversation : !cwd)
	);

	const pickers = useChatFormPickers({
		canCompact: () => chatStore.canCompact(conversationsStore.activeMessages),
		focusInput: refocusInput,
		getCaretOffset: () => inputRef?.getCaretOffset(),
		getCwd: () => cwd,
		getPickersRef: () => pickersRef,
		getServerHome: () => toolsStore.serverHome ?? null,
		getShowModelSelector: () => showModelSelector,
		getValue: () => value,
		hasCwdTools: () => conversationsStore.preferences.hasEnabledCwdTools(),
		// policy-aware, same rule as the agentic flow: MCP category on and at
		// least one globally-enabled server whose group key is not disabled
		hasPrompts: () => {
			const prefs = conversationsStore.preferences;

			if (!prefs.isCategoryEnabled(ToolSource.MCP)) return false;

			return mcpStore
				.getServers()
				.some((s) => s.enabled && prefs.isServerToolsEnabled(s.id) && s.url.trim());
		},
		onCompact: () => chatStore.compactConversation(),
		openModelSelector: () => chatFormActionsRef?.openModelSelector(),
		// TODO: no typed pickers with an on-screen keyboard for now: each mobile browser moves
		// the page its own way as it opens, so a list by the form can not stay clear of the
		// text. A drawer with its own search field (as the directory picker is) could.
		opensOnInput: () => !deviceStore.isTouch,
		setCaretOffset: (offset) => inputRef?.setCaretOffset(offset),
		setValue: (v) => {
			value = v;
			onValueChange?.(v);
		}
	});

	const voice = useChatFormVoice({
		addVoiceFile: (file) => {
			uploadedFiles = [...uploadedFiles, file];
			onUploadedFilesChange?.(uploadedFiles);
		},
		getCaretOffset: () => inputRef?.getCaretOffset() ?? value.length,
		getValue: () => value,
		onMicError: (message) => (micError = message),
		recorder,
		// on a phone placing the caret would open the keyboard
		setCaretOffset: (offset) => {
			if (!deviceStore.isMobile) inputRef?.setCaretOffset(offset);
		},
		setValue: (next) => {
			value = next;
			onValueChange?.(next);
		}
	});

	async function handleWorkingDirectoryChange(newDir: string | null) {
		// Committing a directory consumes the `/cwd` token; the chip's
		// clear-X path has no token to consume.
		const token = findCommandToken(value);

		if (token && token.name === 'cwd') {
			value = '';
			onValueChange?.('');
		}

		await conversationsStore.preferences.setCwd(newDir);

		if (conversationsStore.activeConversation) {
			await chatStore.recordCwdChange(newDir?.trim() || null);
		}
	}

	// Resource Dialog State
	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	// MCP Servers Dialog State

	let currentConfig = $derived(settingsStore.config);

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);

		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	let isRouter = $derived(serverStore.isRouterMode);
	let conversationModel = $derived(
		getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);
	let activeModelId = $derived(modelsStore.activeModelId);

	let hasModelSelected = $derived(
		!isRouter || !!conversationModel || !!modelsStore.selectedModelId
	);
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);
	// while a voice message records, what is being said shows in the empty message box; once
	// it stops, the text goes with the clip, as its transcript
	let voicePlaceholder = $derived(
		voice.isActive && voice.liveCaption && !value.trim() ? voice.liveCaption : null
	);

	// while a reply streams, sending what is recorded waits for its first words in the form
	let hasVoiceText = $derived(!!value.trim() || !!voice.liveCaption);

	const EXPANDED_SHEET_FILL = 'max-md:bg-(--chat-form-sheet-fill)';

	let cardEl: HTMLDivElement | undefined = $state();
	let actionsEl: HTMLDivElement | undefined = $state();

	const expansion = useChatFormExpansion({
		// on a phone this leaves focus where it is: focusing would open the keyboard
		focusInput: () => inputRef?.focus(),
		getActions: () => actionsEl,
		getCard: () => cardEl,
		getContent: () => inputRef?.getContentDOM(),
		getEditorBox: () => inputRef?.getBox(),
		// the conversation the message is for: its draft arrives without a remount
		getResetKey: () => conversationsStore.activeConversation?.id,
		getValue: () => value,
		isEnabled: () => expandable
	});
	let isExpanded = $derived(expansion.isExpanded);

	// One line of text shares the actions' row; a second line, or one that wraps beside the
	// actions, takes the two rows. It stays there until the text is shorter than where it
	// wrapped, as the wider row of its own would fit it again and flip it straight back
	let textHeight = $state<number | null>(null);
	let wrappedAtLength = Infinity;
	let lastLength = 0;
	let canBeSingleRow = $derived(
		singleRow &&
			!deviceStore.isMobile &&
			!isExpanded &&
			!voice.isActive &&
			!hasAttachments &&
			!systemMessageAttachment &&
			!mcpStore.resources.hasAttachments &&
			!value.includes('\n')
	);
	// starts as it will settle, so a phone never draws a frame of the single row
	let isSingleRow = $state(untrack(() => canBeSingleRow));

	// Re-runs after its own switch too, and measures the text as it is laid out then: the
	// height the input last reported can be the other layout's
	$effect(() => {
		void textHeight;

		const content = inputRef?.getContentDOM();
		const lineHeight = content ? parseFloat(getComputedStyle(content).lineHeight) : 0;
		const lines = content && lineHeight ? content.offsetHeight / lineHeight : 0;
		const fitsOneLine = lines < CHAT_FORM_SINGLE_ROW_MAX_TEXT_LINES;
		const length = value.length;

		// an emptied form tries the one row again; one that wraps while empty keeps the two rows
		if (!length && lastLength) wrappedAtLength = Infinity;

		lastLength = length;

		if (!canBeSingleRow) {
			isSingleRow = false;
		} else if (isSingleRow && !fitsOneLine) {
			isSingleRow = false;
			wrappedAtLength = length;
		} else if (!isSingleRow && length < wrappedAtLength) {
			isSingleRow = true;
		}
	});
	let isPickerOpen = $derived(
		pickers.isCommandPickerOpen ||
			pickers.isMentionPickerOpen ||
			pickers.isPromptPickerOpen ||
			pickers.isWorkingDirectoryPickerOpen
	);

	// On a phone the sheet's pickers float by its actions, which ride above the keyboard
	let activePickerAnchor = $derived(
		isExpanded && deviceStore.isMobile ? (actionsEl ?? null) : pickerAnchor
	);

	// before the popover is drawn, and again as the query is typed and the line moves
	$effect.pre(() => {
		if (!isPickerOpen) return;

		void value;

		const line = isExpanded && !deviceStore.isMobile ? inputRef?.getCaretRect() : null;
		const formTop = pickerAnchor?.parentElement?.getBoundingClientRect().top ?? 0;

		pickerAnchorBox = line
			? { height: line.bottom - line.top, top: line.top - formTop }
			: { height: 1, top: 0 };
	});
	// shown on hover (always on touch screens) while collapsed, always while expanded: on
	// the phone sheet it keeps the collapsed card's place, below the status bar
	let expandToggleClass = $derived(
		[
			'absolute top-2.5 right-2.5 z-10 h-8 w-8 rounded-full text-muted-foreground transition-opacity duration-150',
			isExpanded
				? 'max-md:top-safe-2.5 max-md:right-[calc(0.625rem+var(--chat-form-side-inset,0.5rem))]'
				: 'opacity-0 group-hover/chat-form:opacity-100 focus-visible:opacity-100 [@media(hover:none)]:opacity-100'
		].join(' ')
	);

	onMount(() => {
		micUnavailableReason = microphoneUnavailableReason();
	});

	/** Expands for the message as it stands, e.g. one put back after a send failed. */
	export function expandForMessage() {
		expansion.handleInput();
	}

	export function focus() {
		inputRef?.focus();
	}

	export function openWorkingDirectoryPicker() {
		pickers.handleWorkingDirectoryOpen();
	}

	export function openModelSelector() {
		chatFormActionsRef?.openModelSelector();
	}

	export function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			chatFormActionsRef?.openModelSelector();

			return false;
		}

		return true;
	}

	function handleFileSelect(files: File[]) {
		onFilesAdd?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handleFileRemove(fileId: string) {
		if (fileId.startsWith('attachment-')) {
			const index = parseInt(fileId.replace('attachment-', ''), 10);

			if (!isNaN(index) && index >= 0 && index < attachments.length) {
				onAttachmentRemove?.(index);
			}
		} else {
			const removed = uploadedFiles.find((file) => file.id === fileId);

			if (removed?.voice) voiceTranscriptsStore.forget(removed.name);

			onUploadedFileRemove?.(fileId);
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		// Pickers consume navigation/escape keys first; when consumed, skip
		// the enter-to-submit logic below and keep the key from the editor.
		if (pickers.handleKeydown(event)) {
			event.preventDefault();

			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			// Enter sends and Ctrl/Cmd+Enter adds a line, the other way round when sending
			// on Enter is off or the keyboard is on screen, whose Return has no Shift;
			// Shift+Enter always adds one. Skipping submit lets the editor's newline through
			const sendOnEnter = currentConfig.sendOnEnter !== false && !deviceStore.isTouch;

			if (isModifier === sendOnEnter) return;

			event.preventDefault();

			void submit();
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFilesAdd?.(files);

			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith(CLIPBOARD_CONTENT_QUOTE_PREFIX)) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0 || parsed.mcpPromptAttachments.length > 0) {
				event.preventDefault();
				value = parsed.message;
				onValueChange?.(parsed.message);

				// Handle text attachments as files
				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);

					onFilesAdd?.(attachmentFiles);
				}

				// Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
				if (parsed.mcpPromptAttachments.length > 0) {
					const mcpPromptFiles: ChatUploadedFile[] = parsed.mcpPromptAttachments.map((att) => ({
						file: new File([att.content], `${att.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						id: uuid(),
						isLoading: false,
						mcpPrompt: {
							arguments: att.arguments,
							promptName: att.promptName,
							serverName: att.serverName
						},
						name: att.name,
						size: att.content.length,
						textContent: att.content,
						type: SpecialFileType.MCP_PROMPT
					}));

					uploadedFiles = [...uploadedFiles, ...mcpPromptFiles];
					onUploadedFilesChange?.(uploadedFiles);
				}

				setTimeout(() => {
					inputRef?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFilesAdd?.([textFile]);
		}
	}

	function handlePromptLoadStart(
		placeholderId: string,
		promptInfo: MCPPromptInfo,
		args?: Record<string, string>
	) {
		pickers.closePromptPicker();

		const promptName = promptInfo.title || promptInfo.name;
		const placeholder: ChatUploadedFile = {
			file: new File([], 'loading'),
			id: placeholderId,
			isLoading: true,
			mcpPrompt: {
				arguments: args ? { ...args } : undefined,
				promptName: promptInfo.name,
				serverName: promptInfo.serverName
			},
			name: promptName,
			size: INITIAL_FILE_SIZE,
			type: SpecialFileType.MCP_PROMPT
		};

		uploadedFiles = [...uploadedFiles, placeholder];
		onUploadedFilesChange?.(uploadedFiles);
		inputRef?.focus();
	}

	function handlePromptLoadComplete(placeholderId: string, result: GetPromptResult) {
		const promptText = result.messages
			?.map((msg: PromptMessage) => {
				if (typeof msg.content === 'string') {
					return msg.content;
				}

				if (msg.content.type === ContentPartType.TEXT) {
					return msg.content.text;
				}

				return '';
			})
			.filter(Boolean)
			.join(PROMPT_CONTENT_SEPARATOR);

		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId
				? {
						...f,
						file: new File([promptText], `${f.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						isLoading: false,
						size: promptText.length,
						textContent: promptText
					}
				: f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptLoadError(placeholderId: string, error: string) {
		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId ? { ...f, isLoading: false, loadError: error } : f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	// Deferred so the closing popover's focus scope tears down first -
	// bits-ui yanks a synchronous focus() back into the still-mounted popover.
	function refocusInput() {
		queueMicrotask(() => inputRef?.focus());
	}

	// A flow outside the chat, e.g. the models page returning to it, asks for the
	// composer back. Focus on a frame of its own, once the route has settled.
	$effect(() => {
		if (!uiStore.composerFocusRequested) return;

		uiStore.composerFocusRequested = false;

		setTimeout(() => inputRef?.focus(), 0);
	});

	// Splice the mention link in place of the `@<query>` token. Uses the
	// live cursor, not a stale snapshot - the token may have been edited.
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = inputRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);

		if (!token) return;

		const built = buildMentionInsertion(entry, value, token);

		if (!built) return;

		// one undoable edit; the input reports the new value like typing
		inputRef?.replaceText(built.newValue, built.caretOffset);
	}

	function handleMicClick() {
		// a phone shows no tooltip on a tap, so the reason comes as a toast
		if (micUnavailableReason) {
			toast.error(micUnavailableReason);

			return;
		}

		void voice.start();
	}

	// the audio set-up takes a moment, so it is made while the message is being written;
	// it asks for no microphone, so the browser shows no recording indicator
	function handleFocusIn(event: FocusEvent) {
		if (voice.showButton && !voice.isActive) warmUpVoiceCapture();

		// the voice message's live text shown as the placeholder turns into text to edit
		if (voicePlaceholder && (event.target as Element | null)?.closest(VOICE_EDITOR_SELECTOR)) {
			voice.adoptLiveText();
		}
	}

	// a send asked for while the dictated text is still being refined waits for it
	let isSendWaiting = $state(false);

	/** Sends the message; dictated text that is still being refined is sent refined. */
	async function submit() {
		if (!canSubmit || disabled || hasLoadingAttachments || isSendWaiting) return;

		if (voice.isRefining) {
			isSendWaiting = true;

			try {
				await voice.settled();
				await tick();
			} finally {
				isSendWaiting = false;
			}

			if (!canSubmit || disabled || hasLoadingAttachments) return;
		}

		onSubmit?.();
	}

	/** Stops the recording and sends the message once its text or clip is in the form. */
	async function handleStopAndSend() {
		await voice.stop();
		await tick();
		await submit();
	}

	// Escape drops a recording, or else stops the reply, wherever the focus is. On the body, so it
	// runs before the overlays' document listeners: a tooltip closing on Escape does not take the key
	function handleBodyKeydown(event: KeyboardEvent) {
		if (event.key !== KeyboardKey.ESCAPE || event.defaultPrevented) return;

		if (document.querySelector(OPEN_OVERLAY_SELECTOR)) return;

		if (voice.isActive) {
			event.preventDefault();
			voice.cancel();

			return;
		}

		// the system message editor and a message edit take Escape in their own window listeners
		if (!isLoading || systemMode || chatStore.isEditing() || isIMEComposing(event)) return;

		event.preventDefault();
		onStop?.();
	}
</script>

<svelte:body onkeydown={handleBodyKeydown} />

{#snippet systemMessageLeading()}
	{#if systemMessageAttachment}
		<ChatFormSystemMessageAttachment content={systemMessageAttachment} {disabled} />
	{/if}
{/snippet}

<ChatFormInputFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class={[
		// one flex column in both states, so expanding changes only sizes, which ease
		'relative flex flex-col',
		// expanded, the form never outgrows the room under the tabs (the whole screen on a
		// phone): the page never scrolls, the editor does
		isExpanded &&
			'max-h-[calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-2rem)] max-md:max-h-dvh',
		className
	]}
	onfocusin={handleFocusIn}
	onsubmit={(event) => {
		event.preventDefault();
		void submit();
	}}
>
	<!-- the pickers' own triggers fill this box, so all of them share the anchor -->
	<div
		bind:this={pickerAnchor}
		style:height="{pickerAnchorBox.height}px"
		style:top="{pickerAnchorBox.top}px"
		class="pointer-events-none absolute right-0 left-0"
	>
		<ChatFormPickers
			bind:this={pickersRef}
			anchor={activePickerAnchor}
			commandQuery={pickers.commandQuery}
			commands={pickers.availableCommands}
			isCommandPickerOpen={pickers.isCommandPickerOpen}
			isMentionPickerOpen={pickers.isMentionPickerOpen}
			isPromptPickerOpen={pickers.isPromptPickerOpen}
			mentionQuery={pickers.mentionQuery}
			onCommandPickerClose={pickers.handleCommandPickerClose}
			onCommandSelect={pickers.handleCommandSelect}
			onMentionOpened={() => inputRef?.focus()}
			onMentionPickerClose={pickers.handleMentionPickerClose}
			onMentionSelect={handleMentionSelect}
			onPromptLoadComplete={handlePromptLoadComplete}
			onPromptLoadError={handlePromptLoadError}
			onPromptLoadStart={handlePromptLoadStart}
			onPromptPickerClose={pickers.handlePromptPickerClose}
			promptSearchQuery={pickers.promptSearchQuery}
			scopePath={pickers.mentionScopePath}
		/>
	</div>

	<div
		bind:this={cardEl}
		class={[
			'chat-form-surface group/chat-form relative flex shrink flex-col overflow-hidden outline-none transition-[min-height,padding,border-radius,border-color,background-color] duration-400 ease-[var(--chat-form-resize-ease,cubic-bezier(0.3,1.12,0.5,1))] motion-reduce:transition-none',
			isSingleRow ? 'rounded-(--chat-form-single-row-radius)' : 'rounded-4xl md:rounded-3xl',
			// on a phone the sheet's animation moves it (see useChatFormExpansion): only its
			// border, hidden on the sheet, fades back in after
			'max-md:transition-[border-color] max-md:duration-200',
			// Expanded, it opens to about half the screen, or on a new chat (where the page
			// sets how far the form drops) to its collapsed height plus that drop, so its top
			// edge stays put. The collapsed height is held at the card's real one while it moves
			'[--chat-form-expanded-min-height:calc(var(--chat-form-drop)+var(--chat-form-collapsed-height))]',
			isExpanded
				? 'min-h-[min(var(--chat-form-expanded-min-height,50dvh),calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-5rem))]'
				: ['min-h-0', expandable && !isSingleRow && 'md:min-h-(--chat-form-collapsed-height)'],
			// the phone sheet's fill is opaque, so Safari's toolbar strip can match it and it
			// needs no backdrop blur, the costliest thing to repaint
			(isExpanded || expansion.isBlurHeld) && 'max-md:backdrop-blur-none',
			isExpanded && [
				// its actions keep the collapsed form's gap from the screen's bottom, or ride
				// 0.5rem above the keyboard (unset, the second term is 0)
				'max-md:min-h-dvh max-md:rounded-none max-md:pb-[max(var(--chat-form-bottom-position,0px),calc(var(--keyboard-inset,-0.5rem)+0.5rem))]',
				// clear of the status bar, which an installed app draws over the page
				'max-md:pt-safe',
				// no rim, ring or visible border along the screen's edges; the border stays
				// transparent so nothing inside shifts by its width
				'max-md:[--glass-edge:transparent] max-md:[--glass-edge-focus:transparent] max-md:[--glass-highlight:transparent]',
				'max-md:border-transparent max-md:focus-within:border-transparent max-md:dark:border-transparent max-md:dark:focus-within:border-transparent',
				EXPANDED_SHEET_FILL
			],
			systemMode && '!border-2 !border-dashed !border-border/50 !bg-muted',
			disabled && 'cursor-not-allowed opacity-60'
		]}
		data-single-row={isSingleRow || undefined}
		data-slot="input-area"
	>
		<!-- a single row has no room for it: a message expands once it has a second line -->
		{#if expandable && !isSingleRow}
			<ActionIcon
				class={expandToggleClass}
				icon={isExpanded ? Minimize : Maximize}
				iconSize={ICON_CLASS_DEFAULT}
				onclick={expansion.toggle}
				tooltip={isExpanded ? 'Collapse editor' : 'Expand editor'}
			/>
		{/if}

		<ChatAttachmentsList
			bind:uploadedFiles
			activeModelId={activeModelId ?? undefined}
			{attachments}
			class={isExpanded ? 'py-5 max-md:px-(--chat-form-side-inset)' : 'py-5'}
			leading={systemMessageAttachment ? systemMessageLeading : undefined}
			limitToSingleRow
			onFileRemove={handleFileRemove}
			style="scroll-padding: 1rem;"
		/>

		<div
			class={[
				'relative flex flex-1 shadow-(--elevation-control) transition-shadow focus-within:shadow-(--elevation-surface)',
				// the add button, the text and the rest of the actions, or the text above them
				isSingleRow
					? 'items-center gap-2 rounded-(--chat-form-single-row-radius) px-3 py-2.5'
					: 'min-h-12 flex-col rounded-4xl py-2 pb-2.25 md:rounded-3xl md:py-3!',
				// the phone sheet is wider than the collapsed card by the page's side inset: its
				// content is inset by as much, so nothing reflows while it moves
				isExpanded && 'max-md:px-(--chat-form-side-inset)',
				// the sheet meets the screen's edges: no corners, and no shadow along them
				isExpanded && 'max-md:rounded-none max-md:shadow-none max-md:focus-within:shadow-none'
			]}
		>
			<ChatFormInput
				bind:contentHeight={textHeight}
				bind:this={inputRef}
				bind:value
				class={isSingleRow ? 'pr-1' : ['px-5 py-1.5 md:pt-0', expandable && 'pr-12']}
				compact={isSingleRow}
				{disabled}
				expanded={isExpanded}
				onInput={() => {
					voice.handleUserInput();
					expansion.handleInput();
					pickers.handleInput();
					onValueChange?.(value);
				}}
				onKeydown={handleKeydown}
				onPaste={handlePaste}
				placeholder={voicePlaceholder ?? placeholder}
			/>

			{#if mcpStore.resources.hasAttachments}
				<ChatFormMcpResourcesList
					class="mb-3"
					onResourceClick={(uri) => {
						preSelectedResourceUri = uri;
						isResourceDialogOpen = true;
					}}
				/>
			{/if}

			<!-- its own box, so the phone sheet can hold it by the bottom edge while it moves -->
			<div bind:this={actionsEl} class={isSingleRow ? 'contents' : 'relative mt-auto'}>
				{#if voice.isActive}
					<ChatFormActionsRecording
						class="px-3"
						elapsedMs={voice.elapsedMs}
						levels={voice.levels}
						liveCaption={voicePlaceholder ? '' : voice.liveCaption}
						mode={voice.mode}
						onCancel={voice.cancel}
						onSend={isLoading && !hasVoiceText ? undefined : () => void handleStopAndSend()}
						onStop={() => void voice.stop()}
						quiet={voice.quiet}
						state={voice.state}
					/>
				{/if}

				<!-- kept while recording, so its selector and menus keep their state -->
				<div class={voice.isActive ? 'hidden' : 'contents'}>
					<ChatFormActions
						bind:this={chatFormActionsRef}
						canSend={canSubmit}
						class={isSingleRow ? undefined : 'px-3'}
						{disabled}
						inline={isSingleRow}
						{isLoading}
						isReasoning={chatStore.isReasoning}
						{isSendWaiting}
						isVoiceRefining={voice.isRefining}
						micWarning={micUnavailableReason ?? micError}
						onFileUpload={handleFileUpload}
						onMicClick={handleMicClick}
						{onStop}
						onSystemPromptClick={() =>
							onSystemPromptClick?.({ files: uploadedFiles, message: value })}
						onWorkingDirectoryClick={isWorkingDirectoryInAddMenu
							? pickers.handleWorkingDirectoryOpen
							: undefined}
						{showAddButton}
						{showModelSelector}
						voice={{
							action: voice.label,
							hint: voice.hint,
							prepare: voice.prepare,
							show: voice.showButton,
							unavailableReason: voice.unavailableReason,
							unprepare: voice.unprepare
						}}
						workingDirectory={cwd}
					/>
				</div>
			</div>
		</div>
	</div>

	{#if deviceStore.isMobile}
		<ContextGaugeDrawer />
	{/if}

	<!-- iOS Safari tints its bottom toolbar from a solid fixed element at the screen's bottom
	     edge: this strip carries the sheet's fill while it rests -->
	{#if isExpanded && !expansion.isSheetMoving && deviceStore.isIOSSafari}
		<div
			aria-hidden="true"
			class={[
				'pointer-events-none fixed inset-x-0 bottom-0 -z-10 h-3 md:hidden',
				EXPANDED_SHEET_FILL
			]}
		></div>
	{/if}

	<div class="flex flex-wrap items-center gap-x-1 self-start">
		{#if showWorkingDirectory}
			<!-- opened from the chip, the + menu, a phone chat's header or `/cwd` -->
			<ChatFormCurrentWorkingDirectory
				bind:query={pickers.workingDirectoryQuery}
				class="self-start"
				customAnchor={activePickerAnchor}
				directory={cwd}
				{disabled}
				isOpen={pickers.isWorkingDirectoryPickerOpen}
				onChange={handleWorkingDirectoryChange}
				onClose={pickers.handleWorkingDirectoryClose}
				onOpen={pickers.handleWorkingDirectoryOpen}
				showChip={showWorkingDirectoryChip}
			/>
		{/if}

		{#if systemMessageChip}
			<ChatFormSystemMessageChip content={systemMessageChip} {disabled} />
		{/if}
	</div>
</form>

<DialogMcpResourcesBrowser
	bind:open={isResourceDialogOpen}
	onAttach={(resource: MCPResourceInfo) => {
		mcpStore.attachResource(resource.uri);
	}}
	onOpenChange={(newOpen: boolean) => {
		if (!newOpen) {
			preSelectedResourceUri = undefined;
		}
	}}
	preSelectedUri={preSelectedResourceUri}
/>
