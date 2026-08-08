<script lang="ts">
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormContenteditable,
		ChatFormFileInputInvisible,
		ChatFormMcpResourcesList,
		ChatFormPickers,
		ChatFormTextarea,
		ChatFormWorkingDirectory,
		DialogMcpResourcesBrowser
	} from '$lib/components/app';
	import {
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		INPUT_CLASSES,
		SETTING_CONFIG_DEFAULT,
		INITIAL_FILE_SIZE,
		PROMPT_CONTENT_SEPARATOR
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType
	} from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
	import { modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpHasResourceAttachments } from '$lib/stores/mcp-resources.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import {
		conversationsStore,
		activeMessages,
		activeConversation,
		pendingCwd
	} from '$lib/stores/conversations.svelte';
	import type {
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '$lib/types';
	import {
		buildMentionInsertion,
		containsCodeSpan,
		containsFileMentionLink,
		findCommandToken,
		findMentionToken,
		isIMEComposing,
		isOffsetInCodeBlock,
		parseClipboardContent,
		uuid
	} from '$lib/utils';
	import { useChatFormPickers } from '$lib/hooks/use-chat-form-pickers.svelte';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showMcpPromptButton?: boolean;
		showAddButton?: boolean;
		showModelSelector?: boolean;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onValueChange?: (value: string) => void;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		placeholder = 'Type a message...',
		showMcpPromptButton = false,
		showAddButton = true,
		showModelSelector = true,
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onUploadedFilesChange,
		onValueChange
	}: Props = $props();

	// Component References
	// Shared handle of the two input renderers (textarea + contenteditable).
	type ChatInputHandle = {
		focus(): void;
		resetHeight(): void;
		getElement(): HTMLElement | undefined;
		getCaretOffset(): number;
		setCaretOffset(offset: number): void;
	};

	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let inputRef: ChatInputHandle | undefined = $state(undefined);

	// Render-mode gate: the plain textarea by default, the contenteditable
	// while the buffer carries a `file://` mention link or a complete code
	// span (badges and code chips need a DOM the textarea cannot provide).
	// Demotes back once neither remains.
	let useContenteditable = $state(false);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Invisible anchor at the form's top edge so the mention/WD popovers
	// float above the box.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	let cwd = $derived(activeConversation()?.cwd ?? pendingCwd());

	const pickers = useChatFormPickers({
		getValue: () => value,
		setValue: (v) => {
			value = v;
			onValueChange?.(v);
		},
		getCaretOffset: () => inputRef?.getCaretOffset(),
		setCaretOffset: (offset) => inputRef?.setCaretOffset(offset),
		focusInput: refocusInput,
		getShowModelSelector: () => showModelSelector,
		hasPrompts: () => mcpStore.hasPromptsCapability(conversationsStore.getAllMcpServerOverrides()),
		hasCwdTools: () => toolsStore.hasEnabledCwdTools,
		getCwd: () => cwd,
		getServerHome: () => toolsStore.serverHome ?? null,
		openModelSelector: () => chatFormActionsRef?.openModelSelector(),
		getPickersRef: () => pickersRef
	});

	async function handleWorkingDirectoryChange(newDir: string | null) {
		// Committing a directory consumes the `/cwd` token; the chip's
		// clear-X path has no token to consume.
		const token = findCommandToken(value);
		if (token && token.name === 'cwd') {
			value = '';
			onValueChange?.('');
		}
		await conversationsStore.setCwd(newDir);
		if (conversationsStore.activeConversation) {
			await chatStore.recordCwdChange(newDir?.trim() || null);
		}
	}

	// Resource Dialog State
	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	let currentConfig = $derived(config());

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);
		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	let isRouter = $derived(isRouterMode());
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);
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

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	// Caret offset restored after a renderer swap. Callers that mutate
	// `value` themselves (e.g. the mention picker) pin the target offset
	// BEFORE the assignment; otherwise the swap effect snapshots the
	// current caret.
	let pendingCaretOffset = 0;
	let caretOffsetPinned = false;

	function queueCaretRestore() {
		queueMicrotask(() => {
			inputRef?.focus();
			inputRef?.setCaretOffset(pendingCaretOffset);
			caretOffsetPinned = false;
		});
	}

	$effect(() => {
		const wantContenteditable =
			containsFileMentionLink(value ?? '') || containsCodeSpan(value ?? '');
		if (useContenteditable === wantContenteditable) return;

		if (!caretOffsetPinned) {
			pendingCaretOffset = inputRef?.getCaretOffset() ?? (value ?? '').length;
		}

		useContenteditable = wantContenteditable;
		queueCaretRestore();
	});

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	export function focus() {
		inputRef?.focus();
	}

	export function resetTextareaHeight() {
		inputRef?.resetHeight();
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
			onUploadedFileRemove?.(fileId);
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		// Pickers consume navigation/escape keys first; when consumed, skip
		// the enter-to-submit logic below.
		if (pickers.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

			// Caret inside a fenced code block (closed, or still open
			// while being typed): Enter adds a line, never submits. The
			// contenteditable consumes this case locally; this gate
			// covers the plain textarea, where skipping submit lets the
			// native newline through.
			if (!isModifier && isOffsetInCodeBlock(value ?? '', inputRef?.getCaretOffset() ?? 0)) {
				return;
			}

			if (sendOnEnter || isModifier) {
				event.preventDefault();

				if (!canSubmit || disabled || hasLoadingAttachments) return;

				onSubmit?.();
			}
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
						id: uuid(),
						name: att.name,
						size: att.content.length,
						type: SpecialFileType.MCP_PROMPT,
						file: new File([att.content], `${att.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						isLoading: false,
						textContent: att.content,
						mcpPrompt: {
							serverName: att.serverName,
							promptName: att.promptName,
							arguments: att.arguments
						}
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
			id: placeholderId,
			name: promptName,
			size: INITIAL_FILE_SIZE,
			type: SpecialFileType.MCP_PROMPT,
			file: new File([], 'loading'),
			isLoading: true,
			mcpPrompt: {
				serverName: promptInfo.serverName,
				promptName: promptInfo.name,
				arguments: args ? { ...args } : undefined
			}
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
						isLoading: false,
						textContent: promptText,
						size: promptText.length,
						file: new File([promptText], `${f.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						})
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

	// Splice the mention link in place of the `@<query>` token. Uses the
	// live cursor, not a stale snapshot - the token may have been edited.
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = inputRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);
		if (!token) return;

		const built = buildMentionInsertion(entry, value, token);
		if (!built) return;

		// Pin the post-insertion caret BEFORE the swap effect runs;
		// otherwise the effect clobbers it with the textarea's selection
		// at promotion time (browser-dependent: usually reset to 0).
		pendingCaretOffset = built.caretOffset;
		caretOffsetPinned = true;

		value = built.newValue;
		onValueChange?.(built.newValue);

		// Already in contenteditable mode: no renderer flip, so the swap
		// effect's caret restore never runs.
		if (useContenteditable) {
			queueCaretRestore();
		}
	}

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');
			return;
		}

		if (isRecording) {
			isRecording = false;
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFilesAdd?.([audioFile]);
			} catch (error) {
				console.error('Failed to stop recording:', error);
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}
</script>

<ChatFormFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class="relative grid {className}"
	onsubmit={(event) => {
		event.preventDefault();

		if (!canSubmit || disabled || hasLoadingAttachments) return;

		onSubmit?.();
	}}
>
	<ChatFormPickers
		bind:this={pickersRef}
		isCommandPickerOpen={pickers.isCommandPickerOpen}
		commandQuery={pickers.commandQuery}
		commands={pickers.availableCommands}
		onCommandPickerClose={pickers.handleCommandPickerClose}
		onCommandSelect={pickers.handleCommandSelect}
		isPromptPickerOpen={pickers.isPromptPickerOpen}
		promptSearchQuery={pickers.promptSearchQuery}
		isMentionPickerOpen={pickers.isMentionPickerOpen}
		mentionQuery={pickers.mentionQuery}
		{mentionAnchor}
		scopePath={pickers.mentionScopePath}
		onPromptPickerClose={pickers.handlePromptPickerClose}
		onMentionPickerClose={pickers.handleMentionPickerClose}
		onMentionOpened={() => inputRef?.focus()}
		onMentionSelect={handleMentionSelect}
		onPromptLoadStart={handlePromptLoadStart}
		onPromptLoadComplete={handlePromptLoadComplete}
		onPromptLoadError={handlePromptLoadError}
	/>

	<div
		bind:this={mentionAnchor}
		class="pointer-events-none absolute top-0 right-0 left-0 h-px"
		aria-hidden="true"
	></div>

	<div
		class="{INPUT_CLASSES} overflow-hidden rounded-4xl md:rounded-3xl backdrop-blur-md {disabled
			? 'cursor-not-allowed opacity-60'
			: ''}"
		data-slot="input-area"
	>
		<ChatAttachmentsList
			{attachments}
			bind:uploadedFiles
			onFileRemove={handleFileRemove}
			limitToSingleRow
			class="py-5"
			style="scroll-padding: 1rem;"
			activeModelId={activeModelId ?? undefined}
		/>

		<div
			class="flex-column relative min-h-12 items-center rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!"
		>
			{#if useContenteditable}
				<ChatFormContenteditable
					class="px-5 py-1.5 md:pt-0 mb-0.5"
					bind:this={inputRef}
					bind:value
					onKeydown={handleKeydown}
					onInput={() => {
						pickers.handleInput();
						onValueChange?.(value);
					}}
					onPaste={handlePaste}
					{disabled}
					{placeholder}
				/>
			{:else}
				<ChatFormTextarea
					class="px-5 py-1.5 md:pt-0"
					bind:this={inputRef}
					bind:value
					onKeydown={handleKeydown}
					onInput={() => {
						pickers.handleInput();
						onValueChange?.(value);
					}}
					onPaste={handlePaste}
					{disabled}
					{placeholder}
				/>
			{/if}

			{#if mcpHasResourceAttachments()}
				<ChatFormMcpResourcesList
					class="mb-3"
					onResourceClick={(uri) => {
						preSelectedResourceUri = uri;
						isResourceDialogOpen = true;
					}}
				/>
			{/if}

			<ChatFormActions
				class="px-3"
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				{disabled}
				{isLoading}
				isReasoning={chatStore.isReasoning}
				{isRecording}
				{showAddButton}
				{showModelSelector}
				{uploadedFiles}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ message: value, files: uploadedFiles })}
				onMcpPromptClick={showMcpPromptButton ? () => pickers.openPromptPicker() : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if toolsStore.hasEnabledCwdTools}
		<ChatFormWorkingDirectory
			directory={cwd}
			isOpen={pickers.isWorkingDirectoryPickerOpen}
			bind:query={pickers.workingDirectoryQuery}
			customAnchor={mentionAnchor}
			onChange={handleWorkingDirectoryChange}
			onClose={pickers.handleWorkingDirectoryClose}
			onOpen={pickers.handleWorkingDirectoryOpen}
			{disabled}
		/>
	{/if}
</form>

<DialogMcpResourcesBrowser
	bind:open={isResourceDialogOpen}
	preSelectedUri={preSelectedResourceUri}
	onAttach={(resource: MCPResourceInfo) => {
		mcpStore.attachResource(resource.uri);
	}}
	onOpenChange={(newOpen: boolean) => {
		if (!newOpen) {
			preSelectedResourceUri = undefined;
		}
	}}
/>
