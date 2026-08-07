<script lang="ts">
	import {
		ChatAttachmentsList,
		ChatFormActions,
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
		PROMPT_CONTENT_SEPARATOR,
		PROMPT_TRIGGER_PREFIX
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
		findMentionToken,
		isIMEComposing,
		mentionLinkEndingAt,
		parseClipboardContent,
		takeMentionDismissSnapshot,
		type MentionDismissSnapshot,
		uuid
	} from '$lib/utils';
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
	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Invisible anchor at the form's top edge so the mention popover floats above the box.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	// Picker State
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');

	// Last dismissed `@`-mention token; while intact the picker does not
	// reopen, so an escaped `@<query>` stays literal until edited.
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;

	let cwd = $derived(activeConversation()?.cwd ?? pendingCwd());

	async function handleWorkingDirectoryChange(value: string | null) {
		await conversationsStore.setCwd(value);
		if (conversationsStore.activeConversation) {
			await chatStore.recordCwdChange(value?.trim() || null);
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

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	// Defer so the closing popover's focus scope tears down first - bits-ui
	// yanks a synchronous focus() back into the still-mounted popover.
	function refocusInput() {
		queueMicrotask(() => textareaRef?.focus());
	}

	export function focus() {
		textareaRef?.focus();
	}

	export function resetTextareaHeight() {
		textareaRef?.resetHeight();
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

	function handleInput() {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
		const hasServers = mcpStore.hasEnabledServers(perChatOverrides);
		const cursor = textareaRef?.getCaretOffset() ?? value.length;
		const mentionToken = findMentionToken(value, cursor);

		// A `@` mention takes precedence; typing one switches from any other open picker.
		if (mentionToken && mentionToken.query.length > 0) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';

			const isDismissedSticky =
				mentionDismissedSnapshot !== null &&
				mentionDismissedSnapshot.start === mentionToken.start &&
				mentionDismissedSnapshot.query === mentionToken.query;

			if (!isDismissedSticky) {
				mentionDismissedSnapshot = null;
				isMentionPickerOpen = true;
				mentionQuery = mentionToken.query;
				return;
			}

			isMentionPickerOpen = false;
			mentionQuery = '';
			return;
		}

		isMentionPickerOpen = false;
		mentionQuery = '';
		// Token gone or changed: reset the snapshot so a fresh `@` reopens.
		if (mentionDismissedSnapshot !== null && !mentionToken) {
			mentionDismissedSnapshot = null;
		}

		if (value.startsWith(PROMPT_TRIGGER_PREFIX) && hasServers) {
			isPromptPickerOpen = true;
			promptSearchQuery = value.slice(1);
		} else {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (pickersRef?.handleKeydown(event)) {
			return;
		}

		// Backspace at a mention link's end deletes the whole token at once.
		if (event.key === KeyboardKey.BACKSPACE && !event.ctrlKey && !event.metaKey && !event.altKey) {
			const el = textareaRef?.getElement();
			if (el instanceof HTMLTextAreaElement && el.selectionStart === el.selectionEnd) {
				const link = mentionLinkEndingAt(value, el.selectionStart);
				if (link) {
					event.preventDefault();
					value = value.slice(0, link.start) + value.slice(link.end);
					onValueChange?.(value);
					queueMicrotask(() => textareaRef?.setCaretOffset(link.start));
					return;
				}
			}
		}

		if (event.key === KeyboardKey.ESCAPE && isPromptPickerOpen) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
			return;
		}

		if (event.key === KeyboardKey.ESCAPE && isMentionPickerOpen) {
			isMentionPickerOpen = false;
			mentionQuery = '';
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

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
					textareaRef?.focus();
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
		// Only clear the value if the prompt was triggered by typing '/'
		if (value.startsWith(PROMPT_TRIGGER_PREFIX)) {
			value = '';
			onValueChange?.('');
		}
		isPromptPickerOpen = false;
		promptSearchQuery = '';

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
		textareaRef?.focus();
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

	function handlePromptPickerClose() {
		isPromptPickerOpen = false;
		promptSearchQuery = '';
		textareaRef?.focus();
	}

	function handleMentionPickerClose() {
		if (isMentionPickerOpen) {
			const cursor = textareaRef?.getCaretOffset() ?? value.length;
			mentionDismissedSnapshot = takeMentionDismissSnapshot(value, cursor);
		}
		isMentionPickerOpen = false;
		mentionQuery = '';
		refocusInput();
	}

	// Splice the `[name](file:///<abs path>)` link in place of the `@<query>`
	// token, restoring the caret after the bindable value settles.
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = textareaRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);
		if (!token) return;

		const built = buildMentionInsertion(entry, value, token);
		if (!built) return;

		value = built.newValue;
		onValueChange?.(built.newValue);

		queueMicrotask(() => {
			textareaRef?.focus();
			textareaRef?.setCaretOffset(built.caretOffset);
		});
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
		{isPromptPickerOpen}
		{promptSearchQuery}
		{isMentionPickerOpen}
		{mentionQuery}
		{mentionAnchor}
		scopePath={cwd}
		onPromptPickerClose={handlePromptPickerClose}
		onMentionPickerClose={handleMentionPickerClose}
		onMentionOpened={() => textareaRef?.focus()}
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
			onpaste={handlePaste}
		>
			<ChatFormTextarea
				class="px-5 py-1.5 md:pt-0"
				bind:this={textareaRef}
				bind:value
				onKeydown={handleKeydown}
				onInput={() => {
					handleInput();
					onValueChange?.(value);
				}}
				{disabled}
				{placeholder}
			/>

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
				onMcpPromptClick={showMcpPromptButton ? () => (isPromptPickerOpen = true) : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if toolsStore.builtinTools.length > 0}
		<ChatFormWorkingDirectory
			directory={cwd}
			onChange={handleWorkingDirectoryChange}
			onClose={refocusInput}
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
