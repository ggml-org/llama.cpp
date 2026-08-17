<script lang="ts">
	import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
	import { goto } from '$app/navigation';
	import {
		ChatAttachmentsList,
		ChatFormActions,
		ChatFormCurrentWorkingDirectory,
		ChatFormInput,
		ChatFormInputFileInputInvisible,
		ChatFormMcpResourcesList,
		ChatFormPickers,
		DialogMcpResourcesBrowser
	} from '$lib/components/app';
	import {
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		INITIAL_FILE_SIZE,
		INPUT_CLASSES,
		PROMPT_CONTENT_SEPARATOR,
		ROUTES,
		SETTING_CONFIG_DEFAULT
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType
	} from '$lib/enums';
	import { useChatFormPickers } from '$lib/hooks/use-chat-form-pickers.svelte';
	import { dispatchSkillActivation } from '$lib/services/skill-command.service';
	import {
		chatStore,
		conversationsStore,
		mcpResourceStore,
		mcpStore,
		modelsStore,
		serverStore,
		settingsStore,
		toolsStore
	} from '$lib/stores';
	import { skillAvailabilityStore } from '$lib/stores/skill-availability.svelte';
	import { skillsStore } from '$lib/stores/skills.svelte';
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
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showMcpPromptButton?: boolean;
		showAddButton?: boolean;
		showModelSelector?: boolean;

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
		showMcpPromptButton = false,
		showModelSelector = true,
		uploadedFiles = $bindable([]),
		value = $bindable('')
	}: Props = $props();

	// Shared handle for the textarea and rich input renderers.
	type ChatInputHandle = {
		focus(): void;
		resetHeight(): void;
		getElement(): HTMLElement | undefined;
		getCaretOffset(): number;
		setCaretOffset(offset: number): void;
	};

	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormInputFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let inputRef: ChatInputHandle | undefined = $state(undefined);

	// Use rich input for file mentions and code spans.
	let useRichInput = $state(false);

	let isRecording = $state(false);
	let recordingSupported = $state(false);

	let mentionAnchor: HTMLDivElement | null = $state(null);

	let cwd = $derived(conversationsStore.activeConversation?.cwd ?? conversationsStore.pendingCwd);

	// Suggest enabled skills from the ready catalog for the active CWD.
	const skillCatalogSlot = $derived(skillsStore.slotFor(cwd ?? undefined));
	const skillSuggestions = $derived(
		skillCatalogSlot?.status === 'ready'
			? (skillCatalogSlot.catalog?.skills ?? []).filter(
					(entry) => !skillAvailabilityStore.isDisabled(entry.id)
				)
			: []
	);

	const pickers = useChatFormPickers({
		dispatchSkillsCommand: handleSkillsCommand,
		focusInput: refocusInput,
		getCaretOffset: () => inputRef?.getCaretOffset(),
		getCwd: () => cwd,
		getPickersRef: () => pickersRef,
		getServerHome: () => toolsStore.serverHome ?? null,
		getShowModelSelector: () => showModelSelector,
		getValue: () => value,
		hasCwdTools: () => toolsStore.hasEnabledCwdTools,
		hasPrompts: () => mcpStore.hasPromptsCapability(conversationsStore.getAllMcpServerOverrides()),
		hasSkills: () => skillsStore.slotFor(cwd ?? undefined)?.status !== 'error',
		openModelSelector: () => chatFormActionsRef?.openModelSelector(),
		setCaretOffset: (offset) => inputRef?.setCaretOffset(offset),
		setValue: (v) => {
			value = v;
			onValueChange?.(v);
		}
	});

	// Dispatch /skills: open the catalog without args, activate a named skill.
	function handleSkillsCommand(args: string): void {
		if (!args) {
			void goto(ROUTES.SKILLS);

			return;
		}

		void dispatchSkillActivation(args).then((outcome) => {
			if (!outcome.ok) {
				if (outcome.reason === 'not-found') {
					toast.error(`Skill "${args}" was not found`);
				} else if (outcome.reason === 'disabled') {
					// Disabled skills never wake the agent.
					toast.error(`Skill "${args}" is disabled`);
				} else if (outcome.reason === 'persistence-failed') {
					toast.error(`Skill "${args}" could not be saved`);
				} else {
					toast.error('Skills are unavailable on this server');
				}

				return;
			}

			if (!outcome.created) {
				toast.info(`Skill "${args}" is already activated in this conversation`);
			} else {
				toast.success(`Skill "${args}" activated`);
			}

			void chatStore.runTurnFromLeaf();
		});
	}

	async function handleWorkingDirectoryChange(newDir: string | null) {
		// Only a committed /cwd token is consumed.
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

	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	let currentConfig = $derived(settingsStore.config);

	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);

		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	let isRouter = $derived(serverStore.isRouterMode);
	let conversationModel = $derived(
		chatStore.getConversationModel(conversationsStore.activeMessages as DatabaseMessage[])
	);
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

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);

			if (model) return model.model;
		}

		return null;
	});

	let hasModelSelected = $derived(
		!isRouter || !!conversationModel || !!modelsStore.selectedModelId
	);
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	// Pin caret offsets before renderer swaps; otherwise the swap effect overwrites them.
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
		const wantRichInput = containsFileMentionLink(value ?? '') || containsCodeSpan(value ?? '');

		if (useRichInput === wantRichInput) return;

		if (!caretOffsetPinned) {
			pendingCaretOffset = inputRef?.getCaretOffset() ?? (value ?? '').length;
		}

		useRichInput = wantRichInput;
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
		// Let pickers handle navigation keys before submit logic.
		if (pickers.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

			// Enter inside a code block inserts a newline instead of submitting.
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

				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);

					onFilesAdd?.(attachmentFiles);
				}

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

	// Wait for the popover focus scope to unmount before refocusing.
	function refocusInput() {
		queueMicrotask(() => inputRef?.focus());
	}

	// Replace the live mention token, not a stale query snapshot.
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = inputRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);

		if (!token) return;

		const built = buildMentionInsertion(entry, value, token);

		if (!built) return;

		// Pin the inserted caret before the promotion effect runs.
		pendingCaretOffset = built.caretOffset;
		caretOffsetPinned = true;

		value = built.newValue;
		onValueChange?.(built.newValue);

		// Restore the caret directly when already using rich input.
		if (useRichInput) {
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

<ChatFormInputFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

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
		isSkillPickerOpen={pickers.isSkillPickerOpen}
		skillQuery={pickers.skillQuery}
		skills={skillSuggestions}
		onSkillPickerClose={pickers.handleSkillPickerClose}
		onSkillSelect={pickers.handleSkillSelect}
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
			<ChatFormInput
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
				{useRichInput}
			/>

			{#if mcpResourceStore.hasAttachments}
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
				onSystemPromptClick={() => onSystemPromptClick?.({ files: uploadedFiles, message: value })}
				onMcpPromptClick={showMcpPromptButton ? () => pickers.openPromptPicker() : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if toolsStore.hasEnabledCwdTools}
		<ChatFormCurrentWorkingDirectory
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
