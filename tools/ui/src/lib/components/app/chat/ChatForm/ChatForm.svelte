<script lang="ts">
	import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
	import { Maximize, Minimize } from '@lucide/svelte';
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
	import { hasMarkdownStructure } from '$lib/components/app/content/MarkdownEditor/markdown-structure';
	import { Button } from '$lib/components/ui/button';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import {
		CHAT_FORM_EXPAND_MIN_LINES,
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		ICON_CLASS_DEFAULT,
		INITIAL_FILE_SIZE,
		INPUT_CLASSES,
		PROMPT_CONTENT_SEPARATOR,
		SETTING_CONFIG_DEFAULT
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType,
		ToolSource
	} from '$lib/enums';
	import { useChatFormPickers } from '$lib/hooks/use-chat-form-pickers.svelte';
	import {
		chatStore,
		conversationsStore,
		mcpStore,
		modelsStore,
		serverStore,
		settingsStore,
		toolsStore,
		uiStore
	} from '$lib/stores';
	import type {
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '$lib/types';
	import {
		buildMentionInsertion,
		findCommandToken,
		findMentionToken,
		getConversationModel,
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
	import { onDestroy, onMount, untrack } from 'svelte';
	import type { ClassValue } from 'svelte/elements';

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
		isLoading?: boolean;
		placeholder?: string;
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
		uploadedFiles = $bindable([]),
		value = $bindable('')
	}: Props = $props();

	// Component References

	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormInputFileInputInvisible | undefined = $state(undefined);
	let pickersRef: { handleKeydown: (event: KeyboardEvent) => boolean } | undefined =
		$state(undefined);
	let inputRef: ChatFormInput | undefined = $state(undefined);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Invisible anchor at the form's top edge so the mention/WD popovers
	// float above the box.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	let cwd = $derived(
		conversationsStore.activeConversation?.cwd ?? conversationsStore.preferences.pendingCwd
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
		setCaretOffset: (offset) => inputRef?.setCaretOffset(offset),
		setValue: (v) => {
			value = v;
			onValueChange?.(v);
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

	// Expanded editor: entered once the message holds a markdown block or several lines,
	// or by hand from the maximize button. Kept while the message has text (deleting a
	// line does not make it jump back) and left once it is cleared or sent. Collapsing by
	// hand holds until then too.
	let isExpanded = $state(false);
	let isCollapsedByUser = $state(false);
	let hadText = false;

	$effect(() => {
		if (!expandable) return;

		const text = value ?? '';

		if (!text.trim()) {
			if (hadText) {
				isExpanded = false;
				isCollapsedByUser = false;
			}

			hadText = false;

			return;
		}

		hadText = true;

		if (untrack(() => isExpanded || isCollapsedByUser)) return;

		isExpanded =
			text.split('\n').length >= CHAT_FORM_EXPAND_MIN_LINES || hasMarkdownStructure(text);
	});

	$effect(() => {
		if (expandable) uiStore.isComposerExpanded = isExpanded;
	});

	onDestroy(() => {
		if (expandable) uiStore.isComposerExpanded = false;
	});

	function toggleExpanded() {
		isExpanded = !isExpanded;
		// collapsing an empty form leaves the next message free to expand it again
		isCollapsedByUser = !isExpanded && !!(value ?? '').trim();
		inputRef?.focus();
	}

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	export function focus() {
		inputRef?.focus();
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
		// the enter-to-submit logic below and keep the key from the editor.
		if (pickers.handleKeydown(event)) {
			event.preventDefault();

			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			const isModifier = event.ctrlKey || event.metaKey;
			const sendOnEnter = currentConfig.sendOnEnter !== false;

			// Caret inside a fenced code block (closed, or still open
			// while being typed): Enter adds a line, never submits -
			// skipping submit lets the editor's own newline through.
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

		value = built.newValue;
		onValueChange?.(built.newValue);
		inputRef?.setCaretOffset(built.caretOffset);
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
	class={[
		// one flex column in both states, so expanding or collapsing changes only
		// sizes, which ease, never the layout
		'relative flex flex-col',
		// Expanded: the whole form (card plus the chips below it) never outgrows the room
		// under the conversation tabs and the form's top padding, so the page itself never
		// scrolls; past that the card shrinks to fit and the editor scrolls inside
		isExpanded &&
			'max-h-[calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-2rem)]',
		className
	]}
	onsubmit={(event) => {
		event.preventDefault();

		if (!canSubmit || disabled || hasLoadingAttachments) return;

		onSubmit?.();
	}}
>
	<ChatFormPickers
		bind:this={pickersRef}
		commandQuery={pickers.commandQuery}
		commands={pickers.availableCommands}
		isCommandPickerOpen={pickers.isCommandPickerOpen}
		isMentionPickerOpen={pickers.isMentionPickerOpen}
		isPromptPickerOpen={pickers.isPromptPickerOpen}
		{mentionAnchor}
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

	<div
		bind:this={mentionAnchor}
		aria-hidden="true"
		class="pointer-events-none absolute top-0 right-0 left-0 h-px"
	></div>

	<div
		class={[
			INPUT_CLASSES,
			'group/chat-form relative flex shrink flex-col overflow-hidden rounded-4xl backdrop-blur-md transition-[min-height] duration-400 ease-[cubic-bezier(0.3,1.12,0.5,1)] md:rounded-3xl',
			// Expanded, it opens to about half the screen (or as the page sets it, see
			// ChatScreen) and grows from there with the text. Collapsed, the minimum is its
			// one-line height, so expanding starts from where the card already is
			isExpanded
				? 'min-h-[min(var(--chat-form-expanded-min-height,50dvh),calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-5rem))]'
				: 'min-h-0 md:min-h-(--chat-form-collapsed-height)',
			disabled && 'cursor-not-allowed opacity-60'
		]}
		data-slot="input-area"
	>
		<!-- Shown on hover (always on touch screens) while collapsed, always while expanded -->
		{#if expandable}
			<Tooltip.Root>
				<Tooltip.Trigger
					class={[
						'absolute top-2.5 right-2.5 z-10 transition-opacity duration-150',
						!isExpanded &&
							'opacity-0 group-hover/chat-form:opacity-100 focus-within:opacity-100 [@media(hover:none)]:opacity-100'
					]}
				>
					<Button
						class="h-8 w-8 rounded-full p-0 text-muted-foreground"
						onclick={toggleExpanded}
						type="button"
						variant="ghost"
					>
						<span class="sr-only">{isExpanded ? 'Collapse editor' : 'Expand editor'}</span>

						{#if isExpanded}
							<Minimize class={ICON_CLASS_DEFAULT} />
						{:else}
							<Maximize class={ICON_CLASS_DEFAULT} />
						{/if}
					</Button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>{isExpanded ? 'Collapse editor' : 'Expand editor'}</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		<ChatAttachmentsList
			bind:uploadedFiles
			activeModelId={activeModelId ?? undefined}
			{attachments}
			class="py-5"
			limitToSingleRow
			onFileRemove={handleFileRemove}
			style="scroll-padding: 1rem;"
		/>

		<div
			class={[
				'relative flex min-h-12 flex-1 flex-col rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!'
			]}
		>
			<ChatFormInput
				bind:this={inputRef}
				bind:value
				class={['px-5 py-1.5 md:pt-0', expandable && 'pr-12']}
				{disabled}
				expanded={isExpanded}
				onInput={() => {
					pickers.handleInput();
					onValueChange?.(value);
				}}
				onKeydown={handleKeydown}
				onPaste={handlePaste}
				{placeholder}
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

			<ChatFormActions
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				class="mt-auto px-3"
				{disabled}
				{isLoading}
				isReasoning={chatStore.isReasoning}
				{isRecording}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ files: uploadedFiles, message: value })}
				{showAddButton}
				{showModelSelector}
				{uploadedFiles}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if conversationsStore.preferences.hasEnabledCwdTools()}
		<ChatFormCurrentWorkingDirectory
			bind:query={pickers.workingDirectoryQuery}
			class="self-start"
			customAnchor={mentionAnchor}
			directory={cwd}
			{disabled}
			isOpen={pickers.isWorkingDirectoryPickerOpen}
			onChange={handleWorkingDirectoryChange}
			onClose={pickers.handleWorkingDirectoryClose}
			onOpen={pickers.handleWorkingDirectoryOpen}
		/>
	{/if}
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
