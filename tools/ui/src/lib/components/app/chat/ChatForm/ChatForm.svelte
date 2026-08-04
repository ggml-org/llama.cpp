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
		PROMPT_CONTENT_SEPARATOR,
		PROMPT_TRIGGER_PREFIX,
		getChatCommands
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
	import { recentMentionsStore } from '$lib/stores/recent-mentions.svelte';
	import type {
		ChatFormCommand,
		FileMentionEntry,
		GetPromptResult,
		MCPPromptInfo,
		MCPResourceInfo,
		PromptMessage
	} from '$lib/types';
	import {
		containsFileMentionLink,
		findCommandToken,
		findMentionToken,
		isIMEComposing,
		lastPathSegment,
		parseClipboardContent,
		takeCommandDismissSnapshot,
		takeMentionDismissSnapshot,
		uuid,
		type CommandDismissSnapshot,
		type MentionDismissSnapshot
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
	// Component handle shared by both the simple textarea and the
	// contenteditable variant - both expose the same surface (focus,
	// resetHeight, getElement, getCaretOffset, setCaretOffset).
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

	// One-way promotion gate: render the simple textarea by default,
	// swap in the contenteditable once a `file://` markdown link lands
	// in the buffer. The promotion is sticky for the lifetime of the
	// composition - backspacing every file link out does NOT demote,
	// preventing the swap-thrash that comes from a textarea tearing
	// down and remounting mid-edit.
	let useContenteditable = $state(false);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Picker State
	let isCommandPickerOpen = $state(false);
	let commandQuery = $state('');
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');
	let isMentionPickerOpen = $state(false);
	let mentionQuery = $state('');
	let isWorkingDirectoryPickerOpen = $state(false);
	let workingDirectoryQuery = $state('');

	/**
	 * Snapshot of the most recent `@`-mention token the user dismissed
	 * (via Escape, outside-click, or simply by deleting it). When the
	 * picker is closed AND the same token is still intact in the buffer,
	 * we do NOT auto-reopen - the user has explicitly told us this
	 * `@<query>` should be treated as literal text. The snapshot
	 * becomes stale the moment any character inside the token changes,
	 * at which point the picker is allowed to reopen on the next input.
	 */
	let mentionDismissedSnapshot: MentionDismissSnapshot | null = null;

	/**
	 * Snapshot of the most recent `/`-command token the user dismissed
	 * (via Escape or outside-click). When the command picker is closed AND
	 * the same token is still intact in the buffer, we neither reopen the
	 * picker nor instant-dispatch - the user has explicitly told us this
	 * `/name` should be treated as literal text. The snapshot becomes stale
	 * the moment any character inside the token changes.
	 */
	let commandDismissedSnapshot: CommandDismissSnapshot | null = null;

	// Invisible anchor for the mention picker: sits at the top edge of the
	// chat form so the popover floats above the box (matches the working-
	// directory picker's `customAnchor` pattern). One anchor per popover we
	// want to anchor above the form.
	let mentionAnchor: HTMLDivElement | null = $state(null);

	let cwd = $derived(activeConversation()?.cwd ?? pendingCwd());

	// Scopes the @-mention search to the cwd the user picked (when set),
	// falling back to the server home so the picker still finds matches
	// before the user has chosen a directory.
	let mentionScopePath = $derived(cwd ?? toolsStore.serverHome ?? null);

	// Slash commands surfaced by the `/` command picker, filtered to those
	// whose backing capability is currently available.
	let availableCommands = $derived(
		getChatCommands({
			showModelSelector,
			hasPrompts: () =>
				mcpStore.hasPromptsCapability(conversationsStore.getAllMcpServerOverrides()),
			hasBuiltinTools: () => toolsStore.builtinTools.length > 0
		})
	);

	async function handleWorkingDirectoryChange(newDir: string | null) {
		// Committing a directory consumes the `/cwd` token (the command is
		// dispatched, not sent as a message). Only clear when the picker was
		// opened via `/cwd` - the chip's clear-X path has no token to consume.
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

	// Chip click opens the picker seeded with the current directory (no
	// `/cwd` token in the input, so no two-way sync happens).
	function handleWorkingDirectoryOpen() {
		workingDirectoryQuery = cwd ?? '';
		isWorkingDirectoryPickerOpen = true;
	}

	function handleWorkingDirectoryClose() {
		isWorkingDirectoryPickerOpen = false;
		workingDirectoryQuery = '';
		refocusInput();
	}

	// Two-way binding between the text after `/cwd ` and the picker's search
	// input: typing in the search input rewrites the `/cwd <query>` token in
	// the chat input. The reverse direction (typing in the chat input) is
	// handled by `handleInput` -> `dispatchCommand` -> `workingDirectoryQuery`.
	$effect(() => {
		if (!isWorkingDirectoryPickerOpen) return;
		const token = findCommandToken(value);
		if (!token || token.name !== 'cwd') return;
		const newValue = `/cwd ${workingDirectoryQuery}`;
		if (newValue === value) return;
		value = newValue;
		onValueChange?.(newValue);
		queueMicrotask(() => inputRef?.setCaretOffset(newValue.length));
	});

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

	// Caret offset restored after a renderer swap. Callers that mutate `value`
	// themselves (e.g. the mention picker splicing in `[name](file://path)`)
	// pin the target offset BEFORE the value assignment; otherwise the swap
	// effect snapshots the current caret.
	let pendingCaretOffset = 0;
	let caretOffsetPinned = false;

	// Runs after the renderer swap settles so the caret lands in the
	// newly-mounted input.
	function queueCaretRestore() {
		queueMicrotask(() => {
			inputRef?.focus();
			inputRef?.setCaretOffset(pendingCaretOffset);
			caretOffsetPinned = false;
		});
	}

	// Render-mode selector: promote to the contenteditable when the
	// value carries a `file://`-mention link, demote to the plain
	// textarea when it doesn't any longer.
	$effect(() => {
		const wantContenteditable = containsFileMentionLink(value ?? '');
		if (useContenteditable === wantContenteditable) return;

		// Pin (set by the mention picker) wins; otherwise snapshot the
		// current caret from whatever renderer is mounted.
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

	function handleInput() {
		const cursor = inputRef?.getCaretOffset() ?? value.length;

		if (value.startsWith(PROMPT_TRIGGER_PREFIX)) {
			// A `/` at the start is a command, not a mention - close the
			// mention and prompt pickers and route to the command picker.
			isMentionPickerOpen = false;
			mentionQuery = '';
			isPromptPickerOpen = false;
			promptSearchQuery = '';

			const token = findCommandToken(value);
			if (!token) {
				isCommandPickerOpen = false;
				commandQuery = '';
				return;
			}

			// Picker's been dismissed for THIS exact token - honor the
			// "literal until delete + retype" rule: don't reopen or
			// instant-dispatch until the token changes.
			const isDismissedSticky =
				commandDismissedSnapshot !== null &&
				commandDismissedSnapshot.name === token.name &&
				commandDismissedSnapshot.args === token.args;

			if (isDismissedSticky) {
				isCommandPickerOpen = false;
				commandQuery = '';
				return;
			}

			// The command name is complete once a space follows it. An exact
			// match dispatches instantly (Slack-style); a non-match falls
			// through to the picker's empty state so the user can still
			// submit the literal text. Disabled commands never dispatch.
			const nameComplete = token.args.length > 0 || value.endsWith(' ');
			if (nameComplete) {
				const command = availableCommands.find((c) => c.name === token.name);
				if (command && !command.disabled) {
					isCommandPickerOpen = false;
					commandQuery = '';
					dispatchCommand(command, token.args);
					return;
				}
			}

			// Still typing the name (or it doesn't match) - show the picker
			// only when there is something to pick.
			if (availableCommands.length > 0) {
				isCommandPickerOpen = true;
				commandQuery = token.name;
			} else {
				isCommandPickerOpen = false;
				commandQuery = '';
			}
			return;
		}

		// Not a command - close the command picker and reset the snapshot.
		isCommandPickerOpen = false;
		commandQuery = '';
		if (commandDismissedSnapshot !== null) {
			commandDismissedSnapshot = null;
		}
		// A non-command edit while the `/cwd` picker is open means the user
		// abandoned the command - close the picker.
		if (isWorkingDirectoryPickerOpen) {
			isWorkingDirectoryPickerOpen = false;
		}

		const token = findMentionToken(value, cursor);

		if (token) {
			// Picker's been dismissed for THIS exact token - honor the
			// "literal until delete + retype" rule: don't reopen until the
			// token changes (typed-then-Esc'd a slot, then kept typing
			// inside the same `@<q>`).
			const isDismissedSticky =
				mentionDismissedSnapshot !== null &&
				mentionDismissedSnapshot.start === token.start &&
				mentionDismissedSnapshot.query === token.query;

			if (!isDismissedSticky) {
				// Show the picker only if it can actually render something
				// useful: either the user has typed at least one
				// character after `@` (live search), or we've previously
				// picked at least one file/folder (recents surface). A
				// bare `@` with no recents is a no-op - re-typing into
				// the token would otherwise flash an empty "start
				// typing..." hint before the user types anything.
				const haveRecents = recentMentionsStore.value.length > 0;
				const haveQuery = token.query.length > 0;

				if (haveRecents || haveQuery) {
					mentionDismissedSnapshot = null;
					isMentionPickerOpen = true;
					mentionQuery = token.query;
					isPromptPickerOpen = false;
					promptSearchQuery = '';
					return;
				}
			}
		}

		isPromptPickerOpen = false;
		promptSearchQuery = '';
		isMentionPickerOpen = false;
		mentionQuery = '';

		// Token gone or no longer intact - the snapshot is stale. Reset so
		// the next fresh `@` opens immediately even at the same offset.
		if (mentionDismissedSnapshot !== null && !token) {
			mentionDismissedSnapshot = null;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (pickersRef?.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ESCAPE && isPromptPickerOpen) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
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

	// Refocus the chat input after a picker closes. Deferred so the
	// closing popover's focus scope tears down first - bits-ui yanks a
	// synchronous focus() back into the still-mounted popover.
	function refocusInput() {
		queueMicrotask(() => inputRef?.focus());
	}

	function handlePromptPickerClose() {
		isPromptPickerOpen = false;
		promptSearchQuery = '';
		refocusInput();
	}

	/**
	 * Dispatch a selected slash command. The command token is consumed
	 * (the input is cleared) and the corresponding picker / selector is
	 * opened. `args` (everything after the command name) seeds the target
	 * picker's search where applicable - e.g. `/prompt rev` opens the MCP
	 * prompt picker pre-filtered by `rev`.
	 */
	function dispatchCommand(command: ChatFormCommand, args: string) {
		switch (command.action) {
			case 'prompt':
				isWorkingDirectoryPickerOpen = false;
				value = '';
				onValueChange?.('');
				isPromptPickerOpen = true;
				promptSearchQuery = args.trim();
				break;
			case 'cwd':
				// Keep `/cwd <args>` in the input so the search field and the
				// token stay two-way bound while the picker is open.
				workingDirectoryQuery = args.trim();
				isWorkingDirectoryPickerOpen = true;
				break;
			case 'model':
				isWorkingDirectoryPickerOpen = false;
				value = '';
				onValueChange?.('');
				chatFormActionsRef?.openModelSelector();
				break;
		}
	}

	function handleCommandSelect(command: ChatFormCommand) {
		// Complete the command name in the input (with a trailing space) and
		// let the normal input flow dispatch it. This way `/cw` + Enter yields
		// `/cwd ` in the chat form, and the instant-dispatch-on-space path
		// opens the target picker exactly as if the user had typed it.
		value = `/${command.name} `;
		onValueChange?.(value);
		handleInput();
	}

	/**
	 * Command picker dismissed (Esc, outside-click, or selection-complete).
	 * Capture a `(name, args)` snapshot of the live token so subsequent
	 * input events that produce the SAME token won't reopen the picker or
	 * instant-dispatch - the user has explicitly told us that `/name`
	 * should be literal until they delete or retype a fresh `/`.
	 */
	function handleCommandPickerClose() {
		if (isCommandPickerOpen) {
			commandDismissedSnapshot = takeCommandDismissSnapshot(value);
		}
		isCommandPickerOpen = false;
		commandQuery = '';
		// When a command was selected, the target picker/selector takes over
		// and manages its own focus - don't yank focus back to the chat input.
		if (!isPromptPickerOpen && !isMentionPickerOpen && !isWorkingDirectoryPickerOpen) {
			refocusInput();
		}
	}

	/**
	 * Mention picker dismissed (Esc, outside-click, or selection-complete).
	 * Capture a `(start, query)` snapshot of the live token so subsequent
	 * input events that produce the SAME token won't reopen the picker -
	 * the user has explicitly told us that `@<query>` should be literal
	 * until they delete or retype a fresh `@`.
	 */
	function handleMentionPickerClose() {
		if (isMentionPickerOpen) {
			const cursor = inputRef?.getCaretOffset() ?? value.length;
			mentionDismissedSnapshot = takeMentionDismissSnapshot(value, cursor);
		}
		isMentionPickerOpen = false;
		mentionQuery = '';
		refocusInput();
	}

	/**
	 * Selection from the mention picker: splice `[name](file://<abs>)`
	 * + trailing space in place of the `@<query>` token. Cursor lands
	 * right after the trailing space so the user can keep typing
	 * naturally. Uses the live cursor position (not the stale snapshot)
	 * because the token might have been edited since we last saw it.
	 *
	 * URI shape follows RFC 8089: `file:` + `//` + absolute path. The
	 * search entry's `path` is already rooted (begins with `/`), so the
	 * prefix is `file://` not `file:///` - that yields the canonical
	 * three-slash form `file:///Users/foo/bar` without an extra `/`.
	 *
	 * Directories get a trailing `/` so the link resolves to a folder
	 * rather than being interpreted as a file with no extension.
	 */
	function handleMentionSelect(entry: FileMentionEntry) {
		const cursor = inputRef?.getCaretOffset() ?? value.length;
		const token = findMentionToken(value, cursor);
		if (!token) return;

		// Strip trailing `/` so that entry.path (which already ends
		// in `/` for directories per the filesystem service) does
		// not get a second `/` appended below. The directory marker
		// is then re-added deterministically.
		const cleanedPath = entry.path.replace(/\/+$/, '');
		const pathWithSeparator = entry.type === 'directory' ? `${cleanedPath}/` : cleanedPath;
		const basename = lastPathSegment(cleanedPath) || entry.name;
		const insertion = `[${basename}](file://${pathWithSeparator}) `;
		const newValue = value.slice(0, token.start) + insertion + value.slice(token.end);

		// Pin the post-insertion caret offset BEFORE the swap effect
		// runs; otherwise the effect would clobber it with whatever
		// the textarea's selection was at promotion time (browser-
		// dependent: usually reset to 0).
		pendingCaretOffset = token.start + insertion.length;
		caretOffsetPinned = true;

		value = newValue;
		onValueChange?.(newValue);

		// Already in contenteditable mode: this insert does not flip the
		// renderer, so the swap effect's caret restore never runs.
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
		{isCommandPickerOpen}
		{commandQuery}
		commands={availableCommands}
		onCommandPickerClose={handleCommandPickerClose}
		onCommandSelect={handleCommandSelect}
		{isPromptPickerOpen}
		{promptSearchQuery}
		{isMentionPickerOpen}
		{mentionQuery}
		{mentionAnchor}
		scopePath={mentionScopePath}
		onPromptPickerClose={handlePromptPickerClose}
		onMentionPickerClose={handleMentionPickerClose}
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
			onpaste={handlePaste}
		>
			{#if useContenteditable}
				<ChatFormContenteditable
					class="px-5 py-1.5 md:pt-0 mb-0.5"
					bind:this={inputRef}
					bind:value
					onKeydown={handleKeydown}
					onInput={() => {
						handleInput();
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
						handleInput();
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
				onMcpPromptClick={showMcpPromptButton ? () => (isPromptPickerOpen = true) : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>

	<ContextGaugePopup />

	{#if toolsStore.builtinTools.length > 0}
		<ChatFormWorkingDirectory
			directory={cwd}
			isOpen={isWorkingDirectoryPickerOpen}
			bind:query={workingDirectoryQuery}
			customAnchor={mentionAnchor}
			onChange={handleWorkingDirectoryChange}
			onClose={handleWorkingDirectoryClose}
			onOpen={handleWorkingDirectoryOpen}
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
