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
		CHAT_FORM_RESIZE_MS,
		CHAT_FORM_SHEET_MS,
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		ICON_CLASS_DEFAULT,
		INITIAL_FILE_SIZE,
		INPUT_CLASSES,
		PROMPT_CONTENT_SEPARATOR,
		SETTING_CONFIG_DEFAULT,
		UI_DATA_ATTRS
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
		deviceStore,
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
	import { flushSync, onDestroy, onMount, tick, untrack } from 'svelte';
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

	// the card's translucent fill as it looks over the page background, made solid
	const EXPANDED_SHEET_FILL =
		'max-md:bg-[color-mix(in_oklab,var(--muted)_60%,var(--background))] max-md:dark:bg-[color-mix(in_oklab,var(--muted)_75%,var(--background))]';
	/** The phone sheet's ease-out: no overshoot, so it lands without a wobble. */
	const SHEET_EASING = 'cubic-bezier(0.32, 0.72, 0, 1)';

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

	// On a phone the expanded form is a full-screen sheet. Animating its size would lay
	// out (and repaint) the whole screen every frame, so it moves by transform alone, on
	// the compositor (FLIP): expanding switches to the sheet's layout at once and eases
	// the card from where the collapsed card was; collapsing slides the sheet down to the
	// collapsed card's place and only then switches back. The actions get the opposite
	// shift, so they stay by the bottom edge instead of travelling with the card's top.
	let cardEl: HTMLDivElement | undefined = $state();
	let actionsEl: HTMLDivElement | undefined = $state();
	let cwdEl: HTMLDivElement | undefined = $state();
	let isSheetMoving = $state(false);
	// WebKit builds the card's backdrop-filter layer with the last clip the sheet's
	// animation drew, in the sheet's coordinates, when the blur comes back right as the
	// card takes its collapsed place: the blur waits for the move to be over
	let isBlurHeld = $state(false);
	let collapsedRects: SheetRects | null = null;
	// where the sheet's bottom edge was when it opened: the collapsed card's place is
	// held relative to it (see collapseSheet)
	let sheetBottomAtOpen = 0;
	let wasExpanded = false;

	type SheetRects = { actions: DOMRect; card: DOMRect; cwd: DOMRect | null };
	type SheetFrame = { actions: Keyframe; card: Keyframe; cwd: Keyframe };

	function animatesSheet(): boolean {
		return (
			matchMedia('(width < 48rem)').matches &&
			!matchMedia('(prefers-reduced-motion: reduce)').matches
		);
	}

	function sheetBottom(): number {
		return cardEl?.closest('form')?.getBoundingClientRect().bottom ?? 0;
	}

	function measure(): SheetRects | null {
		return cardEl && actionsEl
			? {
					actions: actionsEl.getBoundingClientRect(),
					card: cardEl.getBoundingClientRect(),
					cwd: cwdEl?.getBoundingClientRect() ?? null
				}
			: null;
	}

	/** The laid-out sheet as it is: full screen, the working-directory chip out of sight. */
	const SHEET_AT_REST: SheetFrame = {
		actions: { top: '0px' },
		card: { borderRadius: '0px', clipPath: 'inset(0 0 0 0 round 0px)', top: '0px' },
		cwd: { left: '0px', opacity: 0, top: '0px' }
	};

	/**
	 * The laid-out sheet drawn as the collapsed card at `to`: moved up or down, its sides
	 * and bottom clipped to that card's edges (its content already sits where the card's
	 * does, see the sheet's insets), its actions and chip held at their collapsed places.
	 *
	 * Everything moves by relative offsets, not transforms, so it all runs on the main
	 * thread, on the clip's clock: WebKit hands transforms to its compositor even within
	 * an animation that also clips, which starts them a frame or two apart, and the
	 * clipped edges and the content they frame drift apart. An offset reflows nothing
	 * around it.
	 */
	function frameAt(to: SheetRects): SheetFrame {
		const from = measure()!;
		const dy = to.card.top - from.card.top;
		const clip = [
			0,
			from.card.right - to.card.right,
			from.card.height - to.card.height,
			to.card.left - from.card.left
		].map((px) => `${Math.max(0, px)}px`);

		return {
			actions: { top: `${to.actions.top - from.actions.top - dy}px` },
			card: {
				borderRadius: '2rem',
				clipPath: `inset(${clip.join(' ')} round 2rem)`,
				top: `${dy}px`
			},
			cwd: {
				left: `${to.cwd && from.cwd ? to.cwd.left - from.cwd.left : 0}px`,
				opacity: 1,
				top: `${to.cwd && from.cwd ? to.cwd.top - from.cwd.top : 0}px`
			}
		};
	}

	/**
	 * Plays the frames on the card, its actions and the chip. Nothing holds the last
	 * frame: WebKit can keep drawing a held, then cancelled, frame's clip on the card
	 * after its layout changed. A move that ends where the layout does not (collapsing)
	 * passes `onEnd`, which switches the layout as the animation ends, before that frame
	 * is painted.
	 */
	function animateSheet(
		frames: SheetFrame[],
		duration = CHAT_FORM_SHEET_MS,
		onEnd?: () => void
	): Promise<unknown> {
		const options: KeyframeAnimationOptions = { duration, easing: SHEET_EASING };
		const targets: [HTMLElement | undefined, keyof SheetFrame][] = [
			[cardEl, 'card'],
			[actionsEl, 'actions'],
			[cwdEl, 'cwd']
		];
		const animations = targets
			.filter(([el]) => el)
			.map(([el, part]) =>
				el!.animate(
					frames.map((f) => f[part]),
					options
				)
			);

		if (onEnd) animations[0].addEventListener('finish', onEnd, { once: true });

		return Promise.all(animations.map((a) => a.finished));
	}

	function clearSheetAnimations() {
		for (const el of [cardEl, actionsEl, cwdEl]) {
			el?.getAnimations()
				.filter((a) => !(a instanceof CSSTransition))
				.forEach((a) => a.cancel());
		}
	}

	// The expanded size is built on the collapsed card's height (on a new chat the card
	// grows by exactly the distance the form drops, so its top edge stays put), and its
	// height transition starts from it. A message of a few lines makes that card taller
	// than its one-line minimum, so the real height is held in a variable on the card,
	// from expanding until the collapse has settled: a quick toggle back and forth keeps
	// it rather than measuring a card that is halfway through a transition.
	let collapsedHold: {
		card: number;
		content: number;
		text: string;
		editorMax: number;
		editorMin: number;
	} | null = null;
	let collapsedReleaseTimer: ReturnType<typeof setTimeout> | undefined;

	function editorContentHeight(): number {
		return cardEl?.querySelector<HTMLElement>('.cm-content')?.offsetHeight ?? 0;
	}

	function setCollapsedHeight(px: number | null) {
		if (px === null) cardEl?.style.removeProperty('--chat-form-collapsed-height');
		else cardEl?.style.setProperty('--chat-form-collapsed-height', `${px}px`);
	}

	function holdCollapsedHeight() {
		clearTimeout(collapsedReleaseTimer);

		if (collapsedHold || !cardEl) return;

		const editor = getComputedStyle(cardEl.querySelector<HTMLElement>('.chat-form-input')!);

		collapsedHold = {
			card: cardEl.getBoundingClientRect().height,
			content: editorContentHeight(),
			editorMax: parseFloat(editor.maxHeight) || Infinity,
			editorMin: parseFloat(editor.minHeight) || 0,
			text: value ?? ''
		};

		// in place before the switch, with no transition of its own, so the height
		// transition the switch starts begins at the card's real height
		cardEl.style.transition = 'none';
		setCollapsedHeight(collapsedHold.card);
		void getComputedStyle(cardEl).minHeight;
		cardEl.style.transition = '';
	}

	function settleCollapsedHeight() {
		if (!collapsedHold) return;

		// Back to the height held on expanding, exactly, unless the text changed while
		// expanded: then to the height the card will have with it (its editor clamped as
		// when collapsed), so the move stays one line. The editor is only measured then:
		// mid-move, and as it redraws on taking focus, its height is not the text's
		const hold = collapsedHold;
		const editorBox = (content: number) =>
			Math.max(hold.editorMin, Math.min(content, hold.editorMax));

		setCollapsedHeight(
			(value ?? '') === hold.text
				? hold.card
				: hold.card - editorBox(hold.content) + editorBox(editorContentHeight())
		);

		clearTimeout(collapsedReleaseTimer);
		collapsedReleaseTimer = setTimeout(() => {
			collapsedHold = null;
			setCollapsedHeight(null);
		}, CHAT_FORM_RESIZE_MS);
	}

	// before the DOM switches: where the collapsed card is, and how tall
	$effect.pre(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded === wasExpanded) return;

			if (!expanded) {
				settleCollapsedHeight();

				return;
			}

			if (!isSheetMoving && animatesSheet()) collapsedRects = measure();

			holdCollapsedHeight();
		});
	});

	// after it switched: ease the sheet in from there
	$effect(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded && !wasExpanded && collapsedRects && !isSheetMoving) {
				void expandSheet(collapsedRects);
			}

			if (!expanded) collapsedRects = null;

			wasExpanded = expanded;
		});
	});

	async function expandSheet(from: SheetRects) {
		isSheetMoving = true;
		// the page lays the form out flush with the screen in the same update (its
		// wrapper follows uiStore): measure once all of it has landed, before a paint
		await tick();
		sheetBottomAtOpen = sheetBottom();

		try {
			await animateSheet([frameAt(from), SHEET_AT_REST]);
		} catch {
			// cancelled (the form went away)
		}

		clearSheetAnimations();
		isSheetMoving = false;
	}

	async function collapseSheet() {
		if (!collapsedRects || !measure()) {
			isExpanded = false;

			return;
		}

		// iOS Safari resizes its viewport while a full-screen sheet is up, and the screen
		// position the collapsed card had when the sheet opened moves with it: shift that
		// place by how far the sheet's bottom edge has moved since, so the sheet lands
		// where the card will rest, in one move
		const shift = sheetBottom() - sheetBottomAtOpen;
		const moved = (rect: DOMRect | null) =>
			rect && new DOMRect(rect.x, rect.y + shift, rect.width, rect.height);
		const target: SheetRects = {
			actions: moved(collapsedRects.actions)!,
			card: moved(collapsedRects.card)!,
			cwd: moved(collapsedRects.cwd)
		};

		isSheetMoving = true;
		isBlurHeld = true;

		try {
			await animateSheet([SHEET_AT_REST, frameAt(target)], CHAT_FORM_SHEET_MS, () => {
				flushSync(() => (isExpanded = false));
			});
		} catch {
			// cancelled (the form went away): settle at once
		}

		isExpanded = false;
		await tick();

		// the text may have changed while expanded, and with it the collapsed height:
		// close the small remaining gap the same way
		const settled = measure();

		if (settled && Math.abs(settled.card.top - target.card.top) > 1) {
			const dy = target.card.top - settled.card.top;

			await animateSheet(
				[
					{ actions: {}, card: { top: `${dy}px` }, cwd: {} },
					{ actions: {}, card: { top: '0px' }, cwd: {} }
				],
				CHAT_FORM_SHEET_MS / 2
			).catch(() => {});
			clearSheetAnimations();
		}

		isSheetMoving = false;

		await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
		isBlurHeld = false;
	}

	// on a phone the page around the expanded sheet takes its fill (see app.css), once
	// the sheet has settled: repainting the whole page under it mid-move costs frames
	$effect(() => {
		if (!expandable || !isExpanded || isSheetMoving) return;

		document.documentElement.setAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET, '');

		return () => document.documentElement.removeAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET);
	});

	onDestroy(() => {
		clearTimeout(collapsedReleaseTimer);

		if (expandable) uiStore.isComposerExpanded = false;
	});

	function toggleExpanded() {
		if (isSheetMoving) return;

		// collapsing an empty form leaves the next message free to expand it again
		isCollapsedByUser = isExpanded && !!(value ?? '').trim();

		if (isExpanded && animatesSheet()) {
			void collapseSheet();

			return;
		}

		isExpanded = !isExpanded;
		// after the switch, so the editor redrawing as it takes focus can not reach into
		// the heights measured for it; on a phone focus stays where it is, as focusing
		// would open the keyboard (and iOS would scroll the page to the editor mid-move)
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

			// In the expanded editor, or with the caret inside a fenced code
			// block (closed, or still open while being typed): Enter adds a
			// line and only Cmd/Ctrl+Enter submits - skipping submit lets the
			// editor's own newline through.
			if (
				!isModifier &&
				(isExpanded || isOffsetInCodeBlock(value ?? '', inputRef?.getCaretOffset() ?? 0))
			) {
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
		// scrolls; past that the card shrinks to fit and the editor scrolls inside. On a
		// phone it is the whole screen
		isExpanded &&
			'max-h-[calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-2rem)] max-md:max-h-dvh',
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
		bind:this={cardEl}
		class={[
			INPUT_CLASSES,
			'group/chat-form relative flex shrink flex-col overflow-hidden rounded-4xl backdrop-blur-md transition-[min-height,padding,border-radius,border-color,background-color] duration-400 ease-[var(--chat-form-resize-ease,cubic-bezier(0.3,1.12,0.5,1))] md:rounded-3xl',
			// on a phone it moves by transform instead (see animateSheet); only its border,
			// hidden on the sheet and clipped away while it moves, fades back in after
			'max-md:transition-[border-color] max-md:duration-200',
			// Expanded, it opens to about half the screen, or on a new chat (where the page
			// sets how far the form drops, see ChatScreen) to its collapsed height plus that
			// drop, and grows from there with the text. Collapsed, the minimum is its
			// one-line height, or its real height while expanding and collapsing (see
			// holdCollapsedHeight), so the transition starts where the card already is
			'[--chat-form-expanded-min-height:calc(var(--chat-form-drop)+var(--chat-form-collapsed-height))]',
			isExpanded
				? 'min-h-[min(var(--chat-form-expanded-min-height,50dvh),calc(100dvh-var(--chat-tabs-offset,0px)-var(--chat-form-padding-top)-5rem))]'
				: 'min-h-0 md:min-h-(--chat-form-collapsed-height)',
			// On a phone the expanded card is a square sheet filling the screen. Its
			// fill is opaque (what the translucent one looks like over the page), so the
			// Safari toolbar strip below can match it and it needs no backdrop blur, the
			// costliest thing to repaint, and its actions keep the gap the collapsed form
			// keeps from the screen's bottom (more in an installed app)
			(isExpanded || isBlurHeld) && 'max-md:backdrop-blur-none',
			isExpanded && [
				'max-md:min-h-dvh max-md:rounded-none max-md:pb-[var(--chat-form-bottom-position,0px)]',
				// no visible border (and so no focus outline): the sheet meets the screen's
				// edges. Transparent rather than removed, so nothing inside shifts by its width
				'max-md:border-transparent max-md:focus-within:border-transparent max-md:dark:border-transparent max-md:dark:focus-within:border-transparent',
				EXPANDED_SHEET_FILL
			],
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
						// on the phone sheet, where the collapsed card's button is (see the
						// sheet's content inset below)
						isExpanded && 'max-md:right-[calc(0.625rem+var(--chat-form-side-inset,0.5rem))]',
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
			class={isExpanded ? 'py-5 max-md:px-(--chat-form-side-inset)' : 'py-5'}
			limitToSingleRow
			onFileRemove={handleFileRemove}
			style="scroll-padding: 1rem;"
		/>

		<div
			class={[
				'relative flex min-h-12 flex-1 flex-col rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3! max-md:transition-none',
				// The phone sheet is wider on each side than the collapsed card by the page's
				// side inset (more in an installed app): its content is inset by as much, so
				// text and actions keep the card's places and widths (nothing reflows, nor
				// truncates the model name anew) while it moves
				isExpanded && 'max-md:px-(--chat-form-side-inset)',
				isExpanded && 'max-md:rounded-none max-md:focus-within:shadow-sm'
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

			<!-- its own box, so the phone sheet can hold it by the bottom edge while it moves -->
			<div bind:this={actionsEl} class="relative mt-auto">
				<ChatFormActions
					bind:this={chatFormActionsRef}
					canSend={canSubmit}
					class="px-3"
					{disabled}
					{isLoading}
					isReasoning={chatStore.isReasoning}
					{isRecording}
					onFileUpload={handleFileUpload}
					onMicClick={handleMicClick}
					{onStop}
					onSystemPromptClick={() =>
						onSystemPromptClick?.({ files: uploadedFiles, message: value })}
					{showAddButton}
					{showModelSelector}
					{uploadedFiles}
				/>
			</div>
		</div>
	</div>

	<ContextGaugePopup />

	<!-- iOS Safari tints its bottom toolbar from a solid fixed element at the screen's
	     bottom edge: this strip, under the expanded sheet, carries the sheet's fill. Like
	     the page's tint, only while the sheet rests: mid-move it would show below it -->
	{#if isExpanded && !isSheetMoving && deviceStore.isIOSSafari}
		<div
			aria-hidden="true"
			class={[
				'pointer-events-none fixed inset-x-0 bottom-0 -z-10 h-3 md:hidden',
				EXPANDED_SHEET_FILL
			]}
		></div>
	{/if}

	{#if conversationsStore.preferences.hasEnabledCwdTools()}
		<div
			bind:this={cwdEl}
			class={[
				'relative self-start transition-[translate,opacity,margin] duration-400 ease-[cubic-bezier(0.3,1.12,0.5,1)] max-md:transition-none',
				// on a phone the expanded sheet reaches the screen's bottom: the chip gives up
				// its room (its 1.5 top margin plus its 6 height) and waits below it, out of
				// sight; it travels there and back with the sheet (see animateSheet)
				isExpanded &&
					'max-md:pointer-events-none max-md:-mb-7.5 max-md:translate-y-full max-md:opacity-0'
			]}
		>
			<ChatFormCurrentWorkingDirectory
				bind:query={pickers.workingDirectoryQuery}
				customAnchor={mentionAnchor}
				directory={cwd}
				{disabled}
				isOpen={pickers.isWorkingDirectoryPickerOpen}
				onChange={handleWorkingDirectoryChange}
				onClose={pickers.handleWorkingDirectoryClose}
				onOpen={pickers.handleWorkingDirectoryOpen}
			/>
		</div>
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
