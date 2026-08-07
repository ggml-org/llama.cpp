<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import {
		badgeAwareWordJump,
		buildFragment,
		leadingBadgeEdgeOffset,
		rangeToTextOffset,
		serializeContent,
		SourceHistory,
		tokenizeContent,
		textOffsetToRange
	} from '$lib/utils';
	import type { ContentToken, SourceHistoryEntry } from '$lib/utils';

	interface Props {
		class?: string;
		disabled?: boolean;
		onInput?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
	}

	let {
		class: className = '',
		disabled = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
	}: Props = $props();

	let rootElement: HTMLDivElement | undefined = $state();
	let lastEmittedValue = '';
	let isComposing = $state(false);

	// Undo/redo in source space: the imperative token rebuilds destroy the
	// browser's native undo stack.
	const history = new SourceHistory();

	// Browsers disagree on what an empty contenteditable contains (`<br>`,
	// `<div><br></div>`, or nothing), so emptiness is decided by the
	// serialized source, not the DOM shape.
	function syncEmptyState(serialized?: string) {
		if (!rootElement) return;
		const source = serialized ?? serializeContent(rootElement);
		rootElement.dataset.empty = source.length === 0 ? 'true' : 'false';
	}

	function renderTokens(tokens: ContentToken[]) {
		if (!rootElement) return;

		const caret = rangeToTextOffset(rootElement, safeRange());

		// eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the contenteditable host, never its children
		rootElement.replaceChildren(buildFragment(tokens));

		restoreCaret(caret);
		resizeHeight();
		syncEmptyState();
	}

	function safeRange(): Range | null {
		if (!rootElement) return null;

		const selection = window.getSelection();
		if (!selection || selection.rangeCount === 0) return null;

		const range = selection.getRangeAt(0);

		if (!rootElement.contains(range.startContainer) || !rootElement.contains(range.endContainer)) {
			return null;
		}

		return range;
	}

	function restoreCaret(offset: number, extend = false) {
		if (!rootElement) return;

		const target = textOffsetToRange(rootElement, offset);
		const selection = window.getSelection();
		if (!selection) return;

		if (extend && selection.anchorNode) {
			selection.setBaseAndExtent(
				selection.anchorNode,
				selection.anchorOffset,
				target.startContainer,
				target.startOffset
			);
			return;
		}

		selection.removeAllRanges();
		selection.addRange(target);
	}

	function resizeHeight() {
		if (!rootElement) return;
		rootElement.style.height = 'auto';
		rootElement.style.height = `${rootElement.scrollHeight}px`;
	}

	function recordHistory(newGroup: boolean) {
		if (!rootElement) return;
		history.push(
			{ value: lastEmittedValue, caret: rangeToTextOffset(rootElement, safeRange()) },
			Date.now(),
			newGroup
		);
	}

	function handleInput(event?: Event) {
		if (isComposing || !rootElement) return;

		const serialized = serializeContent(rootElement);
		syncEmptyState(serialized);
		if (serialized === lastEmittedValue) return;

		// Plain typing/deletes coalesce per time window; structural edits
		// (paste, newline, cut, autocorrect) start a new undo group.
		const inputType = event instanceof InputEvent ? event.inputType : undefined;
		recordHistory(inputType !== 'insertText' && !inputType?.startsWith('deleteContent'));

		lastEmittedValue = serialized;
		value = serialized;
		onInput?.();
	}

	function handleCompositionStart() {
		isComposing = true;
	}

	function handleCompositionEnd() {
		isComposing = false;
		if (!rootElement) return;
		const serialized = serializeContent(rootElement);
		syncEmptyState(serialized);
		if (serialized === lastEmittedValue) return;
		recordHistory(true); // an IME commit is its own undo step
		lastEmittedValue = serialized;
		value = serialized;
		onInput?.();
	}

	/**
	 * Undo/redo is replayed from source snapshots (the token rebuilds
	 * destroy the native undo stack). Arrow keys around badges are
	 * repaired locally: a badge is a non-editable island, so plain
	 * ArrowLeft after a leading badge has no native previous position
	 * and word jumps overshoot it by a word.
	 */
	function handleKeydown(event: KeyboardEvent) {
		const mod = event.ctrlKey || event.metaKey;
		if (mod && !event.altKey && !isComposing && rootElement) {
			const key = event.key.toLowerCase();
			const isUndo = key === 'z' && !event.shiftKey;
			const isRedo = key === 'y' || (key === 'z' && event.shiftKey);

			if (isUndo || isRedo) {
				event.preventDefault();
				const current = {
					value: lastEmittedValue,
					caret: rangeToTextOffset(rootElement, safeRange())
				};
				const entry = isUndo ? history.undo(current) : history.redo(current);
				if (entry) applyHistoryEntry(entry);
				return;
			}
		}

		if (rootElement && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) {
			const isWordJump = (event.altKey || event.ctrlKey) && !event.metaKey;
			const isPlainLeft =
				event.key === 'ArrowLeft' && !event.altKey && !event.ctrlKey && !event.metaKey;

			if (isWordJump || isPlainLeft) {
				const source = serializeContent(rootElement);
				const caret = rangeToTextOffset(rootElement, safeRange());
				const target = isWordJump
					? badgeAwareWordJump(source, caret, event.key === 'ArrowRight' ? 'forward' : 'backward')
					: leadingBadgeEdgeOffset(source, caret);

				if (target !== null) {
					event.preventDefault();
					restoreCaret(target, event.shiftKey);
					return;
				}
			}
		}

		onKeydown?.(event);
	}

	// lastEmittedValue is set before `value` so the sync effect treats the
	// change as our own and does not re-render.
	function applyHistoryEntry(entry: SourceHistoryEntry) {
		if (!rootElement) return;
		renderTokens(tokenizeContent(entry.value));
		lastEmittedValue = entry.value;
		value = entry.value;
		onInput?.();
		restoreCaret(entry.caret);
	}

	/**
	 * Plain-text paste. preventDefault + manual insertText keeps the
	 * browser from producing stray `<div>` wrappers mid-paste, and the
	 * result is fed through the tokenizer so pasted `[name](file://...)`
	 * segments render as badges right away.
	 */
	function handlePasteEvent(event: ClipboardEvent) {
		const pasted = event.clipboardData?.getData('text/plain');
		if (pasted && pasted.length > 0) {
			event.preventDefault();

			// Snap a collapsed caret through the offset mapping first: at
			// element-boundary carets (e.g. right before a badge) Chromium's
			// insertText can drop the preceding text node's trailing whitespace.
			const range = safeRange();
			if (rootElement && range && range.collapsed) {
				restoreCaret(rangeToTextOffset(rootElement, range));
			}

			document.execCommand('insertText', false, pasted);

			// insertText fires `input` synchronously, so the source is already
			// emitted; rebuild only to badge-ify pasted mention links.
			if (rootElement && tokenizeContent(pasted).some((token) => token.kind === 'badge')) {
				renderTokens(tokenizeContent(serializeContent(rootElement)));
			}
		}
	}

	// The parent's paste handler runs first and preventDefaults when it
	// consumes the event (files, quoted prompts, long text).
	function handlePaste(event: ClipboardEvent) {
		onPaste?.(event);
		if (!event.defaultPrevented) {
			handlePasteEvent(event);
		}
	}

	// The selection as markdown SOURCE (each badge contributes its full
	// `[name](file://...)` link), so copy/cut carry raw markdown and
	// pasting back re-renders the badges. Null for collapsed/outside
	// selections - native clipboard behavior is fine there.
	function selectionSourceSlice(): { text: string; range: Range } | null {
		if (!rootElement) return null;

		const range = safeRange();
		if (!range || range.collapsed) return null;

		const startRange = range.cloneRange();
		startRange.collapse(true);

		const source = serializeContent(rootElement);
		const start = rangeToTextOffset(rootElement, startRange);
		const end = rangeToTextOffset(rootElement, range);

		return { text: source.slice(start, end), range };
	}

	function handleCopy(event: ClipboardEvent) {
		const slice = selectionSourceSlice();
		if (!slice) return;

		event.clipboardData?.setData('text/plain', slice.text);
		event.preventDefault();
	}

	function handleCut(event: ClipboardEvent) {
		const slice = selectionSourceSlice();
		if (!slice) return;

		event.clipboardData?.setData('text/plain', slice.text);
		event.preventDefault();

		// preventDefault suppresses the native deletion, so remove the
		// selection manually and re-emit.
		slice.range.deleteContents();
		handleInput();
	}

	onMount(() => {
		// untrack: the DOM is managed manually from input events, so the
		// initial render must not subscribe to the value.
		renderTokens(tokenizeContent(untrack(() => value)));
		lastEmittedValue = untrack(() => value ?? '');
		resizeHeight();
		syncEmptyState();
		if (!isMobile.current) {
			rootElement?.focus({ preventScroll: true });
		}
	});

	// External `value` updates. When incoming === lastEmittedValue the
	// change came from our own input, so leave the DOM alone - the
	// browser already owns the right shape.
	$effect(() => {
		const incoming = value ?? '';
		if (incoming === lastEmittedValue) return;

		recordHistory(true); // external edit (mention insert, clear, ...): own undo step
		renderTokens(tokenizeContent(incoming));
		lastEmittedValue = incoming;
	});

	export function getElement() {
		return rootElement;
	}

	export function getCaretOffset(): number {
		if (!rootElement) return 0;
		return rangeToTextOffset(rootElement, safeRange());
	}

	// Focus first: `selection.addRange` requires it on some browsers.
	export function setCaretOffset(offset: number) {
		if (rootElement && rootElement !== document.activeElement) {
			rootElement.focus({ preventScroll: true });
		}
		restoreCaret(offset);
	}

	export function focus() {
		if (isMobile.current) return;
		rootElement?.focus({ preventScroll: true });
	}

	export function resetHeight() {
		if (rootElement) {
			rootElement.style.height = '';
			resizeHeight();
		}
	}
</script>

<div class="flex-1 {className}">
	<div
		bind:this={rootElement}
		contenteditable={!disabled}
		role="textbox"
		aria-multiline="true"
		aria-disabled={disabled}
		aria-placeholder={placeholder}
		data-placeholder={placeholder}
		tabindex={disabled ? -1 : 0}
		class={[
			'chat-form-contenteditable text-md min-h-12 w-full whitespace-pre-wrap wrap-break-word border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0',
			disabled && 'cursor-not-allowed'
		]}
		style="max-height: var(--max-message-height);"
		oncompositionstart={handleCompositionStart}
		oncompositionend={handleCompositionEnd}
		oninput={handleInput}
		onkeydown={handleKeydown}
		onpaste={handlePaste}
		oncopy={handleCopy}
		oncut={handleCut}
	></div>
</div>

<style>
	.chat-form-contenteditable:global([data-empty='true'])::before {
		content: attr(data-placeholder);
		color: var(--muted-foreground);
		pointer-events: none;
	}
</style>
