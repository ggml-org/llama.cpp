<script lang="ts">
	import { onDestroy, onMount, untrack } from 'svelte';
	import { mode } from 'mode-watcher';
	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';
	import { isMobile } from '$lib/stores/viewport.svelte';
	import { ColorMode } from '$lib/enums';
	import { TRIM_LEADING_PADDING_REGEX, TRIM_TRAILING_PADDING_REGEX } from '$lib/constants';
	import {
		badgeAwareWordJump,
		buildFragment,
		domMatchesTokens,
		highlightCode,
		isIMEComposing,
		isOffsetInCodeBlock,
		leadingBadgeEdgeOffset,
		rangeToTextOffset,
		serializeContent,
		SourceHistory,
		stripBlockBoundaryLineBreaks,
		syncCodeBlockHatches,
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

		syncCodeBlockHatches(rootElement);
		highlightCodeBlocks(rootElement);

		restoreCaret(caret);
		resizeHeight();
		syncEmptyState();
	}

	// Last highlighted source segment per block element - typing inside
	// a block re-highlights only when the segment actually changed.
	const highlightedSegments = new WeakMap<HTMLElement, string>();

	const CODE_BLOCK_OPEN_RE = /^```([^\n`]*)\n/;

	/**
	 * Apply syntax highlighting to a code block element's CONTENT. The
	 * fence lines stay plain text, and the blank padding that
	 * `highlightCode` trims is re-added as plain text, so the element's
	 * textContent stays byte-exact with the source segment. Replaces
	 * the element's children - callers restore the caret afterwards.
	 * Returns false when nothing changed.
	 */
	function highlightCodeBlockElement(el: HTMLElement): boolean {
		const segment = el.textContent ?? '';
		if (highlightedSegments.get(el) === segment) return false;

		const open = CODE_BLOCK_OPEN_RE.exec(segment);
		if (!open) return false;

		const prefix = open[0];
		const language = open[1].trim().split(/\s+/)[0] ?? '';
		const content = segment.slice(prefix.length, -3);

		const leading = content.match(TRIM_LEADING_PADDING_REGEX)?.[0] ?? '';
		const trailing = content.match(TRIM_TRAILING_PADDING_REGEX)?.[0] ?? '';
		const core = content.slice(leading.length, content.length - trailing.length);

		// autoDetect off: re-guessing the language on every keystroke
		// costs ~38ms a call and flickers while typing
		const html = core ? highlightCode(core, language || 'text', false) : '';
		const tpl = document.createElement('template');
		tpl.innerHTML = html;

		el.replaceChildren(
			document.createTextNode(prefix + leading),
			tpl.content.cloneNode(true),
			document.createTextNode(trailing + '```')
		);
		highlightedSegments.set(el, segment);
		return true;
	}

	function highlightCodeBlocks(root: HTMLElement) {
		for (const el of root.querySelectorAll<HTMLElement>('code[data-code-token="block"]')) {
			highlightCodeBlockElement(el);
		}
	}

	/**
	 * Re-highlight the code block the caret sits in after an edit.
	 * Skipped when the block's segment is unchanged since its last
	 * highlight, so edits outside blocks cost nothing.
	 */
	function rehighlightCaretCodeBlock() {
		if (!rootElement) return;

		const range = safeRange();
		if (!range) return;

		let node: Node | null = range.startContainer;
		if (node === rootElement) {
			node = rootElement.childNodes[range.startOffset - 1] ?? null;
		}

		while (node && node !== rootElement) {
			if (node instanceof HTMLElement && node.dataset.codeToken === 'block') {
				const caret = rangeToTextOffset(rootElement, range);
				if (highlightCodeBlockElement(node)) {
					restoreCaret(caret);
				}
				return;
			}
			node = node.parentNode;
		}
	}

	/**
	 * Is the caret inside a fenced code block region? Source-level
	 * (not DOM-level) so the still-OPEN fence counts too: while the
	 * user is typing a block, no closing ``` exists yet and the
	 * buffer is plain text with no block element to find. Root-level
	 * caret positions right at a closed block's edge (escape
	 * hatches, element boundaries restored by `textOffsetToRange`)
	 * resolve past the closing fence, so they count as OUTSIDE.
	 */
	function caretInCodeBlock(): boolean {
		if (!rootElement) return false;

		return isOffsetInCodeBlock(
			serializeContent(rootElement),
			rangeToTextOffset(rootElement, safeRange())
		);
	}

	/**
	 * hljs theme for the highlighted code blocks. Mirrors
	 * SyntaxHighlightedCode.svelte: one shared style element
	 * (deduped via the data attribute) swapped on mode change.
	 */
	function loadHighlightTheme(isDark: boolean) {
		document.querySelectorAll('style[data-highlight-theme-preview]').forEach((s) => s.remove());

		const style = document.createElement('style');
		style.setAttribute('data-highlight-theme-preview', 'true');
		style.textContent = isDark ? githubDarkCss : githubLightCss;

		document.head.appendChild(style);
	}

	$effect(() => {
		loadHighlightTheme(mode.current === ColorMode.DARK);
	});

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

	/**
	 * Re-emit the current markdown source value to the parent, then
	 * reconcile the DOM against the token stream: when a code span
	 * was just completed or broken, the token boundaries no longer
	 * match the element structure and the DOM is rebuilt (caret
	 * preserved through the source-offset mapping).
	 */
	function processInput(inputType?: string) {
		if (isComposing || !rootElement) return;

		syncEmptyState();
		resizeHeight();

		// Shift+Enter right after a code block leaves an all-newline
		// text node (the fence's separator line plus Chromium's
		// artificial end-of-buffer line break). Strip both so the caret
		// lands on the line directly below the block.
		if (inputType === 'insertLineBreak' || inputType === 'insertParagraph') {
			const caret = rangeToTextOffset(rootElement, safeRange());
			if (stripBlockBoundaryLineBreaks(rootElement)) {
				restoreCaret(caret);
			} else {
				const source = serializeContent(rootElement);
				let end = caret;

				// the caret must end up after the inserted \n; some browsers
				// leave it before (stuck at the end of the old line). A
				// preceding \n means it already sits past the break
				// (Chromium's artificial trailing newline) - leave it.
				if (source[end] === '\n' && source[end - 1] !== '\n') {
					end += 1;
					restoreCaret(end);
				}

				// a line break at the buffer end renders only with a second,
				// artificial trailing \n: a lone trailing \n is collapsed, so
				// the new line is invisible and the next typed character
				// consumes it. Append it when missing - unless the trailing
				// \n doubles as a block's separator line (source ends with
				// \n\n) or sits inside a block element.
				let last = rootElement.lastChild;
				while (last && last.nodeName === 'BR') last = last.previousSibling;
				if (
					end === source.length &&
					source.endsWith('\n') &&
					source[source.length - 2] !== '\n' &&
					last?.nodeType === Node.TEXT_NODE
				) {
					// eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the contenteditable host, never its children
					rootElement.appendChild(document.createTextNode('\n'));
					restoreCaret(source.length);
					resizeHeight();
				}
			}
		}

		syncCodeBlockHatches(rootElement);

		const serialized = serializeContent(rootElement);
		syncEmptyState(serialized);
		if (serialized === lastEmittedValue) return;

		// Plain typing/deletes coalesce per time window; structural edits
		// (paste, newline, cut, autocorrect) start a new undo group.
		recordHistory(inputType !== 'insertText' && !inputType?.startsWith('deleteContent'));

		lastEmittedValue = serialized;
		value = serialized;

		// Rebuild when token boundaries shifted (a code span was just
		// completed or broken) - the browser-owned text nodes cannot
		// restyle themselves across element boundaries.
		const tokens = tokenizeContent(serialized);
		if (!domMatchesTokens(rootElement, tokens)) {
			renderTokens(tokens);

			// The rebuild can re-shape the DOM in a way that changes the
			// serialization (e.g. Chromium merged trailing text into the
			// block element and the rebuild splits it back out, which
			// synthesizes the separator newline) - keep value in sync.
			const reserialized = serializeContent(rootElement);
			if (reserialized !== serialized) {
				lastEmittedValue = reserialized;
				value = reserialized;
			}
		} else {
			rehighlightCaretCodeBlock();
		}

		onInput?.();
	}

	function handleInput(event: Event) {
		processInput((event as InputEvent).inputType);
	}

	function handleCompositionStart() {
		isComposing = true;
	}

	function handleCompositionEnd() {
		isComposing = false;
		processInput();
	}

	/**
	 * Insert a line break at the caret MANUALLY. Native Shift+Enter at
	 * the buffer end varies across browsers (a lone trailing \n that the
	 * renderer collapses, or a <br> that the hatch sync strips), which
	 * can leave the caret stuck on the old line; splitting the text node
	 * ourselves keeps the DOM shape - and the caret - deterministic.
	 * `processInput` then appends the artificial trailing \n when the
	 * break lands at the buffer end.
	 */
	function insertLineBreak() {
		if (!rootElement) return;

		const range = safeRange();
		if (!range) return;

		if (!range.collapsed) {
			range.deleteContents();
		}

		const container = range.startContainer;
		const offset = range.startOffset;
		const nl = document.createTextNode('\n');

		// a break at the very end of a code block exits the block (the
		// new line belongs below it, not inside)
		let exitBlock: HTMLElement | null = null;
		if (container.nodeType === Node.TEXT_NODE) {
			let node: Node | null = container.parentNode;
			while (node && node !== rootElement) {
				if (node instanceof HTMLElement && node.dataset.codeToken === 'block') {
					const tail = document.createRange();
					tail.setStart(container, offset);
					tail.setEnd(node, node.childNodes.length);
					if (tail.toString().length === 0) exitBlock = node;
					break;
				}
				node = node.parentNode;
			}
		}

		if (exitBlock) {
			exitBlock.after(nl);
		} else if (container.nodeType === Node.TEXT_NODE) {
			const text = container as Text;
			if (offset === 0) {
				text.before(nl);
			} else if (offset === text.length) {
				text.after(nl);
			} else {
				text.splitText(offset).before(nl);
			}
		} else {
			container.insertBefore(nl, container.childNodes[offset] ?? null);
		}

		const selection = window.getSelection();
		const after = document.createRange();
		after.setStartAfter(nl);
		after.collapse(true);
		selection?.removeAllRanges();
		selection?.addRange(after);

		processInput('insertLineBreak');
	}

	/**
	 * Arrow escape to the line BEFORE a leading code block. Native
	 * caret movement has no position above a buffer-starting block,
	 * so a transient `<br>` hatch is created on demand: it gives the
	 * caret a visible line, is consumed by the first character typed
	 * on it, and is removed again when the caret leaves (see
	 * handleSelectionChange). Returns true when the caret was moved.
	 */
	function moveCaretBeforeLeadingCodeBlock(key: string, extend: boolean): boolean {
		if (!rootElement) return false;

		// a hatch already exists - native movement handles it
		if (rootElement.firstChild?.nodeName === 'BR') return false;

		const first = rootElement.firstChild;
		if (!(first instanceof HTMLElement) || first.dataset.codeToken !== 'block') return false;

		const range = safeRange();
		if (!range || !range.collapsed) return false;

		// the caret must sit inside the block: on its very first
		// character for ArrowLeft, anywhere on its first line for
		// ArrowUp
		if (!first.contains(range.startContainer)) return false;

		const caret = rangeToTextOffset(rootElement, range);
		if (key === 'ArrowLeft') {
			if (caret !== 0) return false;
		} else {
			const firstLineEnd = (first.textContent ?? '').indexOf('\n');
			if (firstLineEnd !== -1 && caret > firstLineEnd) return false;
		}

		// eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the contenteditable host, never its children
		rootElement.prepend(document.createElement('br'));
		restoreCaret(0, extend);
		return true;
	}

	/**
	 * Remove the transient leading hatch once the caret leaves it.
	 * The hatch only exists to give the caret a line above a leading
	 * code block; with the caret anywhere else the empty line would
	 * just be visual noise. Typing on the hatch line consumes it via
	 * the stale-hatch removal in `syncCodeBlockHatches` instead (the
	 * new text node takes its place before the block).
	 */
	function handleSelectionChange() {
		if (!rootElement) return;

		const first = rootElement.firstChild;
		if (first?.nodeName !== 'BR') return;

		const second = first.nextSibling;
		if (!(second instanceof HTMLElement) || second.dataset.codeToken !== 'block') return;

		const range = safeRange();
		const onHatch =
			range !== null && range.startContainer === rootElement && range.startOffset === 0;
		if (!onHatch) {
			first.remove();
		}
	}

	/**
	 * Undo/redo is replayed from source snapshots (the token rebuilds
	 * destroy the native undo stack). Arrow keys around badges are
	 * repaired locally: a badge is a non-editable island, so plain
	 * ArrowLeft after a leading badge has no native previous position
	 * and word jumps overshoot it by a word.
	 *
	 * Plain Enter inside a fenced code block (closed, or still open
	 * while being typed) acts as Shift+Enter and adds a line instead of
	 * submitting. ArrowLeft/ArrowUp at the edge of a leading code block
	 * create the transient before-block hatch.
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

		if (
			event.key === 'Enter' &&
			event.shiftKey &&
			!event.ctrlKey &&
			!event.metaKey &&
			!event.altKey &&
			!isIMEComposing(event) &&
			!disabled &&
			!caretInCodeBlock() &&
			safeRange()
		) {
			// Own the break outside code blocks: native end-of-buffer
			// behavior varies across browsers and can leave the caret
			// stuck on the old line (see insertLineBreak).
			event.preventDefault();
			insertLineBreak();
			return;
		}

		if (
			event.key === 'Enter' &&
			!event.shiftKey &&
			!event.ctrlKey &&
			!event.metaKey &&
			!event.altKey &&
			!isIMEComposing(event) &&
			caretInCodeBlock()
		) {
			// The native plain-Enter path must never run: it splits the
			// buffer into `<div>` wrappers that `serializeContent` cannot
			// see. `insertLineBreak` reproduces the Shift+Enter DOM (a `\n`
			// text node) and fires `input` synchronously, so the usual
			// re-tokenize/re-highlight follows.
			event.preventDefault();
			document.execCommand('insertLineBreak');
			return;
		}

		if (
			rootElement &&
			(event.key === 'ArrowLeft' || event.key === 'ArrowUp') &&
			!event.altKey &&
			!event.ctrlKey &&
			!event.metaKey
		) {
			if (moveCaretBeforeLeadingCodeBlock(event.key, event.shiftKey)) {
				event.preventDefault();
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
	 * browser from producing stray `<div>` wrappers mid-paste; insertText
	 * fires `input` synchronously, so `processInput` re-tokenizes the
	 * buffer and rebuilds when the pasted text carries badge or code
	 * tokens.
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
		processInput('deleteByCut');
	}

	onMount(() => {
		// untrack: the DOM is managed manually from input events, so the
		// initial render must not subscribe to the value.
		renderTokens(tokenizeContent(untrack(() => value)));
		lastEmittedValue = untrack(() => value ?? '');
		resizeHeight();
		syncEmptyState();
		document.addEventListener('selectionchange', handleSelectionChange);
		if (!isMobile.current) {
			rootElement?.focus({ preventScroll: true });
		}
	});

	onDestroy(() => {
		document.removeEventListener('selectionchange', handleSelectionChange);
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
	/* pre-wrap is load-bearing: without it Chromium collapses \n in
	   text nodes and converts them to spaces while typing */
	.chat-form-contenteditable {
		white-space: pre-wrap;
	}

	.chat-form-contenteditable:global([data-empty='true'])::before {
		content: attr(data-placeholder);
		color: var(--muted-foreground);
		pointer-events: none;
	}

	/* Inline code - mirrors markdown-content.css */
	.chat-form-contenteditable :global(code[data-code-token='inline']) {
		background: var(--muted);
		color: var(--muted-foreground);
		padding: 0.125rem 0.375rem;
		border-radius: 0.375rem;
		font-size: 0.875rem;
	}

	/* Fenced code block - mirrors .code-block-wrapper in markdown-content.css */
	.chat-form-contenteditable :global(code[data-code-token='block']) {
		display: block;
		margin: 0.25rem 0;
		padding: 0.75rem 1rem;
		border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);
		border-radius: 0.75rem;
		background: var(--code-background);
		color: var(--code-foreground);
		font-size: 0.875rem;
		line-height: 1.3;
	}
</style>
