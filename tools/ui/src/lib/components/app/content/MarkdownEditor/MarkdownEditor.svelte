<script lang="ts" module>
	import { EditorView } from '@codemirror/view';

	/**
	 * Takes text input through contenteditable on every browser: Chrome on Android opens no
	 * virtual keyboard for an editor that takes it through EditContext.
	 */
	class ContentEditableView extends EditorView {
		static EDIT_CONTEXT = false;
	}
</script>

<script lang="ts">
	import '$lib/styles/katex-custom.scss';
	import MarkdownSurface from '../MarkdownContent/MarkdownSurface.svelte';
	import {
		markdownLivePreview,
		type MarkdownLivePreviewOptions,
		markdownPlainText
	} from './markdown-live-preview';
	import {
		defaultKeymap,
		history,
		historyKeymap,
		insertNewlineAndIndent
	} from '@codemirror/commands';
	import {
		insertNewlineContinueMarkup,
		markdownKeymap,
		markdownLanguage
	} from '@codemirror/lang-markdown';
	import { HighlightStyle, LanguageSupport, syntaxHighlighting } from '@codemirror/language';
	import {
		Annotation,
		Compartment,
		EditorState,
		type Extension,
		Transaction
	} from '@codemirror/state';
	import { keymap, placeholder as placeholderExt } from '@codemirror/view';
	import { tags } from '@lezer/highlight';
	import { DialogCodePreview, DialogMermaidPreview } from '$lib/components/app';
	import { ColorMode } from '$lib/enums';
	import { loadHighlightTheme } from '$lib/utils';
	import { mode } from 'mode-watcher';
	import { onDestroy, onMount, untrack } from 'svelte';

	interface Props {
		/** The editor's accessible name; the placeholder when not given. */
		ariaLabel?: string;
		class?: string;
		disabled?: boolean;
		/** Widgets for the raw text of the block being edited, see MarkdownLivePreviewOptions. */
		editingWidgets?: MarkdownLivePreviewOptions['editingWidgets'];
		/** Extra extensions, ahead of the default keymaps (e.g. their own key handling). */
		extensions?: Extension[];
		/** Render the markdown as it is written; off, the editor shows the plain text. */
		livePreview?: boolean;
		onChange?: (value: string) => void;
		placeholder?: string;
		/** What the rendered blocks depend on besides their text, see MarkdownLivePreviewOptions. */
		renderKey?: string;
		value: string;
	}

	let {
		ariaLabel,
		class: className = '',
		disabled = false,
		editingWidgets,
		extensions = [],
		livePreview = true,
		onChange,
		placeholder = '',
		renderKey = '',
		value
	}: Props = $props();

	// The raw block being edited keeps the typography of its rendered version
	// (sizes and weights mirror markdown-content.css)
	const rawMarkdownStyle = HighlightStyle.define([
		{ fontSize: '1.875rem', fontWeight: '700', lineHeight: '1.2', tag: tags.heading1 },
		{ fontSize: '1.5rem', fontWeight: '600', lineHeight: '1.3', tag: tags.heading2 },
		{ fontSize: '1.25rem', fontWeight: '600', lineHeight: '1.4', tag: tags.heading3 },
		{ fontSize: '1.125rem', fontWeight: '600', tag: tags.heading4 },
		{ fontSize: '1rem', fontWeight: '600', tag: tags.heading5 },
		{ fontSize: '0.875rem', fontWeight: '600', tag: tags.heading6 },
		{ fontWeight: '600', tag: tags.strong },
		{ fontStyle: 'italic', tag: tags.emphasis },
		{ color: 'var(--muted-foreground)', fontStyle: 'italic', tag: tags.quote },
		{ opacity: '0.7', tag: tags.strikethrough, textDecoration: 'line-through' },
		{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', tag: tags.monospace }
	]);

	// The markdown language alone: markdown() would also bring the html, css and
	// javascript parsers (for html tag completion), which the editor doesn't use
	const markdownSupport = new LanguageSupport(markdownLanguage);

	// Props can change after the editor is created: each lives in a compartment
	const customCompartment = new Compartment();
	const editableCompartment = new Compartment();
	const labelsCompartment = new Compartment();
	const previewCompartment = new Compartment();
	// Marks the edits that bring the editor in line with `value`, which onChange skips
	const valueSync = Annotation.define<boolean>();

	let host: HTMLDivElement | undefined = $state();
	// reactive, so what follows the editor's DOM (see getContentDOM) starts once it exists
	let view = $state.raw<EditorView>();
	let previewCode = $state('');
	let previewLanguage = $state('text');
	let previewOpen = $state(false);
	let diagramSvg = $state('');
	let diagramOpen = $state(false);
	// The document's text as of the last edit or sync, so syncing an unchanged `value`
	// (each keystroke comes back through onChange) costs no serialization
	let docText = '';

	function labels(): Extension {
		return [
			placeholderExt(placeholder),
			// the content is a textbox with no name of its own otherwise
			EditorView.contentAttributes.of({ 'aria-label': ariaLabel || placeholder })
		];
	}

	function previewExtension(): Extension {
		if (!livePreview) return markdownPlainText(editingWidgets);

		return [
			syntaxHighlighting(rawMarkdownStyle),
			markdownLivePreview({
				editingWidgets,
				onPreviewCode: (code, language) => {
					previewCode = code;
					previewLanguage = language;
					previewOpen = true;
				},
				onPreviewDiagram: (svgHtml) => {
					diagramSvg = svgHtml;
					diagramOpen = true;
				},
				renderKey
			})
		];
	}

	onMount(() => {
		if (!host) return;

		docText = value;
		view = new ContentEditableView({
			parent: host,
			state: EditorState.create({
				doc: value,
				extensions: [
					customCompartment.of(extensions),
					history(),
					// Shift+Enter and Ctrl/Cmd+Enter add a line, continuing lists and quotes. Mod is
					// Ctrl off macOS, where the chat form still takes Meta+Enter as Ctrl+Enter
					keymap.of(
						['Shift-Enter', 'Mod-Enter', 'Ctrl-Enter', 'Meta-Enter'].map((key) => ({
							key,
							run: (target) => insertNewlineContinueMarkup(target) || insertNewlineAndIndent(target)
						}))
					),
					keymap.of([...markdownKeymap, ...defaultKeymap, ...historyKeymap]),
					markdownSupport,
					previewCompartment.of(previewExtension()),
					editableCompartment.of(EditorView.editable.of(!disabled)),
					EditorView.lineWrapping,
					labelsCompartment.of(labels()),
					// Report every edit but the ones syncing from `value`
					EditorView.updateListener.of((update) => {
						const edited = update.transactions.some(
							(tr) => tr.docChanged && !tr.annotation(valueSync)
						);

						if (!edited) return;

						docText = update.state.doc.toString();
						onChange?.(docText);
					})
				]
			})
		});
	});

	onDestroy(() => view?.destroy());

	function reconfigure(compartment: Compartment, extension: Extension) {
		view?.dispatch({ effects: compartment.reconfigure(extension) });
	}

	/** Runs `apply` when a value `read` gives changes: not on mount, the editor starts with them. */
	function whenChanged(read: () => unknown[], apply: () => void) {
		let last = read();

		$effect(() => {
			const next = read();

			if (next.every((value, index) => value === last[index])) return;

			last = next;
			untrack(apply);
		});
	}

	/** The single change that turns `current` into `next`: their differing middle. */
	function changeBetween(current: string, next: string) {
		let from = 0;

		while (from < current.length && from < next.length && current[from] === next[from]) from++;

		let currentTo = current.length;
		let nextTo = next.length;

		while (currentTo > from && nextTo > from && current[currentTo - 1] === next[nextTo - 1]) {
			currentTo--;
			nextTo--;
		}

		return { from, insert: next.slice(from, nextTo), to: currentTo };
	}

	/**
	 * Bring the editor in line with `value` (streamed tokens, a cleared form, ...). Only
	 * the changed middle is replaced, so the viewport and the selection around it
	 * survive; these edits stay out of the undo history and out of onChange.
	 */
	function syncValue(next: string) {
		if (!view || next === docText) return;

		const current = docText;

		docText = next;
		view.dispatch({
			annotations: [Transaction.addToHistory.of(false), valueSync.of(true)],
			changes: changeBetween(current, next)
		});
	}

	$effect(() => {
		syncValue(value);
	});

	whenChanged(
		() => [extensions],
		() => reconfigure(customCompartment, extensions)
	);
	whenChanged(
		() => [disabled],
		() => reconfigure(editableCompartment, EditorView.editable.of(!disabled))
	);
	whenChanged(
		() => [ariaLabel, placeholder],
		() => reconfigure(labelsCompartment, labels())
	);
	// a new preview renders the blocks again when the render key is another one
	whenChanged(
		() => [editingWidgets, livePreview, renderKey],
		() => reconfigure(previewCompartment, previewExtension())
	);

	$effect(() => {
		loadHighlightTheme(mode.current === ColorMode.DARK);
	});

	/** The editor's content element, once mounted (e.g. to follow its height). */
	export function getContentDOM(): HTMLElement | undefined {
		return view?.contentDOM;
	}

	/** Where the cursor is on screen, e.g. to place a popover by it. */
	export function getCaretRect() {
		return view?.coordsAtPos(view.state.selection.main.head) ?? null;
	}

	/** Cursor offset in the markdown source. */
	export function getCaretOffset(): number {
		return view?.state.selection.main.head ?? 0;
	}

	/** Focus the editor with the cursor at `offset` (default: the end of the text). */
	export function focusAt(offset?: number) {
		if (!view) return;

		// the caller may have just changed `value`: place the cursor in the new text
		syncValue(value);

		const length = view.state.doc.length;

		view.dispatch({ selection: { anchor: Math.min(offset ?? length, length) } });
		view.focus();
	}

	/**
	 * Replace the text with `next` as one edit the user can undo (e.g. a picked mention),
	 * focused with the cursor at `caret`; onChange reports it like typing.
	 */
	export function replaceText(next: string, caret = next.length) {
		if (!view) return;

		view.dispatch({
			changes: next === docText ? undefined : changeBetween(docText, next),
			selection: { anchor: Math.min(caret, next.length) },
			userEvent: 'input.complete'
		});
		view.focus();
	}

	export function focus() {
		view?.focus();
	}
</script>

<!-- markdown-content: rendered blocks get exactly the chat message styles -->
<div class={className}>
	<MarkdownSurface bind:ref={host} />
</div>

<DialogCodePreview
	code={previewCode}
	language={previewLanguage}
	onOpenChange={(open) => (previewOpen = open)}
	open={previewOpen}
/>

<DialogMermaidPreview
	onOpenChange={(open) => (diagramOpen = open)}
	open={diagramOpen}
	svgHtml={diagramSvg}
/>

<style>
	div :global(.cm-editor) {
		background: transparent;
		color: inherit;
	}

	div :global(.cm-editor.cm-focused) {
		outline: none;
	}

	div :global(.cm-scroller) {
		font-family: inherit;
		line-height: inherit;
	}

	div :global(.cm-content) {
		padding: 0;
		caret-color: var(--foreground);
	}

	div :global(.cm-line) {
		padding: 0;
	}

	div :global(.cm-placeholder) {
		color: var(--muted-foreground);
		text-shadow: none;
	}

	/* The markdown img styles must not reach the editor's invisible caret helpers */
	div :global(img.cm-widgetBuffer) {
		margin: 0 !important;
		box-shadow: none;
		height: 1em;
		width: 0;
	}

	/* Rendered block: margins come from the computed gap, so drop the outer ones */
	div :global(.cm-md-block) {
		display: flow-root;
		/* the editor's pre-wrap would turn the newlines between tags into blank lines */
		white-space: normal;
		cursor: text;
	}

	div :global(.cm-md-block > :first-child) {
		margin-top: 0 !important;
	}

	div :global(.cm-md-block > :last-child) {
		margin-bottom: 0 !important;
	}

	/* Raw code block being edited: the same box as a rendered code block */
	div :global(.cm-md-raw-code) {
		padding-inline: 1rem;
		font-family: var(--font-mono);
		font-size: 0.875rem;
		background: var(--code-background);
		border-inline: 1px solid color-mix(in oklch, var(--border) 30%, transparent);
	}

	div :global(.cm-md-raw-code-first) {
		padding-top: 0.75rem;
		border-top: 1px solid color-mix(in oklch, var(--border) 30%, transparent);
		border-radius: 0.75rem 0.75rem 0 0;
	}

	div :global(.cm-md-raw-code-last) {
		padding-bottom: 0.75rem;
		border-bottom: 1px solid color-mix(in oklch, var(--border) 30%, transparent);
		border-radius: 0 0 0.75rem 0.75rem;
	}

	/* A link reference definition renders to nothing, so it stays raw and muted */
	div :global(.cm-md-raw-definition) {
		color: var(--muted-foreground);
		font-size: 0.875rem;
	}
</style>
