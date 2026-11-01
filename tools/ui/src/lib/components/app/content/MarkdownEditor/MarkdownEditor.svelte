<script lang="ts">
	import '$lib/styles/katex-custom.scss';
	import { markdownLivePreview, type MarkdownLivePreviewOptions } from './markdown-live-preview';
	import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
	import {
		insertNewlineContinueMarkup,
		markdown,
		markdownKeymap,
		markdownLanguage
	} from '@codemirror/lang-markdown';
	import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
	import { Compartment, EditorState, type Extension, Transaction } from '@codemirror/state';
	import { EditorView, keymap, placeholder as placeholderExt } from '@codemirror/view';
	import { tags } from '@lezer/highlight';
	import { DialogCodePreview } from '$lib/components/app';
	import { ColorMode } from '$lib/enums';
	import { loadHighlightTheme } from '$lib/utils';
	import { mode } from 'mode-watcher';
	import { onDestroy, onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		/** Widgets for the raw text of the block being edited, see MarkdownLivePreviewOptions. */
		editingWidgets?: MarkdownLivePreviewOptions['editingWidgets'];
		/** Extra extensions, ahead of the default keymaps (e.g. their own key handling). */
		extensions?: Extension[];
		onChange?: (value: string) => void;
		placeholder?: string;
		value: string;
	}

	let {
		class: className = '',
		disabled = false,
		editingWidgets,
		extensions = [],
		onChange,
		placeholder = '',
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

	const editable = new Compartment();

	let host: HTMLDivElement;
	let view: EditorView | undefined = $state();
	let previewCode = $state('');
	let previewLanguage = $state('text');
	let previewOpen = $state(false);

	onMount(() => {
		view = new EditorView({
			parent: host,
			state: EditorState.create({
				doc: value,
				extensions: [
					...extensions,
					history(),
					// Shift+Enter is a plain line break that still continues lists and quotes
					keymap.of([{ key: 'Shift-Enter', run: insertNewlineContinueMarkup }]),
					keymap.of([...markdownKeymap, ...defaultKeymap, ...historyKeymap]),
					markdown({ base: markdownLanguage }),
					syntaxHighlighting(rawMarkdownStyle),
					markdownLivePreview({
						editingWidgets,
						onPreviewCode: (code, language) => {
							previewCode = code;
							previewLanguage = language;
							previewOpen = true;
						}
					}),
					editable.of(EditorView.editable.of(!disabled)),
					EditorView.lineWrapping,
					placeholderExt(placeholder),
					// Only report edits made by the user; syncs from `value` carry no user event
					EditorView.updateListener.of((update) => {
						const fromUser = update.transactions.some(
							(tr) => tr.annotation(Transaction.userEvent) !== undefined
						);

						if (update.docChanged && fromUser) onChange?.(update.state.doc.toString());
					})
				]
			})
		});
	});

	onDestroy(() => view?.destroy());

	/**
	 * Bring the editor in line with `value` (streamed tokens, mention inserts, a cleared
	 * form, ...). Only the changed middle is replaced, so the viewport and the selection
	 * around it survive; these edits stay out of the undo history.
	 */
	function syncValue(next: string) {
		if (!view) return;

		const current = view.state.doc.toString();

		if (next === current) return;

		let from = 0;

		while (from < current.length && from < next.length && current[from] === next[from]) from++;

		let currentTo = current.length;
		let nextTo = next.length;

		while (currentTo > from && nextTo > from && current[currentTo - 1] === next[nextTo - 1]) {
			currentTo--;
			nextTo--;
		}

		view.dispatch({
			annotations: Transaction.addToHistory.of(false),
			changes: { from, insert: next.slice(from, nextTo), to: currentTo }
		});
	}

	$effect(() => {
		syncValue(value);
	});

	$effect(() => {
		view?.dispatch({ effects: editable.reconfigure(EditorView.editable.of(!disabled)) });
	});

	$effect(() => {
		loadHighlightTheme(mode.current === ColorMode.DARK);
	});

	/** The editor view, once mounted. */
	export function getView(): EditorView | undefined {
		return view;
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

	export function focus() {
		view?.focus();
	}
</script>

<!-- markdown-content: rendered blocks get exactly the chat message styles -->
<div bind:this={host} class="markdown-content {className}"></div>

<DialogCodePreview
	code={previewCode}
	language={previewLanguage}
	onOpenChange={(open) => (previewOpen = open)}
	open={previewOpen}
/>

<style>
	@import '../MarkdownContent/markdown-content.css';

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
</style>
