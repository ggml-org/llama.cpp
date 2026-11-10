<script lang="ts">
	import { fileMentionWidgets } from './mention-badge-widget';
	import { EditorView } from '@codemirror/view';
	import { MarkdownEditor } from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { deviceStore } from '$lib/stores';
	import { settingsStore } from '$lib/stores/settings/index.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { onMount } from 'svelte';
	import type { ClassValue } from 'svelte/elements';

	interface Props {
		class?: ClassValue;
		/** One line tall, beside the actions instead of above them. */
		compact?: boolean;
		/** The height of the text, which the form reads to know when it fills a line. */
		contentHeight?: number | null;
		disabled?: boolean;
		/** Fill the expanded chat form: the editor takes all its height and scrolls inside. */
		expanded?: boolean;
		onInput?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
	}

	let {
		class: className = '',
		compact = false,
		contentHeight = $bindable(null),
		disabled = false,
		expanded = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
	}: Props = $props();

	let editor: MarkdownEditor | undefined = $state();
	let box: HTMLDivElement | undefined = $state();

	let livePreview = $derived(settingsStore.config[SETTINGS_KEYS.LIVE_MARKDOWN_PREVIEW] !== false);

	// Rendered mention badges label paths by these, so the blocks render again on a change
	let renderKey = $derived(
		`${settingsStore.config[SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS]}|${toolsStore.serverHome ?? ''}`
	);

	// CSS can not transition a height that follows its content, so the editor's content
	// height is mirrored onto this wrapper as a length the stylesheet transitions
	$effect(() => {
		const content = editor?.getContentDOM();

		if (!content) return;

		const observer = new ResizeObserver(() => (contentHeight = content.offsetHeight));

		observer.observe(content);

		return () => observer.disconnect();
	});

	// The form gets the keys and pastes first: whatever it handles (submit, pickers,
	// pasted files) never reaches the editor
	const extensions = [
		EditorView.domEventHandlers({
			keydown: (event) => {
				onKeydown?.(event);

				return event.defaultPrevented;
			},
			paste: (event) => {
				onPaste?.(event);

				return event.defaultPrevented;
			}
		})
	];

	function handleChange(next: string) {
		value = next;
		onInput?.();
	}

	// The box is larger than its text (padding, and the room below it when expanded): a press
	// on it outside the text continues writing at the end, as in a textarea. A press on a
	// scrollbar, past the target's client width, still drags it
	function handleMouseDown(event: MouseEvent) {
		const content = editor?.getContentDOM();
		const target = event.target as HTMLElement;

		if (
			event.button !== 0 ||
			disabled ||
			!content ||
			content.contains(target) ||
			event.offsetX >= target.clientWidth
		) {
			return;
		}

		event.preventDefault();
		editor?.focusAt();
	}

	onMount(() => {
		if (!deviceStore.isMobile) editor?.focus();
	});

	export function focus() {
		if (deviceStore.isMobile) return;

		editor?.focus();
	}

	/** The element that sets the editor's height. */
	export function getBox() {
		return box;
	}

	export function getContentDOM() {
		return editor?.getContentDOM();
	}

	export function getCaretRect() {
		return editor?.getCaretRect() ?? null;
	}

	/** Cursor offset in the markdown source. */
	export function getCaretOffset(): number {
		return editor?.getCaretOffset() ?? 0;
	}

	export function setCaretOffset(offset: number) {
		editor?.focusAt(offset);
	}

	/** Replace the text as one undoable edit with the cursor at `caret` (e.g. a picked mention). */
	export function replaceText(next: string, caret: number) {
		if (editor) {
			editor.replaceText(next, caret);
		} else {
			handleChange(next);
		}
	}
</script>

<div
	bind:this={box}
	style:height={contentHeight === null ? undefined : `${contentHeight}px`}
	class={[
		'chat-form-input',
		compact ? 'compact' : 'mb-0.5',
		expanded && 'expanded',
		disabled && 'cursor-not-allowed',
		className
	]}
	onmousedown={handleMouseDown}
	role="presentation"
>
	<MarkdownEditor
		bind:this={editor}
		class="chat-form-editor"
		{disabled}
		editingWidgets={fileMentionWidgets}
		{extensions}
		{livePreview}
		onChange={handleChange}
		{placeholder}
		{renderKey}
		value={value ?? ''}
	/>
</div>

<style>
	/* Plain text reads exactly like the former textarea. The height (set from the
	   content) excludes the padding, which the form sets around it; expanded, it grows
	   into the room the form gives the editor */
	.chat-form-input {
		display: flex;
		flex: 1 1 auto;
		flex-direction: column;
		box-sizing: content-box;
		min-height: 2.25rem;
		max-height: var(--max-message-height);
		line-height: 1.5rem;
		transition: height 400ms cubic-bezier(0.3, 1.12, 0.5, 1);
	}

	.chat-form-input.expanded {
		max-height: none;
	}

	.chat-form-input.compact {
		min-height: 1.5rem;
	}

	@media (prefers-reduced-motion: reduce) {
		.chat-form-input {
			transition: none;
		}
	}

	/* Every box down to the editor takes the wrapper's height, so the editor scrolls inside it */
	.chat-form-input :global(:is(.chat-form-editor, .markdown-content)) {
		display: flex;
		flex: 1;
		flex-direction: column;
		min-height: 0;
	}

	.chat-form-input :global(.cm-editor) {
		flex: 1;
		min-height: 0;
	}

	.chat-form-input :global(.cm-scroller) {
		overflow-y: auto;
	}

	/* The content keeps the height of its text: CodeMirror stretches it over the whole
	   editor by default, which would feed the editor's own height back as the content
	   height and hold the form at its expanded size */
	.chat-form-input :global(.cm-content) {
		min-height: 0;
	}

	/* Rendered paragraphs and lists keep the line height of the text being typed */
	.chat-form-input :global(.cm-md-block :is(p, li)) {
		line-height: inherit;
	}
</style>
