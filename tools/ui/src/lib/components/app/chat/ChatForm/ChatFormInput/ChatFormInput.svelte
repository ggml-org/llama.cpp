<script lang="ts">
	import { fileMentionWidgets } from './mention-badge-widget';
	import { EditorView } from '@codemirror/view';
	import { MarkdownEditor } from '$lib/components/app';
	import { deviceStore } from '$lib/stores';
	import { onMount } from 'svelte';
	import type { ClassValue } from 'svelte/elements';

	interface Props {
		class?: ClassValue;
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
		disabled = false,
		expanded = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
	}: Props = $props();

	let editor: MarkdownEditor | undefined = $state();

	// CSS can not transition a height that follows its content (an `auto` height never
	// changes value), so the editor's content height is mirrored onto this wrapper as a
	// length the stylesheet transitions: the form grows and shrinks smoothly as lines
	// come and go, and scrolls inside once it reaches its maximum
	let contentHeight = $state<number | null>(null);

	$effect(() => {
		const content = editor?.getView()?.contentDOM;

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

	// The expanded editor is taller than its text: a click on the empty room below it
	// continues writing at the end, as in a textarea
	function handleMouseDown(event: MouseEvent) {
		if (!expanded || (event.target as HTMLElement).closest('.cm-content')) return;

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

	/** Cursor offset in the markdown source. */
	export function getCaretOffset(): number {
		return editor?.getCaretOffset() ?? 0;
	}

	export function setCaretOffset(offset: number) {
		editor?.focusAt(offset);
	}
</script>

<div
	style:height={contentHeight === null ? undefined : `${contentHeight}px`}
	class={[
		'chat-form-input mb-0.5',
		expanded && 'expanded',
		disabled && 'cursor-not-allowed',
		className
	]}
	onmousedown={handleMouseDown}
	role="presentation"
>
	<MarkdownEditor
		bind:this={editor}
		{disabled}
		editingWidgets={fileMentionWidgets}
		{extensions}
		onChange={handleChange}
		{placeholder}
		value={value ?? ''}
	/>
</div>

<style>
	/* Plain text reads exactly like the former textarea. The height (set from the
	   content) excludes the padding, which the form sets around it */
	.chat-form-input {
		display: flex;
		flex-direction: column;
		box-sizing: content-box;
		min-height: 2.25rem;
		max-height: var(--max-message-height);
		line-height: 1.5rem;
		transition: height 400ms cubic-bezier(0.3, 1.12, 0.5, 1);
	}

	/* Grow into the room the form gives the editor (expanded), never past it; the
	   minimum height holds in both states, so switching never snaps it */
	.chat-form-input {
		flex: 1 1 auto;
	}

	.chat-form-input.expanded {
		max-height: none;
	}

	@media (prefers-reduced-motion: reduce) {
		.chat-form-input {
			transition: none;
		}
	}

	/* The editor takes the wrapper's height and scrolls inside it */
	.chat-form-input :global(.markdown-content) {
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
