<script lang="ts">
	import { fileMentionWidgets } from './mention-badge-widget';
	import { EditorView } from '@codemirror/view';
	import { MarkdownEditor } from '$lib/components/app';
	import { deviceStore } from '$lib/stores';
	import { onMount } from 'svelte';

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

	let editor: MarkdownEditor | undefined = $state();

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
	class={['chat-form-input mb-0.5 min-h-12 flex-1', disabled && 'cursor-not-allowed', className]}
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
	/* Plain text reads exactly like the former textarea */
	.chat-form-input {
		line-height: 1.5rem;
	}

	.chat-form-input :global(.cm-scroller) {
		max-height: var(--max-message-height);
		overflow-y: auto;
	}

	/* Rendered paragraphs and lists keep the line height of the text being typed */
	.chat-form-input :global(.cm-md-block :is(p, li)) {
		line-height: inherit;
	}
</style>
