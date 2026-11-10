<script lang="ts">
	import { fileMentionWidgets } from './mention-badge-widget';
	import { EditorView } from '@codemirror/view';
	import { MarkdownEditor } from '$lib/components/app';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { deviceStore } from '$lib/stores';
	import { settingsStore } from '$lib/stores/settings/index.svelte';
	import { toolsStore } from '$lib/stores/tools.svelte';
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

	let livePreview = $derived(settingsStore.config[SETTINGS_KEYS.LIVE_MARKDOWN_PREVIEW] !== false);

	// Rendered mention badges label paths by these, so the blocks render again on a change
	let renderKey = $derived(
		`${settingsStore.config[SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS]}|${toolsStore.serverHome ?? ''}`
	);

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

	/** Is the cursor in a code block, where Enter adds a line? */
	export function isCaretInCodeBlock(): boolean {
		return editor?.isCaretInCodeBlock() ?? false;
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
	class={[
		'chat-form-input mb-0.5 min-h-[2.375rem] flex-1',
		disabled && 'cursor-not-allowed',
		className
	]}
>
	<MarkdownEditor
		bind:this={editor}
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
