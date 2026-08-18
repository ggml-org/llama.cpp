<script lang="ts">
	import { deviceStore } from '$lib/stores';
	import { autoResizeTextarea } from '$lib/utils';
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

	let textareaElement: HTMLTextAreaElement | undefined;

	onMount(() => {
		if (textareaElement) {
			autoResizeTextarea(textareaElement);
			textareaElement.focus({ preventScroll: true });
		}
	});

	export function getElement() {
		return textareaElement;
	}

	export function focus() {
		if (deviceStore.isMobile) return;

		textareaElement?.focus({ preventScroll: true });
	}

	export function resetHeight() {
		if (textareaElement) {
			textareaElement.style.height = '1rem';
		}
	}

	// Plain-text caret offsets, shared with the rich chat form input variant so
	// the picker/paste flows can address either renderer through one handle.
	export function getCaretOffset(): number {
		if (!textareaElement) return 0;

		return textareaElement.selectionStart ?? textareaElement.value.length;
	}

	export function setCaretOffset(offset: number) {
		textareaElement?.setSelectionRange(offset, offset);
	}
</script>

<div class="flex-1 {className}">
	<textarea
		bind:this={textareaElement}
		bind:value
		class={[
			'text-md min-h-12 w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0',
			disabled && 'cursor-not-allowed'
		]}
		style="max-height: var(--max-message-height);"
		{disabled}
		onkeydown={onKeydown}
		oninput={(event) => {
			autoResizeTextarea(event.currentTarget);
			onInput?.();
		}}
		onpaste={onPaste}
		{placeholder}
	></textarea>
</div>
