<script lang="ts">
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { onMount } from 'svelte';

	interface Props {
		value?: string;
		disabled?: boolean;
		placeholder?: string;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		class?: string;
	}

	let {
		value = $bindable(''),
		disabled = false,
		placeholder = 'Ask anything...',
		onKeydown,
		onPaste,
		class: className = ''
	}: Props = $props();

	let textareaElement: HTMLTextAreaElement | undefined;

	onMount(() => {
		if (textareaElement) {
			textareaElement.focus();
		}
	});

	// Expose the textarea element for external access
	export function getElement() {
		return textareaElement;
	}

	export function focus() {
		textareaElement?.focus();
	}

	export function resetHeight() {
		if (textareaElement) {
			textareaElement.style.height = 'auto';
		}
	}
</script>

<div class="flex-1 {className}">
	<textarea
		bind:this={textareaElement}
		bind:value
		onkeydown={onKeydown}
		oninput={(event) => autoResizeTextarea(event.currentTarget)}
		onpaste={onPaste}
		{placeholder}
		class="placeholder:text-muted-foreground text-md max-h-32 min-h-[24px] w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0"
		{disabled}
	></textarea>
</div>
