<script lang="ts">
	import autoResizeTextarea from '$lib/utils/autoresize-textarea';
	import { onMount } from 'svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
	}

	let {
		class: className = '',
		disabled = false,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		value = $bindable('')
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
			textareaElement.style.height = '1rem';
		}
	}
</script>

<div class="flex-1 {className}">
	<textarea
		bind:this={textareaElement}
		bind:value
		class="text-md max-h-32 min-h-12 w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0"
		class:cursor-not-allowed={disabled}
		{disabled}
		onkeydown={onKeydown}
		oninput={(event) => autoResizeTextarea(event.currentTarget)}
		onpaste={onPaste}
		{placeholder}
	></textarea>
</div>
