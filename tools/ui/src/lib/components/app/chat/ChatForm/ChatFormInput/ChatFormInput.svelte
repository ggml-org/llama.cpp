<script lang="ts">
	import ChatFormInputBasic from './ChatFormInputBasic.svelte';
	import ChatFormInputRich from './ChatFormInputRich.svelte';

	interface Props {
		class?: string;
		disabled?: boolean;
		onInput?: () => void;
		onKeydown?: (event: KeyboardEvent) => void;
		onPaste?: (event: ClipboardEvent) => void;
		placeholder?: string;
		value?: string;
		useRichInput?: boolean;
	}

	let {
		class: className = '',
		disabled = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		useRichInput = false,
		value = $bindable('')
	}: Props = $props();

	let basicRef: ChatFormInputBasic | undefined = $state();
	let richRef: ChatFormInputRich | undefined = $state();

	// The two renderers share one imperative handle (focus/caret/height), so
	// the parent can drive whichever variant is mounted through this one.
	export function getElement() {
		return useRichInput ? richRef?.getElement() : basicRef?.getElement();
	}

	export function focus() {
		if (useRichInput) richRef?.focus();
		else basicRef?.focus();
	}

	export function resetHeight() {
		if (useRichInput) richRef?.resetHeight();
		else basicRef?.resetHeight();
	}

	export function getCaretOffset(): number {
		return useRichInput ? (richRef?.getCaretOffset() ?? 0) : (basicRef?.getCaretOffset() ?? 0);
	}

	export function setCaretOffset(offset: number) {
		if (useRichInput) richRef?.setCaretOffset(offset);
		else basicRef?.setCaretOffset(offset);
	}
</script>

{#if useRichInput}
	<ChatFormInputRich
		bind:this={richRef}
		class={className}
		{disabled}
		{onInput}
		{onKeydown}
		{onPaste}
		{placeholder}
		bind:value
	/>
{:else}
	<ChatFormInputBasic
		bind:this={basicRef}
		class={className}
		{disabled}
		{onInput}
		{onKeydown}
		{onPaste}
		{placeholder}
		bind:value
	/>
{/if}
