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
		useContenteditable?: boolean;
	}

	let {
		class: className = '',
		disabled = false,
		onInput,
		onKeydown,
		onPaste,
		placeholder = 'Ask anything...',
		useContenteditable = false,
		value = $bindable('')
	}: Props = $props();

	let basicRef: ChatFormInputBasic | undefined = $state();
	let richRef: ChatFormInputRich | undefined = $state();

	// The two renderers share one imperative handle (focus/caret/height), so
	// the parent can drive whichever variant is mounted through this one.
	export function getElement() {
		return useContenteditable ? richRef?.getElement() : basicRef?.getElement();
	}

	export function focus() {
		if (useContenteditable) richRef?.focus();
		else basicRef?.focus();
	}

	export function resetHeight() {
		if (useContenteditable) richRef?.resetHeight();
		else basicRef?.resetHeight();
	}

	export function getCaretOffset(): number {
		return useContenteditable
			? (richRef?.getCaretOffset() ?? 0)
			: (basicRef?.getCaretOffset() ?? 0);
	}

	export function setCaretOffset(offset: number) {
		if (useContenteditable) richRef?.setCaretOffset(offset);
		else basicRef?.setCaretOffset(offset);
	}
</script>

{#if useContenteditable}
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
