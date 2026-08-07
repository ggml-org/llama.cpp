<script lang="ts">
	import { untrack } from 'svelte';
	import ChatFormContenteditable from '$lib/components/app/chat/ChatForm/ChatFormContenteditable.svelte';

	interface Props {
		value?: string;
	}

	let { value: initial = '' }: Props = $props();

	let value = $state(untrack(() => initial));
	let inputRef: ChatFormContenteditable | undefined = $state(undefined);

	export function getValue() {
		return value;
	}

	export function getCaretOffset() {
		return inputRef?.getCaretOffset();
	}

	export function setCaretOffset(offset: number) {
		inputRef?.setCaretOffset(offset);
	}
</script>

<ChatFormContenteditable bind:this={inputRef} bind:value />
