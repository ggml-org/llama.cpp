<script lang="ts">
	import ChatFormContentEditable from '$lib/components/app/chat/ChatForm/ChatFormContentEditable.svelte';
	import { untrack } from 'svelte';

	interface Props {
		value?: string;
	}

	let { value: initial = '' }: Props = $props();

	let value = $state(untrack(() => initial));
	let inputRef: ChatFormContentEditable | undefined = $state(undefined);

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

<ChatFormContentEditable bind:this={inputRef} bind:value />
