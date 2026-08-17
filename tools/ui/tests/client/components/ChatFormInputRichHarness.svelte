<script lang="ts">
	import ChatFormInputRich from '$lib/components/app/chat/ChatForm/ChatFormInput/ChatFormInputRich.svelte';
	import { untrack } from 'svelte';

	interface Props {
		value?: string;
	}

	let { value: initial = '' }: Props = $props();

	let value = $state(untrack(() => initial));
	let inputRef: ChatFormInputRich | undefined = $state(undefined);

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

<ChatFormInputRich bind:this={inputRef} bind:value />
