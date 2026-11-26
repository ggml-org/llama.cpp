<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatForm from '#lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { untrack } from 'svelte';

	interface Props {
		expandable?: boolean;
		/** The message the form starts with, as a restored draft would be. */
		initialValue?: string;
		/** A reply is streaming: the form offers stop in place of send. */
		isLoading?: boolean;
		onStop?: () => void;
		onSubmit?: () => void;
		singleRow?: boolean;
		systemMode?: boolean;
	}

	let {
		expandable = false,
		initialValue = '',
		isLoading = false,
		onStop,
		onSubmit,
		singleRow = false,
		systemMode = false
	}: Props = $props();

	let value = $state(untrack(() => initialValue));
	let form: ChatForm | undefined = $state();

	export function expandForMessage() {
		form?.expandForMessage();
	}
</script>

<Tooltip.Provider>
	<ChatForm
		bind:this={form}
		bind:value
		{expandable}
		{isLoading}
		{onStop}
		{onSubmit}
		{singleRow}
		{systemMode}
	/>
</Tooltip.Provider>
