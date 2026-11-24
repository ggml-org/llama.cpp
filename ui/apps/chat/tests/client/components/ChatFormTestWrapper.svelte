<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import ChatForm from '#lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { untrack } from 'svelte';

	interface Props {
		expandable?: boolean;
		/** The message the form starts with, as a restored draft would be. */
		initialValue?: string;
		onSubmit?: () => void;
		singleRow?: boolean;
	}

	let { expandable = false, initialValue = '', onSubmit, singleRow = false }: Props = $props();

	let value = $state(untrack(() => initialValue));
	let form: ChatForm | undefined = $state();

	export function expandForMessage() {
		form?.expandForMessage();
	}
</script>

<Tooltip.Provider>
	<ChatForm bind:this={form} bind:value {expandable} {onSubmit} {singleRow} />
</Tooltip.Provider>
