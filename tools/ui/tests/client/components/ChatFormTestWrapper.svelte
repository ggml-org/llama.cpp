<script lang="ts">
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { untrack } from 'svelte';

	interface Props {
		expandable?: boolean;
		/** The message the form starts with, as a restored draft would be. */
		initialValue?: string;
		onSubmit?: () => void;
	}

	let { expandable = false, initialValue = '', onSubmit }: Props = $props();

	let value = $state(untrack(() => initialValue));
</script>

<Tooltip.Provider>
	<ChatForm bind:value {expandable} {onSubmit} />
</Tooltip.Provider>
