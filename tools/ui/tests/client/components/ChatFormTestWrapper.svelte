<script lang="ts">
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';

	let { onSubmit }: { onSubmit?: () => void } = $props();

	let value = $state('');
</script>

<Tooltip.Provider>
	<ChatForm bind:value {onSubmit} />
</Tooltip.Provider>
