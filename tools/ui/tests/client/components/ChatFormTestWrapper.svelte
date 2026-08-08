<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';

	let { onSubmit }: { onSubmit?: () => void } = $props();

	let value = $state('');
</script>

<Tooltip.Provider>
	<ChatForm bind:value {onSubmit} />
</Tooltip.Provider>
