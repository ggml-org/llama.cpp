<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage.svelte';

	const props = $props<{
		message: DatabaseMessage;
		onDelete: (message: DatabaseMessage) => void;
	}>();
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatMessage message={props.message} onDelete={props.onDelete} />
	</Sidebar.Provider>
</Tooltip.Provider>
