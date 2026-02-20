<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import ChatMessages from '$lib/components/app/chat/ChatMessages/ChatMessages.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	const props = $props<{ messages: DatabaseMessage[] }>();
	const messages = props.messages || [];
	conversationsStore.activeMessages = messages;
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatMessages {messages} />
	</Sidebar.Provider>
</Tooltip.Provider>
