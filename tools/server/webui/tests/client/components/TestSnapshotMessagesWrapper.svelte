<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import ChatMessages from '$lib/components/app/chat/ChatMessages/ChatMessages.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	// Snapshot once at mount, non-reactive (mirrors messages={activeMessages()} usage)
	const snapshot = conversationsStore.activeMessages;
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatMessages messages={snapshot} />
	</Sidebar.Provider>
</Tooltip.Provider>
