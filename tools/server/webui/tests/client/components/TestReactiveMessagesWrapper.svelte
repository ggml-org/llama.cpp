<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import ChatMessages from '$lib/components/app/chat/ChatMessages/ChatMessages.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';

	// Always pass a new array copy so downstream components re-render on store mutations
	const liveMessages = $derived.by(() => [...conversationsStore.activeMessages]);
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatMessages messages={liveMessages} />
	</Sidebar.Provider>
</Tooltip.Provider>
