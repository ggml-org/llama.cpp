<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatScreen />
	</Sidebar.Provider>
</Tooltip.Provider>
