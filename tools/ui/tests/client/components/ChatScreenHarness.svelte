<script lang="ts">
	import '../../../src/app.css';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
</script>

<!-- the chat screen alone, with the providers of +layout.svelte -->
<Tooltip.Provider>
	<ChatScreen />
</Tooltip.Provider>
