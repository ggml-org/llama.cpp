<script lang="ts">
	import ChatScreenForm from '#lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import * as Tooltip from '#lib/components/ui/tooltip/index.ts';
</script>

<Tooltip.Provider>
	<ChatScreenForm />
</Tooltip.Provider>
