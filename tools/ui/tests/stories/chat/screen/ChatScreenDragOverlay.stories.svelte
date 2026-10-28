<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenDragOverlay from '$lib/components/app/chat/ChatScreen/ChatScreenDragOverlay.svelte';

	const { Story } = defineMeta({
		component: ChatScreenDragOverlay,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Chat/ChatScreen/ChatScreenDragOverlay'
	});
</script>

<Story asChild name="Default">
	<div class="flex h-screen items-center justify-center p-8 text-muted-foreground">
		Page content behind the overlay
	</div>

	<ChatScreenDragOverlay />
</Story>
