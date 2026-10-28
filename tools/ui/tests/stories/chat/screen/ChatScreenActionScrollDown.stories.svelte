<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenActionScrollDown from '$lib/components/app/chat/ChatScreen/ChatScreenActionScrollDown.svelte';

	const { Story } = defineMeta({
		component: ChatScreenActionScrollDown,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatScreenActionScrollDown'
	});
</script>

<!-- the button floats above its zero-height anchor, so give it room -->
<Story asChild name="Default">
	<div class="flex h-32 w-80 flex-col justify-end rounded-lg border border-dashed border-border">
		<ChatScreenActionScrollDown onclick={() => console.log('scroll to bottom')} />
	</div>
</Story>
