<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { chatCommands } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormPickerCommand from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickerCommand.svelte';
	import { fn } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatFormPickerCommand,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormPickerCommand'
	});

	const common = { commands: chatCommands, isOpen: true, onClose: fn(), onSelect: fn() };
</script>

<!--
	the popover is portalled and opens above its anchor, so each cell reserves
	room above a fixed-height anchor to keep the popovers apart
-->
{#snippet anchored(query: string)}
	<div class="flex h-96 flex-col justify-end">
		<div class="relative h-12 w-full rounded-3xl border">
			<ChatFormPickerCommand {...common} {query} />
		</div>
	</div>
{/snippet}

<Story asChild name="States">
	<StateGallery columns={3}>
		<StateGalleryItem label="All commands">
			{@render anchored('')}
		</StateGalleryItem>

		<StateGalleryItem label="Filtered" note="query: dir">
			{@render anchored('dir')}
		</StateGalleryItem>

		<StateGalleryItem label="No match" note="query: xyz">
			{@render anchored('xyz')}
		</StateGalleryItem>
	</StateGallery>
</Story>
