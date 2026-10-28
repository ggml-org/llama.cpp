<script lang="ts" module>
	import { chatCommands } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormPickerCommand from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickerCommand.svelte';
	import { fn } from 'storybook/test';
	import type { ComponentProps } from 'svelte';

	const { Story } = defineMeta({
		args: {
			commands: chatCommands,
			isOpen: true,
			onClose: fn(),
			onSelect: fn(),
			query: ''
		},
		component: ChatFormPickerCommand,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormPickerCommand'
	});
</script>

{#snippet template(args: ComponentProps<typeof ChatFormPickerCommand>)}
	<!-- the popover anchors to its parent and opens above it -->
	<div class="relative mt-80 h-12 w-[32rem] rounded-3xl border">
		<ChatFormPickerCommand {...args} />
	</div>
{/snippet}

<Story name="Default" {template} />

<Story args={{ query: 'dir' }} name="Filtered" {template} />

<Story args={{ query: 'xyz' }} name="NoMatch" {template} />
