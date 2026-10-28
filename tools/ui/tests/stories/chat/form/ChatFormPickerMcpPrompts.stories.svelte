<script lang="ts" module>
	import { mcpPrompts, mcpServers, override, seedStores } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormPickerMcpPrompts from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
	import { mcpStore } from '$lib/stores';
	import { fn, screen } from 'storybook/test';
	import type { ComponentProps } from 'svelte';

	const { Story } = defineMeta({
		args: {
			isOpen: true,
			onClose: fn(),
			onPromptLoadComplete: fn(),
			onPromptLoadError: fn(),
			onPromptLoadStart: fn()
		},
		component: ChatFormPickerMcpPrompts,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormPickerMcpPrompts'
	});

	// keeps the picker in its loading state
	function seedLoading() {
		const restore = seedStores({ mcpServers });
		const restoreInit = override(
			mcpStore,
			'ensureInitialized',
			() => new Promise<boolean>(() => {})
		);

		return () => {
			restoreInit();
			restore();
		};
	}
</script>

{#snippet template(args: ComponentProps<typeof ChatFormPickerMcpPrompts>)}
	<!-- the popover anchors to its parent and opens above it -->
	<div class="relative mt-96 h-12 w-[32rem] rounded-3xl border">
		<ChatFormPickerMcpPrompts {...args} />
	</div>
{/snippet}

<Story beforeEach={() => seedStores({ mcpPrompts, mcpServers })} name="Default" {template} />

<Story
	beforeEach={() => seedStores({ mcpPrompts: mcpPrompts.slice(0, 2), mcpServers })}
	name="FewPrompts"
	{template}
/>

<Story
	args={{ searchQuery: 'review' }}
	beforeEach={() => seedStores({ mcpPrompts, mcpServers })}
	name="Filtered"
	{template}
/>

<Story beforeEach={seedLoading} name="Loading" {template} />

<Story beforeEach={() => seedStores({ mcpPrompts: [], mcpServers })} name="Empty" {template} />

<Story
	beforeEach={() => seedStores({ mcpPrompts, mcpServers })}
	name="ArgumentForm"
	play={async ({ userEvent }) => {
		await userEvent.click(await screen.findByText('Code review'));
		await screen.findByRole('button', { name: 'Use Prompt' });
	}}
	{template}
/>
