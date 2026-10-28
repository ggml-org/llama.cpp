<script lang="ts" module>
	import {
		cwdServerTools,
		globSearchResult,
		mentionEntries,
		pendingGlobSearch,
		seedStores,
		SERVER_HOME,
		toGlobEntries
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormPickerMention from '$lib/components/app/chat/ChatForm/ChatFormPickers/ChatFormPickerMention.svelte';
	import { fn } from 'storybook/test';
	import type { ComponentProps } from 'svelte';

	const { Story } = defineMeta({
		args: {
			isOpen: true,
			onClose: fn(),
			onSelect: fn(),
			query: 'chat',
			scopePath: SERVER_HOME
		},
		component: ChatFormPickerMention,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormPickerMention'
	});

	const searchTools = { serverHome: SERVER_HOME, serverTools: cwdServerTools };
</script>

{#snippet template(args: ComponentProps<typeof ChatFormPickerMention>)}
	<!-- the popover anchors to its parent and opens above it -->
	<div class="relative mt-80 h-12 w-[32rem] rounded-3xl border">
		<ChatFormPickerMention {...args} />
	</div>
{/snippet}

<Story
	beforeEach={() =>
		seedStores({ ...searchTools, globSearch: globSearchResult(toGlobEntries(mentionEntries)) })}
	name="Default"
	{template}
/>

<Story
	args={{ query: 'loading' }}
	beforeEach={() => seedStores({ ...searchTools, globSearch: pendingGlobSearch() })}
	name="Loading"
	{template}
/>

<Story
	args={{ query: 'nothing' }}
	beforeEach={() => seedStores({ ...searchTools, globSearch: globSearchResult([]) })}
	name="NoMatch"
	{template}
/>

<Story
	args={{ query: 'denied' }}
	beforeEach={() =>
		seedStores({
			...searchTools,
			globSearch: async () => ({ error: 'path is outside the allowed roots' })
		})}
	name="SearchError"
	{template}
/>

<Story
	args={{ query: 'readme' }}
	beforeEach={() => seedStores({ serverHome: SERVER_HOME, serverTools: [] })}
	name="SearchUnavailable"
	{template}
/>
