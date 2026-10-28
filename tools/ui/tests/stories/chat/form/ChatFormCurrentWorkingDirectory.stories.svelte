<script lang="ts" module>
	import {
		cwdServerTools,
		directoryResults,
		globSearchResult,
		pendingGlobSearch,
		seedStores,
		SERVER_HOME,
		type StoreSeed,
		toGlobEntries
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormCurrentWorkingDirectory from '$lib/components/app/chat/ChatForm/ChatFormCurrentWorkingDirectory/ChatFormCurrentWorkingDirectory.svelte';
	import { FileMentionEntryType } from '$lib/enums';
	import { fn } from 'storybook/test';
	import type { ComponentProps } from 'svelte';

	const { Story } = defineMeta({
		args: {
			directory: null,
			isOpen: false,
			onChange: fn(),
			onClose: fn(),
			onOpen: fn(),
			query: ''
		},
		component: ChatFormCurrentWorkingDirectory,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormCurrentWorkingDirectory'
	});

	const directoryEntries = toGlobEntries(
		directoryResults.map((path) => ({ path, type: FileMentionEntryType.DIRECTORY }))
	);

	function seedSearch(seed: StoreSeed = {}) {
		return () => seedStores({ serverHome: SERVER_HOME, serverTools: cwdServerTools, ...seed });
	}
</script>

{#snippet template(args: ComponentProps<typeof ChatFormCurrentWorkingDirectory>)}
	<!-- the picker anchors to the form; the chip sits below it -->
	<div class="relative mt-80 grid w-[32rem]">
		<div class="h-12 rounded-3xl border"></div>

		<ChatFormCurrentWorkingDirectory {...args} />
	</div>
{/snippet}

<Story beforeEach={seedSearch()} name="Default" {template} />

<Story
	args={{ directory: `${SERVER_HOME}/llama.cpp/tools/ui` }}
	beforeEach={seedSearch()}
	name="WithDirectory"
	{template}
/>

<Story
	args={{ directory: `${SERVER_HOME}/llama.cpp/tools/ui`, disabled: true }}
	beforeEach={seedSearch()}
	name="Disabled"
	{template}
/>

<Story args={{ isOpen: true }} beforeEach={seedSearch()} name="PickerOpen" {template} />

<Story
	args={{ isOpen: true, query: 'llama' }}
	beforeEach={seedSearch({ globSearch: globSearchResult(directoryEntries) })}
	name="PickerWithResults"
	{template}
/>

<Story
	args={{ isOpen: true, query: 'searching' }}
	beforeEach={seedSearch({ globSearch: pendingGlobSearch() })}
	name="PickerSearching"
	{template}
/>

<Story
	args={{ isOpen: true, query: 'nowhere' }}
	beforeEach={seedSearch({ globSearch: globSearchResult([]) })}
	name="PickerNoMatch"
	{template}
/>

<Story
	args={{ isOpen: true, query: 'root' }}
	beforeEach={seedSearch({
		globSearch: async () => ({ error: 'path is outside the allowed roots' })
	})}
	name="PickerSearchError"
	{template}
/>

<Story
	args={{ isOpen: true }}
	beforeEach={seedSearch({ serverTools: [] })}
	name="PickerSearchUnavailable"
	{template}
/>
