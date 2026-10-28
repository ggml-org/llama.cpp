<script lang="ts" module>
	import { SERVER_HOME } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormInput from '$lib/components/app/chat/ChatForm/ChatFormInput/ChatFormInput.svelte';

	const { Story } = defineMeta({
		args: {
			class: 'w-[32rem] px-5 py-1.5',
			placeholder: 'Type a message...'
		},
		component: ChatFormInput,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormInput'
	});

	const mentionValue = `Summarize [README.md](file://${SERVER_HOME}/llama.cpp/README.md) and [ui](file://${SERVER_HOME}/llama.cpp/tools/ui/) please`;
</script>

<Story name="Default" />

<Story
	args={{
		value:
			'Explain how the KV cache works in llama.cpp.\nKeep it short, and list the main flags that change its size.'
	}}
	name="WithText"
/>

<Story args={{ disabled: true, value: 'Waiting for the model to load' }} name="Disabled" />

<Story
	args={{ useRichInput: true, value: 'Why does `npm run check` fail after `npm install`?' }}
	name="RichWithCodeSpan"
/>

<Story args={{ useRichInput: true, value: mentionValue }} name="RichWithFileMention" />
