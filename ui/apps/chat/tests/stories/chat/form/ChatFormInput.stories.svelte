<script lang="ts" module>
	import { SERVER_HOME } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormInput from '#lib/components/app/chat/ChatForm/ChatFormInput/ChatFormInput.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: ChatFormInput,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatForm/ChatFormInput'
	});

	const common = {
		class: 'w-full px-5 py-1.5',
		placeholder: 'Type a message...'
	};

	const multilineValue =
		'Explain how the KV cache works in llama.cpp.\nKeep it short, and list the main flags that change its size.';

	const mentionValue = `Summarize [README.md](file://${SERVER_HOME}/llama.cpp/README.md) and [chat](file://${SERVER_HOME}/llama.cpp/ui/apps/chat/) please`;
</script>

<Story asChild name="States">
	<StateGallery columns={2}>
		<StateGalleryItem label="Empty" note="placeholder">
			<ChatFormInput {...common} />
		</StateGalleryItem>

		<StateGalleryItem label="Multiline text">
			<ChatFormInput {...common} value={multilineValue} />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ChatFormInput {...common} disabled value="Waiting for the model to load" />
		</StateGalleryItem>

		<StateGalleryItem label="Code span">
			<ChatFormInput {...common} value="Why does `npm run check` fail after `npm install`?" />
		</StateGalleryItem>

		<StateGalleryItem label="File mentions">
			<ChatFormInput {...common} value={mentionValue} />
		</StateGalleryItem>
	</StateGallery>
</Story>
