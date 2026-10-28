<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { SERVER_HOME } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormInput from '$lib/components/app/chat/ChatForm/ChatFormInput/ChatFormInput.svelte';

	const { Story } = defineMeta({
		component: ChatFormInput,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormInput'
	});

	const common = {
		class: 'w-full px-5 py-1.5',
		placeholder: 'Type a message...'
	};

	const multilineValue =
		'Explain how the KV cache works in llama.cpp.\nKeep it short, and list the main flags that change its size.';

	const mentionValue = `Summarize [README.md](file://${SERVER_HOME}/llama.cpp/README.md) and [ui](file://${SERVER_HOME}/llama.cpp/tools/ui/) please`;
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

		<StateGalleryItem label="Rich, code span">
			<ChatFormInput
				{...common}
				useRichInput
				value="Why does `npm run check` fail after `npm install`?"
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Rich, file mentions">
			<ChatFormInput {...common} useRichInput value={mentionValue} />
		</StateGalleryItem>
	</StateGallery>
</Story>
