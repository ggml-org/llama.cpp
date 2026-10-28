<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormActionSubmit from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionSubmit.svelte';

	const { Story } = defineMeta({
		component: ChatFormActionSubmit,
		parameters: {
			layout: 'centered'
		},
		title: 'Chat/ChatScreen/ChatForm/ChatFormActionSubmit'
	});
</script>

<Story asChild name="States">
	<StateGallery columns={5}>
		<StateGalleryItem label="Empty" note="nothing to send">
			<ChatFormActionSubmit canSend={false} />
		</StateGalleryItem>

		<StateGalleryItem label="Enabled">
			<ChatFormActionSubmit canSend />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<ChatFormActionSubmit canSend disabled />
		</StateGalleryItem>

		<StateGalleryItem label="No model" note="tooltip on hover">
			<ChatFormActionSubmit canSend={false} tooltipLabel="Please select a model first" />
		</StateGalleryItem>

		<StateGalleryItem label="Unavailable" note="error, tooltip on hover">
			<ChatFormActionSubmit
				canSend={false}
				showErrorState
				tooltipLabel="Selected model is not available, please select another"
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
