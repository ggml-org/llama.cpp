<script lang="ts" module>
	import {
		mcpResourceAttachment,
		mcpResourceImageAttachment,
		toMcpResourceAttachment
	} from '../screen/fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatAttachmentsListItemMcpResource from '#lib/components/app/chat/ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemMcpResource.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: ChatAttachmentsListItemMcpResource,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatAttachments/ChatAttachmentsListItemMcpResource'
	});

	const onclick = () => console.log('preview');
</script>

<Story asChild name="States">
	<StateGallery columns={3}>
		<StateGalleryItem label="Text resource">
			<ChatAttachmentsListItemMcpResource
				attachment={toMcpResourceAttachment(mcpResourceAttachment)}
				{onclick}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Image resource">
			<ChatAttachmentsListItemMcpResource
				attachment={toMcpResourceAttachment(mcpResourceImageAttachment)}
				{onclick}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Removable">
			<ChatAttachmentsListItemMcpResource
				attachment={toMcpResourceAttachment(mcpResourceAttachment)}
				onRemove={(id: string) => console.log('remove', id)}
				{onclick}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Loading">
			<ChatAttachmentsListItemMcpResource
				attachment={toMcpResourceAttachment(mcpResourceAttachment, { loading: true })}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Error">
			<ChatAttachmentsListItemMcpResource
				attachment={toMcpResourceAttachment(mcpResourceAttachment, {
					error: 'Resource not found'
				})}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
