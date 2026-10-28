<script lang="ts" module>
	import {
		mcpResourceAttachment,
		mcpResourceImageAttachment,
		toMcpResourceAttachment
	} from '../fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatAttachmentsListItemMcpResource from '$lib/components/app/chat/ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemMcpResource.svelte';

	const { Story } = defineMeta({
		component: ChatAttachmentsListItemMcpResource,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatAttachments/ChatAttachmentsListItemMcpResource'
	});

	const onclick = () => console.log('preview');
</script>

<Story
	args={{ attachment: toMcpResourceAttachment(mcpResourceAttachment), onclick }}
	name="Default"
/>

<Story
	args={{ attachment: toMcpResourceAttachment(mcpResourceImageAttachment), onclick }}
	name="ImageResource"
/>

<Story
	args={{
		attachment: toMcpResourceAttachment(mcpResourceAttachment),
		onclick,
		onRemove: (id: string) => console.log('remove', id)
	}}
	name="Removable"
/>

<Story
	args={{ attachment: toMcpResourceAttachment(mcpResourceAttachment, { loading: true }) }}
	name="Loading"
/>

<Story
	args={{
		attachment: toMcpResourceAttachment(mcpResourceAttachment, {
			error: 'Resource not found'
		})
	}}
	name="Error"
/>
