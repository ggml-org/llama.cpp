<script lang="ts" module>
	import {
		audioAttachment,
		gradientAttachment,
		imageAttachment,
		logoAttachment,
		mcpPromptAttachment,
		mcpResourceAttachment,
		notesAttachment,
		pdfAsImagesAttachment,
		pdfAttachment,
		textAttachment,
		uploadedArchive,
		uploadedAudio,
		uploadedGradient,
		uploadedImage,
		uploadedLogo,
		uploadedMcpPrompt,
		uploadedMcpPromptError,
		uploadedMcpPromptLoading,
		uploadedNotes,
		uploadedPdf,
		uploadedText,
		uploadedVideo,
		videoAttachment
	} from '../fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatAttachmentsList from '$lib/components/app/chat/ChatAttachments/ChatAttachmentsList/ChatAttachmentsList.svelte';

	const { Story } = defineMeta({
		component: ChatAttachmentsList,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatAttachments/ChatAttachmentsList'
	});

	const onFileRemove = (id: string) => console.log('remove', id);
</script>

<!-- pending uploads in the chat form, removable -->
<Story
	args={{
		class: 'w-[40rem]',
		onFileRemove,
		uploadedFiles: [uploadedImage, uploadedText, uploadedPdf]
	}}
	name="Default"
/>

<Story
	args={{
		attachments: [imageAttachment, logoAttachment, gradientAttachment],
		class: 'w-[40rem]',
		imageHeight: 'h-48',
		imageWidth: 'max-w-72',
		readonly: true
	}}
	name="ImagesOnly"
/>

<Story
	args={{
		class: 'w-[40rem]',
		onFileRemove,
		uploadedFiles: [
			uploadedText,
			uploadedNotes,
			uploadedPdf,
			uploadedAudio,
			uploadedVideo,
			uploadedArchive
		]
	}}
	name="FilesOnly"
/>

<Story
	args={{
		class: 'w-[40rem]',
		onFileRemove,
		uploadedFiles: [uploadedImage, uploadedLogo, uploadedText, uploadedPdf, uploadedAudio]
	}}
	name="MixedRemovable"
/>

<!-- stored attachments as shown in a sent message -->
<Story
	args={{
		attachments: [
			imageAttachment,
			textAttachment,
			pdfAttachment,
			pdfAsImagesAttachment,
			audioAttachment,
			videoAttachment
		],
		class: 'w-[40rem]',
		imageHeight: 'h-48',
		imageWidth: 'max-w-72',
		readonly: true
	}}
	name="MixedReadonly"
/>

<Story
	args={{
		attachments: [notesAttachment],
		class: 'w-[40rem]',
		readonly: true
	}}
	name="ReadonlyTextFile"
/>

<Story
	args={{
		class: 'w-[40rem]',
		onFileRemove,
		uploadedFiles: [uploadedMcpPrompt, uploadedMcpPromptLoading, uploadedMcpPromptError]
	}}
	name="McpPrompts"
/>

<Story
	args={{
		attachments: [mcpPromptAttachment, mcpResourceAttachment],
		class: 'w-[40rem]',
		readonly: true
	}}
	name="McpReadonly"
/>

<Story
	args={{
		class: 'w-[28rem]',
		limitToSingleRow: true,
		onFileRemove,
		uploadedFiles: [
			uploadedImage,
			uploadedLogo,
			uploadedGradient,
			uploadedText,
			uploadedNotes,
			uploadedPdf,
			uploadedAudio,
			uploadedVideo
		]
	}}
	name="SingleRowOverflow"
/>
