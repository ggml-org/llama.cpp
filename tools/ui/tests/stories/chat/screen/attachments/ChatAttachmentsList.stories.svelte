<script lang="ts" module>
	import StateGallery from '../../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../../helpers/StateGalleryItem.svelte';
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

<Story asChild name="States">
	<StateGallery columns={1}>
		<StateGalleryItem label="Pending uploads" note="chat form, removable">
			<ChatAttachmentsList
				class="w-[40rem]"
				{onFileRemove}
				uploadedFiles={[uploadedImage, uploadedText, uploadedPdf]}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Images only" note="readonly">
			<ChatAttachmentsList
				attachments={[imageAttachment, logoAttachment, gradientAttachment]}
				class="w-[40rem]"
				imageHeight="h-48"
				imageWidth="max-w-72"
				readonly
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Files only">
			<ChatAttachmentsList
				class="w-[40rem]"
				{onFileRemove}
				uploadedFiles={[
					uploadedText,
					uploadedNotes,
					uploadedPdf,
					uploadedAudio,
					uploadedVideo,
					uploadedArchive
				]}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Mixed, removable">
			<ChatAttachmentsList
				class="w-[40rem]"
				{onFileRemove}
				uploadedFiles={[uploadedImage, uploadedLogo, uploadedText, uploadedPdf, uploadedAudio]}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Mixed, readonly" note="as shown in a sent message">
			<ChatAttachmentsList
				attachments={[
					imageAttachment,
					textAttachment,
					pdfAttachment,
					pdfAsImagesAttachment,
					audioAttachment,
					videoAttachment
				]}
				class="w-[40rem]"
				imageHeight="h-48"
				imageWidth="max-w-72"
				readonly
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Readonly text file">
			<ChatAttachmentsList attachments={[notesAttachment]} class="w-[40rem]" readonly />
		</StateGalleryItem>

		<StateGalleryItem label="MCP prompts" note="ready, loading, error">
			<ChatAttachmentsList
				class="w-[40rem]"
				{onFileRemove}
				uploadedFiles={[uploadedMcpPrompt, uploadedMcpPromptLoading, uploadedMcpPromptError]}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="MCP, readonly">
			<ChatAttachmentsList
				attachments={[mcpPromptAttachment, mcpResourceAttachment]}
				class="w-[40rem]"
				readonly
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Single row overflow">
			<ChatAttachmentsList
				class="w-[28rem]"
				limitToSingleRow
				{onFileRemove}
				uploadedFiles={[
					uploadedImage,
					uploadedLogo,
					uploadedGradient,
					uploadedText,
					uploadedNotes,
					uploadedPdf,
					uploadedAudio,
					uploadedVideo
				]}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
