<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { expect } from 'storybook/test';
	import { mockServerProps, mockConfigs } from './fixtures/storybook-mocks';
	import jpgAsset from './fixtures/assets/1.jpg?url';
	import svgAsset from './fixtures/assets/hf-logo.svg?url';
	import pdfAsset from './fixtures/assets/example.pdf?raw';

	const { Story } = defineMeta({
		title: 'Components/ChatScreen/ChatForm',
		component: ChatForm,
		parameters: {
			layout: 'centered'
		}
	});

	let fileAttachments = $state([
		{
			id: '1',
			name: '1.jpg',
			type: 'image/jpeg',
			size: 44891,
			preview: jpgAsset,
			file: new File([''], '1.jpg', { type: 'image/jpeg' })
		},
		{
			id: '2',
			name: 'hf-logo.svg',
			type: 'image/svg+xml',
			size: 1234,
			preview: svgAsset,
			file: new File([''], 'hf-logo.svg', { type: 'image/svg+xml' })
		},
		{
			id: '3',
			name: 'example.pdf',
			type: 'application/pdf',
			size: 351048,
			file: new File([pdfAsset], 'example.pdf', { type: 'application/pdf' })
		}
	]);
</script>

<Story
	name="Default"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		mockServerProps(mockConfigs.noModalities);

		const textarea = await canvas.findByRole('textbox');
		const submitButton = await canvas.findByRole('button', { name: 'Send' });

		// Expect the input to be focused after the component is mounted
		await expect(textarea).toHaveFocus();

		// Expect the submit button to be disabled
		await expect(submitButton).toBeDisabled();

		const text = 'What is the meaning of life?';

		await userEvent.clear(textarea);
		await userEvent.type(textarea, text);

		await expect(textarea).toHaveValue(text);

		const fileInput = document.querySelector('input[type="file"]');
		await expect(fileInput).not.toHaveAttribute('accept');

		// Open file attachments dropdown
		const fileUploadButton = canvas.getByText('Attach files');
		await userEvent.click(fileUploadButton);

		// Check dropdown menu items are disabled (no modalities)
		const imagesButton = document.querySelector('.images-button');
		const audioButton = document.querySelector('.audio-button');

		await expect(imagesButton).toHaveAttribute('data-disabled');
		await expect(audioButton).toHaveAttribute('data-disabled');

		// Close dropdown by pressing Escape
		await userEvent.keyboard('{Escape}');
	}}
/>

<Story name="Loading" args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', isLoading: true }} />

<Story
	name="VisionModality"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		mockServerProps(mockConfigs.visionOnly);

		// Open file attachments dropdown and verify it works
		const fileUploadButton = canvas.getByText('Attach files');
		await userEvent.click(fileUploadButton);

		// Verify dropdown menu items exist
		const imagesButton = document.querySelector('.images-button');
		const audioButton = document.querySelector('.audio-button');

		await expect(imagesButton).toBeInTheDocument();
		await expect(audioButton).toBeInTheDocument();

		// Close dropdown by pressing Escape
		await userEvent.keyboard('{Escape}');

		console.log('✅ Vision modality: Dropdown menu verified');
	}}
/>

<Story
	name="AudioModality"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		mockServerProps(mockConfigs.audioOnly);

		// Open file attachments dropdown and verify it works
		const fileUploadButton = canvas.getByText('Attach files');
		await userEvent.click(fileUploadButton);

		// Verify dropdown menu items exist
		const imagesButton = document.querySelector('.images-button');
		const audioButton = document.querySelector('.audio-button');

		await expect(imagesButton).toBeInTheDocument();
		await expect(audioButton).toBeInTheDocument();

		// Close dropdown by pressing Escape
		await userEvent.keyboard('{Escape}');

		console.log('✅ Audio modality: Dropdown menu verified');
	}}
/>

<Story
	name="FileAttachments"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		uploadedFiles: fileAttachments
	}}
	play={async ({ canvas }) => {
		mockServerProps(mockConfigs.bothModalities);

		const jpgAttachment = canvas.getByAltText('1.jpg');
		const svgAttachment = canvas.getByAltText('hf-logo.svg');
		const pdfFileExtension = canvas.getByText('PDF');
		const pdfAttachment = canvas.getByText('example.pdf');
		const pdfSize = canvas.getByText('342.82 KB');

		await expect(jpgAttachment).toBeInTheDocument();
		await expect(jpgAttachment).toHaveAttribute('src', jpgAsset);

		await expect(svgAttachment).toBeInTheDocument();
		await expect(svgAttachment).toHaveAttribute('src', svgAsset);

		await expect(pdfFileExtension).toBeInTheDocument();
		await expect(pdfAttachment).toBeInTheDocument();
		await expect(pdfSize).toBeInTheDocument();
	}}
/>
