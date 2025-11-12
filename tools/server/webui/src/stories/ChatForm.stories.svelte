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
		const acceptAttr = fileInput?.getAttribute('accept');
		await expect(fileInput).toHaveAttribute('accept');
		await expect(acceptAttr).not.toContain('image/');
		await expect(acceptAttr).not.toContain('audio/');

		const fileUploadButton = canvas.getByText('Attach files');

		await userEvent.click(fileUploadButton);

		const recordButton = canvas.getAllByRole('button', { name: 'Start recording' })[1];
		const imagesButton = document.querySelector('.images-button');
		const audioButton = document.querySelector('.audio-button');

		await expect(recordButton).toBeDisabled();
		await expect(imagesButton).toHaveAttribute('data-disabled');
		await expect(audioButton).toHaveAttribute('data-disabled');
	}}
/>

<Story name="Loading" args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', isLoading: true }} />

<Story
	name="VisionModality"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		mockServerProps(mockConfigs.visionOnly);

		// Test initial file input state (should accept images but not audio)
		const fileInput = document.querySelector('input[type="file"]');
		const acceptAttr = fileInput?.getAttribute('accept');
		console.log('Vision modality accept attr:', acceptAttr);

		const fileUploadButton = canvas.getByText('Attach files');
		await userEvent.click(fileUploadButton);

		// Test that record button is disabled (no audio support)
		const recordButton = canvas.getAllByRole('button', { name: 'Start recording' })[1];
		await expect(recordButton).toBeDisabled();

		// Test that Images button is enabled (vision support)
		const imagesButton = document.querySelector('.images-button');
		await expect(imagesButton).not.toHaveAttribute('data-disabled');

		// Test that Audio button is disabled (no audio support)
		const audioButton = document.querySelector('.audio-button');
		await expect(audioButton).toHaveAttribute('data-disabled');

		// Fix for dropdown menu side effect
		const body = document.querySelector('body');
		if (body) body.style.pointerEvents = 'all';

		console.log('✅ Vision modality: Images enabled, Audio/Recording disabled');
	}}
/>

<Story
	name="AudioModality"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		mockServerProps(mockConfigs.audioOnly);

		// Test initial file input state (should accept audio but not images)
		const fileInput = document.querySelector('input[type="file"]');
		const acceptAttr = fileInput?.getAttribute('accept');
		console.log('Audio modality accept attr:', acceptAttr);

		const fileUploadButton = canvas.getByText('Attach files');
		await userEvent.click(fileUploadButton);

		// Test that record button is enabled (audio support)
		const recordButton = canvas.getAllByRole('button', { name: 'Start recording' })[1];
		await expect(recordButton).not.toBeDisabled();

		// Test that Images button is disabled (no vision support)
		const imagesButton = document.querySelector('.images-button');
		await expect(imagesButton).toHaveAttribute('data-disabled');

		// Test that Audio button is enabled (audio support)
		const audioButton = document.querySelector('.audio-button');
		await expect(audioButton).not.toHaveAttribute('data-disabled');

		// Fix for dropdown menu side effect
		const body = document.querySelector('body');
		if (body) body.style.pointerEvents = 'all';

		console.log('✅ Audio modality: Audio/Recording enabled, Images disabled');
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
