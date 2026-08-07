<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { expect } from 'storybook/test';
	import jpgAsset from './fixtures/assets/1.jpg?url';
	import svgAsset from './fixtures/assets/hf-logo.svg?url';
	import pdfAsset from './fixtures/assets/example.pdf?raw';

	const { Story } = defineMeta({
		title: 'Components/ChatScreen/ChatScreenForm',
		component: ChatScreenForm,
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
	}}
/>

<Story name="Loading" args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', isLoading: true }} />

<Story
	name="FileAttachments"
	args={{
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
		uploadedFiles: fileAttachments
	}}
	play={async ({ canvas }) => {
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
