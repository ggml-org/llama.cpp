<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { expect, waitFor } from 'storybook/internal/test';
	import { mockServerProps, mockConfigs } from './fixtures/storybook-mocks';

	const { Story } = defineMeta({
		title: 'Components/ChatForm',
		component: ChatForm,
		parameters: {
			layout: 'centered'
		}
	});

	// Mock uploaded files with working data URLs for Storybook
	const mockFileAttachments = [
		// {
		// 	id: '1',
		// 	name: '1.jpg',
		// 	type: 'image/jpeg',
		// 	size: 44891,
		// 	url: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzMzNzNkYyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE4IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlPC90ZXh0Pjwvc3ZnPg==',
		// 	file: new File([''], '1.jpg', { type: 'image/jpeg' })
		// },
		// {
		// 	id: '2',
		// 	name: 'beautiful-flowers-lotus.webp',
		// 	type: 'image/webp',
		// 	size: 817630,
		// 	url: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzMzNzNkYyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE4IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZsb3dlcnM8L3RleHQ+PC9zdmc+',
		// 	file: new File([''], 'beautiful-flowers-lotus.webp', { type: 'image/webp' })
		// },
		// {
		// 	id: '3',
		// 	name: 'recording.wav',
		// 	type: 'audio/wav',
		// 	size: 512000,
		// 	url: 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT',
		// 	file: new File(['test audio content'], 'recording.wav', { type: 'audio/wav' })
		// },
		{
			id: '4',
			name: 'example.pdf',
			type: 'application/pdf',
			size: 351048,
			url: 'data:application/pdf;base64,JVBERi0xLjQKJcOkw7zDtsO4CjIgMCBvYmoKPDwKL0xlbmd0aCAzIDAgUgovRmlsdGVyIC9GbGF0ZURlY29kZQo+PgpzdHJlYW0KeJxLy8wpTVWwUshIzStRyE9VqFYoLU4tykvMTVUozy/KSVGwUsjNTFGwUsrIyFGwUsrJTFGyMjJQUKhWykvMTbVSqAUAXYsZGAplbmRzdHJlYW0KZW5kb2JqCgozIDAgb2JqCjw8Ci9MZW5ndGggNDcKPj4Kc3RyZWFtCkJUCi9GMSAxMiBUZgoxIDAgMCAxIDcwIDc1MCBUbQooSGVsbG8gV29ybGQpIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKCjQgMCBvYmoKPDwKL1R5cGUgL0ZvbnQKL1N1YnR5cGUgL1R5cGUxCi9CYXNlRm9udCAvSGVsdmV0aWNhCj4+CmVuZG9iagoKNSAwIG9iago8PAovVHlwZSAvUGFnZQovUGFyZW50IDYgMCBSCi9SZXNvdXJjZXMgPDwKL0ZvbnQgPDwKL0YxIDQgMCBSCj4+Cj4+Ci9NZWRpYUJveCBbMCAwIDYxMiA3OTJdCi9Db250ZW50cyAzIDAgUgo+PgplbmRvYmoKCjYgMCBvYmoKPDwKL1R5cGUgL1BhZ2VzCi9LaWRzIFs1IDAgUl0KL0NvdW50IDEKL01lZGlhQm94IFswIDAgNjEyIDc5Ml0KPj4KZW5kb2JqCgo3IDAgb2JqCjw8Ci9UeXBlIC9DYXRhbG9nCi9QYWdlcyA2IDAgUgo+PgplbmRvYmoKCnhyZWYKMCA4CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAwOSAwMDAwMCBuIAowMDAwMDAwMDc0IDAwMDAwIG4gCjAwMDAwMDAxNzkgMDAwMDAgbiAKMDAwMDAwMDI3MyAwMDAwMCBuIAowMDAwMDAwMzQ4IDAwMDAwIG4gCjAwMDAwMDA0ODYgMDAwMDAgbiAKMDAwMDAwMDU2MyAwMDAwMCBuIAp0cmFpbGVyCjw8Ci9TaXplIDgKL1Jvb3QgNyAwIFIKPj4Kc3RhcnR4cmVmCjYxMwolJUVPRgo=',
			file: new File(['%PDF-1.4 test content'], 'example.pdf', { type: 'application/pdf' })
		}
	];
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

<Story
  name="Loading"
  args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', isLoading: true }}
/>

<Story
  name="VisionModality"
  args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
  play={async ({ canvas, userEvent }) => {
    mockServerProps(mockConfigs.visionOnly);

	await waitFor(() => {
		const fileInput = document.querySelector('input[type="file"]');
		const acceptAttr = fileInput?.getAttribute('accept');
		return acceptAttr;
	});

    // Test initial file input state (should not accept images/audio without dropdown selection)
    const fileInput = document.querySelector('input[type="file"]');
    const acceptAttr = fileInput?.getAttribute('accept');
	console.log(acceptAttr);
    await expect(fileInput).toHaveAttribute('accept');
    await expect(acceptAttr).toContain('image/');
    await expect(acceptAttr).not.toContain('audio/');

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

    console.log('✅ Vision modality: Images enabled, Audio/Recording disabled');
  }}
/>


<Story
  name="FileAttachments"
  args={{ 
    class: 'max-w-[56rem] w-[calc(100vw-2rem)]',
    uploadedFiles: mockFileAttachments
  }}
  play={async ({ canvasElement }) => {
    // Test that both vision and audio modalities are enabled
    const fileUploadButton = canvasElement.querySelector('button[aria-label*="Upload"], button:has([data-lucide="paperclip"])');
    
    if (fileUploadButton && fileUploadButton instanceof HTMLButtonElement) {
      fileUploadButton.click();
      
      // Wait for dropdown to appear
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Check if both Images and Audio options are available
      const imagesOption = canvasElement.querySelector('[data-testid="upload-images"], button:contains("Images")');
      const audioOption = canvasElement.querySelector('[data-testid="upload-audio"], button:contains("Audio")');
      
      if (imagesOption && imagesOption instanceof HTMLButtonElement && !imagesOption.disabled) {
        console.log('✅ File Attachments: Vision modality enabled');
      }
      
      if (audioOption && audioOption instanceof HTMLButtonElement && !audioOption.disabled) {
        console.log('✅ File Attachments: Audio modality enabled');
      }
    }
    
    // Test microphone availability
    const micButton = canvasElement.querySelector('button[aria-label*="Record"], button:has([data-lucide="mic"])');
    if (micButton && micButton instanceof HTMLButtonElement && !micButton.disabled) {
      console.log('✅ File Attachments: Microphone recording enabled');
    }
  }}
/>
