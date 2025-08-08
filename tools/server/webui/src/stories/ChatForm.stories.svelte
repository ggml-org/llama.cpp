<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';
	import { expect } from 'storybook/internal/test';

	const { Story } = defineMeta({
		title: 'Components/ChatForm',
		component: ChatForm,
		parameters: {
			layout: 'centered'
		}
	});
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
  }}
/>

<Story
  name="Loading"
  args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]', isLoading: true }}
/>
