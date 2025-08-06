<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatForm from '$lib/components/app/chat/ChatForm/ChatForm.svelte';

	const { Story } = defineMeta({
		title: 'Components/ChatForm',
		component: ChatForm,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story name="Default" args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }} />
