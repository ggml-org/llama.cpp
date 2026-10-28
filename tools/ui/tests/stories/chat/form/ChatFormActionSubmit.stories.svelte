<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormActionSubmit from '$lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActionSubmit.svelte';

	const { Story } = defineMeta({
		component: ChatFormActionSubmit,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormActionSubmit'
	});
</script>

<Story args={{ canSend: false }} name="Default" />

<Story args={{ canSend: true }} name="Enabled" />

<Story args={{ canSend: true, disabled: true }} name="Disabled" />

<Story
	args={{ canSend: false, tooltipLabel: 'Please select a model first' }}
	name="NoModelSelected"
/>

<Story
	args={{
		canSend: false,
		showErrorState: true,
		tooltipLabel: 'Selected model is not available, please select another'
	}}
	name="ModelNotAvailable"
/>
