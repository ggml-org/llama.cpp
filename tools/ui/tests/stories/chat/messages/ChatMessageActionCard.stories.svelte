<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatMessageActionCardContinueRequest from '$lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardContinueRequest.svelte';
	import ChatMessageActionCardPermissionRequest from '$lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardPermissionRequest.svelte';
	import { BuiltInTool } from '$lib/enums';

	const { Story } = defineMeta({
		args: {
			onDecision: () => {},
			serverLabel: 'GitHub',
			toolName: 'create_issue'
		},
		component: ChatMessageActionCardPermissionRequest,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatMessages/ChatMessageActionCard'
	});
</script>

<Story name="Default" />

<Story
	args={{ serverLabel: '', toolName: BuiltInTool.SERVER_EXEC_SHELL_COMMAND }}
	name="PermissionBuiltInTool"
/>

<Story asChild name="ContinueRequest">
	<ChatMessageActionCardContinueRequest onDecision={() => {}} />
</Story>
