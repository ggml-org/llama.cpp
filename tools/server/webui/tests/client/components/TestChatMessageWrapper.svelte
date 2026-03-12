<script lang="ts">
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Sidebar from '$lib/components/ui/sidebar';
	import { setChatActionsContext } from '$lib/contexts';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage.svelte';

	const props = $props<{
		message: DatabaseMessage;
		onDelete: (message: DatabaseMessage) => void;
	}>();

	setChatActionsContext({
		copy: () => {},
		delete: props.onDelete,
		navigateToSibling: () => {},
		editWithBranching: () => {},
		editWithReplacement: () => {},
		editUserMessagePreserveResponses: () => {},
		regenerateWithBranching: () => {},
		continueAssistantMessage: () => {}
	});
</script>

<Tooltip.Provider>
	<Sidebar.Provider open={false}>
		<ChatMessage message={props.message} />
	</Sidebar.Provider>
</Tooltip.Provider>
