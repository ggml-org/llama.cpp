<script lang="ts">
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import ChatMessageUserPending from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageUser/ChatMessageUserPending.svelte';
	import ChatScreen from '$lib/components/app/chat/ChatScreen/ChatScreen.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { ChatMessageActions, DatabaseMessage, DatabaseMessageExtra } from '$lib/types';
	import { Toaster } from 'svelte-sonner';

	type Props =
		| { surface: 'screen' }
		| { surface: 'saved'; message: DatabaseMessage; chatActions: ChatMessageActions }
		| {
				surface: 'pending';
				content: string;
				extras: DatabaseMessageExtra[];
				onEdit: (content: string, extras?: DatabaseMessageExtra[]) => void;
		  };
	let props: Props = $props();
</script>

<Tooltip.Provider>
	{#if props.surface === 'screen'}
		<ChatScreen showCenteredEmpty={true} />
	{:else if props.surface === 'saved'}
		<ChatMessage chatActions={props.chatActions} message={props.message} />
	{:else}
		<ChatMessageUserPending
			content={props.content}
			extras={props.extras}
			onDelete={() => {}}
			onEdit={props.onEdit}
			onSendImmediately={() => {}}
		/>
	{/if}

	<Toaster closeButton richColors />
</Tooltip.Provider>
