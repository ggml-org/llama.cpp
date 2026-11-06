<!--
	The crystal goblet in Llama Chat: a real exchange (the user bubble, the model's answer) over the real chat form, on
	the app's own page colour. The story seeds the chat stores (seedChatStory) so the form can render.
-->
<script lang="ts">
	import { assistantMessage, noopChatActions, userMessage } from '../../../chat/messages/fixtures';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { onMount } from 'svelte';

	let chat: HTMLElement | undefined = $state();

	// the chat form focuses itself when it mounts; on a page that is read, hand focus back and stay at the top
	onMount(() => {
		const id = setTimeout(() => {
			if (chat?.contains(document.activeElement)) {
				(document.activeElement as HTMLElement).blur();
				window.scrollTo(0, 0);
			}
		}, 200);

		return () => clearTimeout(id);
	});
</script>

<div bind:this={chat} class="goblet">
	<div class="messages">
		<ChatMessage chatActions={noopChatActions} message={userMessage} />

		<ChatMessage chatActions={noopChatActions} message={assistantMessage} />
	</div>

	<ChatScreenForm />
</div>

<style>
	.goblet {
		display: grid;
		gap: 24px;
		width: min(44rem, 100%);
		margin-inline: auto;
		padding: clamp(16px, 3vw, 32px);
		border-radius: 10px;
		background: var(--surface-page);
	}

	.messages {
		display: grid;
		gap: 24px;
	}
</style>
