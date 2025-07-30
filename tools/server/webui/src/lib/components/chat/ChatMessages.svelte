<script lang="ts">
	import type { Message } from '$lib/types/database';
	import { updateMessage, regenerateMessage } from '$lib/stores/chat.svelte';
	import { ChatMessage } from '$lib/components';
	interface Props {
		class?: string;
		messages?: Message[];
		isLoading?: boolean;
	}

	let { class: className, messages = [], isLoading = false }: Props = $props();
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each messages as message}
		<ChatMessage
			class="mx-auto w-full max-w-[56rem]"
			{message}
			onUpdateMessage={async (msg, newContent) => {
				await updateMessage(msg.id, newContent);
			}}
			onRegenerate={async (msg) => {
				await regenerateMessage(msg.id);
			}}
		/>
	{/each}
</div>
