<script lang="ts">
	import type { DatabaseMessage } from '$lib/types/database';
	import { updateMessage, regenerateMessage } from '$lib/stores/chat.svelte';
	import { ChatMessage } from '$lib/components/app';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
	}

	let { class: className, messages = [] }: Props = $props();
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each messages as message}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
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
