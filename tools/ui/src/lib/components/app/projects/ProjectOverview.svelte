<script lang="ts">
	import ProjectConversations from './ProjectConversations.svelte';
	import { MessageSquare } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { RouterService } from '$lib/services/router.service';
	import { conversationsStore } from '$lib/stores';
	import type { Project } from '$lib/types';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	let conversations = $derived(
		conversationsStore.conversations
			.filter((conversation) => conversation.projectId === project.id)
			.sort((a, b) => b.lastModified - a.lastModified)
	);
</script>

<div class="w-full">
	{#if conversations.length > 0}
		<ProjectConversations
			{conversations}
			onOpen={(id) => void goto(RouterService.projectChat(project.id, id))}
		/>
	{:else}
		<div class="flex flex-col items-center gap-2 px-6 text-center">
			<MessageSquare class="h-5 w-5 text-muted-foreground" />

			<p class="max-w-sm text-sm text-balance text-muted-foreground">
				Chats in this project share its instructions and context every time you talk to them.
			</p>
		</div>
	{/if}
</div>
