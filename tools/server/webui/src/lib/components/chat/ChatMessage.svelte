<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { User, Bot } from '@lucide/svelte';

	let { message }: { message: ChatMessageData } = $props();
</script>

<div class="flex gap-3 {message.role === 'user' ? 'justify-end' : 'justify-start'}">
	{#if message.role === 'assistant'}
		<div
			class="bg-background flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow"
		>
			<Bot class="h-4 w-4" />
		</div>
	{/if}

	<Card
		class="max-w-[80%] gap-2 px-4 py-3 {message.role === 'user'
			? 'bg-primary text-primary-foreground'
			: 'bg-muted'}"
	>
		<div class="whitespace-pre-wrap text-sm">
			{message.content}
		</div>
		{#if message.timestamp}
			<div class="text-xs opacity-70">
				{new Date(message.timestamp).toLocaleTimeString()}
			</div>
		{/if}
	</Card>

	{#if message.role === 'user'}
		<div
			class="bg-background flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow"
		>
			<User class="h-4 w-4" />
		</div>
	{/if}
</div>
