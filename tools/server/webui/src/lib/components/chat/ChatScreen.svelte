<script lang="ts">
	import ChatMessages from './ChatMessages.svelte';
	import ChatForm from './ChatForm.svelte';
	import ServerStatus from './ServerStatus.svelte';
	import { chatMessages, isLoading, sendMessage, stopGeneration } from '$lib/stores/chat.svelte';
	import { fly, slide } from 'svelte/transition';

	// Check if we have any messages to determine layout
	const hasMessages = $derived(chatMessages.length > 0 || isLoading);
</script>

{#if hasMessages}
	<!-- Chat with messages - normal layout -->
	<div class="flex h-full flex-col">
		<!-- Messages area -->
		<div class="flex-1 overflow-hidden">
			<ChatMessages
				class="mx-auto w-full max-w-[50rem]"
				messages={chatMessages}
				{isLoading}
			/>
		</div>

		<!-- Form at bottom -->
		<div in:slide={{ duration: 400, axis: 'y' }}>
			<ChatForm
				class="border-t"
				{isLoading}
				showHelperText={false}
				onsend={(message) => sendMessage(message)}
				onstop={() => stopGeneration()}
			/>
		</div>
	</div>
{:else}
	<!-- Empty state - centered form -->
	<div class="flex h-full items-center justify-center">
		<div class="w-full max-w-2xl px-4">
			<!-- Welcome message -->
			<div class="mb-8 text-center" in:fly={{ y: -30, duration: 600 }}>
				<h1 class="mb-2 text-3xl font-semibold tracking-tight">llama.cpp</h1>
				<p class="text-muted-foreground text-lg">How can I help you today?</p>
			</div>

			<!-- Server Status -->
			<div class="mb-6 flex justify-center" in:slide={{ duration: 500, delay: 300, axis: 'y' }}>
				<ServerStatus variant="inline" class="rounded-lg border bg-card/50 px-4 py-2" />
			</div>

			<!-- Centered form -->
			<div in:slide={{ duration: 600, delay: 500, axis: 'y' }}>
				<ChatForm
					{isLoading}
					showHelperText={true}
					onsend={(message) => sendMessage(message)}
					onstop={() => stopGeneration()}
				/>
			</div>
		</div>
	</div>
{/if}
