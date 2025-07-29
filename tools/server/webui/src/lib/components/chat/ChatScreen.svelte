<script lang="ts">
	import ChatMessages from './ChatMessages.svelte';
	import ChatForm from './ChatForm.svelte';
	import ServerInfo from './ServerInfo.svelte';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration
	} from '$lib/stores/chat.svelte';
	import { fly, slide } from 'svelte/transition';

	let { showCenteredEmpty = false } = $props();

	const isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

	async function handleSendMessage(message: string) {
		await sendMessage(message);
	}
</script>

{#if !isEmpty}
	<div class="flex h-full flex-col">
		<div class="flex-1 overflow-hidden">
			<ChatMessages class="mb-36" messages={activeMessages()} isLoading={isLoading()} />

			<div
				class="z-999 sticky bottom-0 m-auto max-w-[56rem]"
				in:slide={{ duration: 400, axis: 'y' }}
			>
				<div class="bg-background m-auto rounded-t-3xl border-t pb-4">
					<ChatForm
						isLoading={isLoading()}
						showHelperText={false}
						onsend={handleSendMessage}
						onstop={() => stopGeneration()}
					/>
				</div>
			</div>
		</div>
	</div>
{:else}
	<div class="flex h-full items-center justify-center">
		<div class="w-full max-w-2xl px-4">
			<div class="mb-8 text-center" in:fly={{ y: -30, duration: 600 }}>
				<h1 class="mb-2 text-3xl font-semibold tracking-tight">llama.cpp</h1>
				<p class="text-muted-foreground text-lg">How can I help you today?</p>
			</div>

			<div
				class="mb-6 flex justify-center"
				in:slide={{ duration: 500, delay: 300, axis: 'y' }}
			>
				<ServerInfo />
			</div>

			<div in:slide={{ duration: 600, delay: 500, axis: 'y' }}>
				<ChatForm
					isLoading={isLoading()}
					showHelperText={true}
					onsend={handleSendMessage}
					onstop={() => stopGeneration()}
				/>
			</div>
		</div>
	</div>
{/if}
