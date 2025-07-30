<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { ChatForm, ChatMessages, ServerInfo } from '$lib/components';
	import {
		activeMessages,
		activeConversation,
		isLoading,
		sendMessage,
		stopGeneration,
	} from '$lib/stores/chat.svelte';
	import { onMount } from 'svelte';
	import { fly, slide } from 'svelte/transition';

	let { showCenteredEmpty = false } = $props();

	let chatScrollContainer: HTMLDivElement | undefined = $state();
	let scrollInterval: ReturnType<typeof setInterval> | undefined;
	let autoScrollEnabled = $state(true);

	const isEmpty = $derived(
		showCenteredEmpty && !activeConversation() && activeMessages().length === 0 && !isLoading()
	);

	async function handleSendMessage(message: string) {
		await sendMessage(message);
	}


	function scrollChatToBottom() {
		chatScrollContainer?.scrollTo({top: chatScrollContainer?.scrollHeight, behavior: 'instant'})
	}

	afterNavigate(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	})

	onMount(() => {
		setTimeout(scrollChatToBottom, 10); //  This is a dirty workaround, need to find racing conditions
	})


	function handleScroll() {
		if (!chatScrollContainer) return;
		
		const { scrollTop, scrollHeight, clientHeight } = chatScrollContainer;
		const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
		
		if (distanceFromBottom > 50) {
			autoScrollEnabled = false;
		} else if (distanceFromBottom <= 1) {
			autoScrollEnabled = true;
		}
	}

	$effect(() => {
		if (isLoading() && autoScrollEnabled) {
			scrollInterval = setInterval(scrollChatToBottom, 100);
		} else if (scrollInterval) {
			clearInterval(scrollInterval);
			scrollInterval = undefined;
		}
	})
</script>

{#if !isEmpty}
	<div class="flex h-full flex-col overflow-y-auto" bind:this={chatScrollContainer} onscroll={handleScroll}>
			<ChatMessages class="mb-36" messages={activeMessages()} />

			<div
				class="z-999 sticky bottom-0 mx-auto mt-auto max-w-[56rem]"
				in:slide={{ duration: 400, axis: 'y' }}
			>
				<div class="bg-background m-auto rounded-t-3xl border-t pb-4 min-w-[56rem]">
					<ChatForm
						isLoading={isLoading()}
						showHelperText={false}
						onsend={handleSendMessage}
						onstop={() => stopGeneration()}
					/>
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
