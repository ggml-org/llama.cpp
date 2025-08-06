<script lang="ts">
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { ChatSidebar } from '$lib/components/app';
	import { activeMessages, isLoading, contextError, clearContextError } from '$lib/stores/chat.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import { page } from '$app/state';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { AlertTriangle } from '@lucide/svelte';

	let { children } = $props();

	let isChatRoute = $derived(page.route.id === '/chat/[id]');
	let isHomeRoute = $derived(page.route.id === '/');
	let isNewChatMode = $derived(page.url.searchParams.get('new_chat') === 'true');
	let showSidebarByDefault = $derived(activeMessages().length > 0 || isLoading());
	let sidebarOpen = $state(false);

	$effect(() => {
		if (isHomeRoute && !isNewChatMode) {
			// Auto-collapse sidebar when navigating to home route (but not in new chat mode)
			sidebarOpen = false;
		} else if (isHomeRoute && isNewChatMode) {
			// Keep sidebar open in new chat mode
			sidebarOpen = true;
		} else if (isChatRoute) {
			// On chat routes, show sidebar by default
			sidebarOpen = true;
		} else {
			// Other routes follow default behavior
			sidebarOpen = showSidebarByDefault;
		}
	});

	// Initialize server properties on app load
	$effect(() => {
		serverStore.fetchServerProps();
	});
</script>

<ModeWatcher />

<Toaster richColors />

<!-- Context Length Error Alert Dialog -->
<AlertDialog.Root
	open={contextError() !== null}
	onOpenChange={(open) => !open && clearContextError()}
>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<AlertTriangle class="text-destructive h-5 w-5" />
				Message Too Long
			</AlertDialog.Title>
			<AlertDialog.Description>
				Your message exceeds the model's context window and cannot be processed.
			</AlertDialog.Description>
		</AlertDialog.Header>

		{#if contextError()}
			<div class="space-y-3 text-sm">
				<div class="bg-muted rounded-lg p-3">
					<div class="mb-2 font-medium">Token Usage:</div>
					<div class="text-muted-foreground space-y-1">
						<div>
							Estimated tokens: <span class="font-mono"
								>{contextError()?.estimatedTokens.toLocaleString()}</span
							>
						</div>
						<div>
							Maximum allowed: <span class="font-mono"
								>{contextError()?.maxAllowed.toLocaleString()}</span
							>
						</div>
						<div>
							Context window: <span class="font-mono"
								>{contextError()?.maxContext.toLocaleString()}</span
							>
						</div>
					</div>
				</div>

				<div>
					<div class="mb-2 font-medium">Suggestions:</div>
					<ul class="text-muted-foreground list-inside list-disc space-y-1">
						<li>Shorten your message</li>
						<li>Remove some file attachments</li>
						<li>Start a new conversation</li>
					</ul>
				</div>
			</div>
		{/if}

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => clearContextError()}>Got it</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>

<Sidebar.Provider bind:open={sidebarOpen}>
	<div class="flex h-screen w-full">
		<Sidebar.Root class="h-full">
			<ChatSidebar />
		</Sidebar.Root>

		<Sidebar.Trigger
			class="transition-left absolute h-8 w-8 duration-200 ease-linear {sidebarOpen
				? 'md:left-[var(--sidebar-width)]'
				: 'left-0'}"
			style="translate: 1rem 1rem; z-index: 99999;"
		/>

		<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
			{@render children?.()}
		</Sidebar.Inset>
	</div>
</Sidebar.Provider>
