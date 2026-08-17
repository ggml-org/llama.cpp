<script lang="ts">
	import { MessageSquarePlus } from '@lucide/svelte';
	import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { UI_DATA_ATTRS } from '$lib/constants';

	interface Props {
		conversations: DatabaseConversation[];
		messageCountMap?: Map<string, number>;
		onOpen: (conversation: DatabaseConversation) => void;
		onClose: () => void;
		open?: boolean;
	}

	let {
		conversations,
		messageCountMap = new Map(),
		onClose,
		onOpen,
		open = $bindable()
	}: Props = $props();

	let searchQuery = $state('');

	function handleOpenChange(newOpen: boolean) {
		if (!newOpen) {
			onClose();
		}
	}

	let filteredConversations = $derived(
		conversations.filter((conv) => {
			const name = conv.name || 'Untitled conversation';

			return name.toLowerCase().includes(searchQuery.toLowerCase());
		})
	);
</script>

<Dialog.Root bind:open onOpenChange={handleOpenChange}>
	<Dialog.Portal>
		<Dialog.Overlay class="z-1000000" />

		<Dialog.Content class="z-1000001 max-w-2xl">
			<Dialog.Header>
				<Dialog.Title class="flex items-center gap-2">
					<MessageSquarePlus class="h-5 w-5" />

					Imported Conversations
				</Dialog.Title>

				<Dialog.Description>
					{conversations.length} conversation{conversations.length === 1 ? '' : 's'} imported. Select
					one to open it.
				</Dialog.Description>
			</Dialog.Header>

			<div class="space-y-4">
				<SearchInput bind:value={searchQuery} placeholder="Search conversations..." />

				<div class="overflow-hidden rounded-md border">
					<ScrollArea class="h-100">
						<table class="w-full">
							<thead class="sticky top-0 z-10 bg-muted">
								<tr class="border-b">
									<th class="p-3 text-left text-sm font-medium">Conversation Name</th>

									<th class="w-32 p-3 text-left text-sm font-medium">Messages</th>
								</tr>
							</thead>

							<tbody>
								{#if filteredConversations.length === 0}
									<tr>
										<td colspan="2" class="p-8 text-center text-sm text-muted-foreground">
											No conversations found matching "{searchQuery}"
										</td>
									</tr>
								{:else}
									{#each filteredConversations as conv (conv.id)}
										<tr
											class="cursor-pointer border-b transition-colors hover:bg-muted/50"
											{...{ [UI_DATA_ATTRS.CONVERSATION_ROW]: conv.id }}
											onclick={() => onOpen(conv)}
										>
											<td class="p-3 text-sm">
												<div class="max-w-68 truncate" title={conv.name || 'Untitled conversation'}>
													{conv.name || 'Untitled conversation'}
												</div>
											</td>

											<td class="p-3 text-sm text-muted-foreground">
												{messageCountMap.get(conv.id) ?? 0}
											</td>
										</tr>
									{/each}
								{/if}
							</tbody>
						</table>
					</ScrollArea>
				</div>
			</div>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
