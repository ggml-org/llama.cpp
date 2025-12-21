<script lang="ts">
	import { Search, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		conversations: DatabaseConversation[];
		messageCountMap?: Map<string, number>;
		mode: 'export' | 'import';
		onCancel: () => void;
		onConfirm: (selectedConversations: DatabaseConversation[]) => void;
	}

	let { conversations, messageCountMap = new Map(), mode, onCancel, onConfirm }: Props = $props();

	let searchQuery = $state('');
	let selectedIds = $state.raw<SvelteSet<string>>(new SvelteSet(conversations.map((c) => c.id)));
	let lastClickedId = $state<string | null>(null);

	let filteredConversations = $derived(
		conversations.filter((conv) => {
			const name = conv.name || 'Untitled conversation';
			return name.toLowerCase().includes(searchQuery.toLowerCase());
		})
	);

	let allSelected = $derived(
		filteredConversations.length > 0 &&
			filteredConversations.every((conv) => selectedIds.has(conv.id))
	);

	let someSelected = $derived(
		filteredConversations.some((conv) => selectedIds.has(conv.id)) && !allSelected
	);

	function toggleConversation(id: string, shiftKey: boolean = false) {
		const newSet = new SvelteSet(selectedIds);

		if (shiftKey && lastClickedId !== null) {
			const lastIndex = filteredConversations.findIndex((c) => c.id === lastClickedId);
			const currentIndex = filteredConversations.findIndex((c) => c.id === id);

			if (lastIndex !== -1 && currentIndex !== -1) {
				const start = Math.min(lastIndex, currentIndex);
				const end = Math.max(lastIndex, currentIndex);

				const shouldSelect = !newSet.has(id);

				for (let i = start; i <= end; i++) {
					if (shouldSelect) {
						newSet.add(filteredConversations[i].id);
					} else {
						newSet.delete(filteredConversations[i].id);
					}
				}

				selectedIds = newSet;
				return;
			}
		}

		if (newSet.has(id)) {
			newSet.delete(id);
		} else {
			newSet.add(id);
		}

		selectedIds = newSet;
		lastClickedId = id;
	}

	function toggleAll() {
		if (allSelected) {
			const newSet = new SvelteSet(selectedIds);

			filteredConversations.forEach((conv) => newSet.delete(conv.id));
			selectedIds = newSet;
		} else {
			const newSet = new SvelteSet(selectedIds);

			filteredConversations.forEach((conv) => newSet.add(conv.id));
			selectedIds = newSet;
		}
	}

	function handleConfirm() {
		const selected = conversations.filter((conv) => selectedIds.has(conv.id));
		onConfirm(selected);
	}

	function handleCancel() {
		selectedIds = new SvelteSet(conversations.map((c) => c.id));
		searchQuery = '';
		lastClickedId = null;

		onCancel();
	}

	export function reset() {
		selectedIds = new SvelteSet(conversations.map((c) => c.id));
		searchQuery = '';
		lastClickedId = null;
	}
</script>

<div class="space-y-4">
	<div class="relative">
		<Search class="absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-muted-foreground" />

		<Input bind:value={searchQuery} placeholder="Search conversations..." class="pr-9 pl-9" />

		{#if searchQuery}
			<button
				class="absolute top-1/2 right-3 -translate-y-1/2 text-muted-foreground hover:text-foreground"
				onclick={() => (searchQuery = '')}
				type="button"
			>
				<X class="h-4 w-4" />
			</button>
		{/if}
	</div>

	<div class="flex items-center justify-between text-sm text-muted-foreground">
		<span>
			{selectedIds.size} of {conversations.length} selected
			{#if searchQuery}
				({filteredConversations.length} shown)
			{/if}
		</span>
	</div>

	<div class="overflow-hidden rounded-md border">
		<ScrollArea class="h-[400px]">
			<table class="w-full">
				<thead class="sticky top-0 z-10 bg-muted">
					<tr class="border-b">
						<th class="w-12 p-3 text-left">
							<Checkbox
								checked={allSelected}
								indeterminate={someSelected}
								onCheckedChange={toggleAll}
							/>
						</th>

						<th class="p-3 text-left text-sm font-medium">Conversation Name</th>

						<th class="w-32 p-3 text-left text-sm font-medium">Messages</th>
					</tr>
				</thead>
				<tbody>
					{#if filteredConversations.length === 0}
						<tr>
							<td colspan="3" class="p-8 text-center text-sm text-muted-foreground">
								{#if searchQuery}
									No conversations found matching "{searchQuery}"
								{:else}
									No conversations available
								{/if}
							</td>
						</tr>
					{:else}
						{#each filteredConversations as conv (conv.id)}
							<tr
								class="cursor-pointer border-b transition-colors hover:bg-muted/50"
								onclick={(e) => toggleConversation(conv.id, e.shiftKey)}
							>
								<td class="p-3">
									<Checkbox
										checked={selectedIds.has(conv.id)}
										onclick={(e) => {
											e.preventDefault();
											e.stopPropagation();
											toggleConversation(conv.id, e.shiftKey);
										}}
									/>
								</td>

								<td class="p-3 text-sm">
									<div class="max-w-[17rem] truncate" title={conv.name || 'Untitled conversation'}>
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

	<div class="flex justify-end gap-2">
		<Button variant="outline" onclick={handleCancel}>Cancel</Button>

		<Button onclick={handleConfirm} disabled={selectedIds.size === 0}>
			{mode === 'export' ? 'Export' : 'Import'} ({selectedIds.size})
		</Button>
	</div>
</div>
