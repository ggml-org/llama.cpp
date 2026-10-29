<script lang="ts">
	import { Check, Plus } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app';
	import * as Dialog from '$lib/components/ui/dialog';
	import { groupsStore, uiStore } from '$lib/stores';

	let options = $derived(uiStore.groupPicker);
	let query = $state('');

	const groups = $derived.by(() => {
		const needle = query.trim().toLowerCase();
		const all = [...groupsStore.groups].sort((a, b) => a.name.localeCompare(b.name));

		return needle ? all.filter((group) => group.name.toLowerCase().includes(needle)) : all;
	});

	const newGroupName = $derived(query.trim());

	function close() {
		uiStore.groupPicker = null;
		query = '';
	}

	function select(groupId: string) {
		const onSelect = options?.onSelect;

		close();
		onSelect?.(groupId);
	}

	async function createAndSelect() {
		if (!newGroupName) return;

		const group = await groupsStore.create(newGroupName);

		if (group) select(group.id);
	}
</script>

<Dialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<Dialog.Content class="gap-0 overflow-hidden p-0 md:max-w-md!">
		<Dialog.Title class="sr-only">Move to group</Dialog.Title>

		<div class="p-2">
			<SearchInput
				bind:value={query}
				autofocus
				onKeyDown={(event) => {
					if (event.key === 'Enter' && newGroupName) void createAndSelect();
				}}
				placeholder="Search or create a group"
				size="sm"
			/>
		</div>

		<div class="flex max-h-80 flex-col gap-0.5 overflow-y-auto p-1">
			<button
				class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
				disabled={!newGroupName}
				onclick={createAndSelect}
				type="button"
			>
				<Plus class="h-4 w-4 shrink-0" />

				<span class="min-w-0 truncate">
					{newGroupName ? `Create "${newGroupName}"` : 'New group'}
				</span>
			</button>

			{#if groups.length > 0}
				<div aria-hidden="true" class="mx-2 my-1 h-px bg-border/40"></div>

				{#each groups as group (group.id)}
					<button
						class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
						onclick={() => select(group.id)}
						type="button"
					>
						<span class="min-w-0 flex-1 truncate">{group.name}</span>

						{#if group.id === options?.currentGroupId}
							<Check class="h-4 w-4 shrink-0 text-primary" />
						{/if}
					</button>
				{/each}
			{:else if newGroupName}
				<p class="px-2 py-2 text-sm text-muted-foreground">No matching groups</p>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
