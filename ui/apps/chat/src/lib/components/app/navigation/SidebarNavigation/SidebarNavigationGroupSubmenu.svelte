<script lang="ts">
	import { Check, FolderInput, Plus } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app';
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import { MenuKind } from '$lib/enums';
	import { groupsStore, uiStore } from '$lib/stores';
	import { tick } from 'svelte';

	interface Props {
		/** Group the chat already belongs to, ticked in the list. */
		currentGroupId?: string | null;
		/** The menu the submenu sits in. */
		kind?: MenuKind;
		label: string;
		/** Closes the parent menu after a pick. */
		onClose?: () => void;
		/** Called with the picked group, or with the one created from here. */
		onSelect: (groupId: string) => void;
	}

	let {
		currentGroupId = null,
		kind = MenuKind.DROPDOWN,
		label,
		onClose,
		onSelect
	}: Props = $props();

	let Menu = $derived(kind === MenuKind.CONTEXT ? ContextMenu : DropdownMenu);

	let query = $state('');
	let searchInput = $state<HTMLInputElement | null>(null);

	const groups = $derived.by(() => {
		const needle = query.trim().toLowerCase();
		const all = [...groupsStore.groups].sort((a, b) => a.name.localeCompare(b.name));

		return needle ? all.filter((group) => group.name.toLowerCase().includes(needle)) : all;
	});

	function select(groupId: string) {
		onSelect(groupId);
		onClose?.();
	}

	function handleNewGroup() {
		uiStore.openGroupDialog({ onCreate: (group) => onSelect(group.id) });
		onClose?.();
	}

	async function handleOpenChange(open: boolean) {
		if (!open) {
			query = '';

			return;
		}

		await tick();
		searchInput?.focus({ preventScroll: true });
	}

	// the menu's own typeahead would swallow the keystrokes typed in the search
	function handleSearchKeydown(event: KeyboardEvent) {
		if (event.key !== 'Escape') event.stopPropagation();
	}
</script>

<Menu.Sub onOpenChange={(open) => void handleOpenChange(open)}>
	<Menu.SubTrigger class="flex cursor-pointer items-center gap-2">
		<FolderInput class={ICON_CLASS_DEFAULT} />

		<span class="flex-1 whitespace-nowrap">{label}</span>
	</Menu.SubTrigger>

	<Menu.SubContent
		alignOffset={-6}
		class="w-64 p-1.5"
		onOpenAutoFocus={(event) => event.preventDefault()}
		sideOffset={6}
	>
		<SearchInput
			bind:ref={searchInput}
			bind:value={query}
			onKeyDown={handleSearchKeydown}
			placeholder="Search groups"
			size="sm"
		/>

		<div aria-hidden="true" class="mx-1 my-1.5 h-px bg-border/40"></div>

		<button
			class="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
			onclick={handleNewGroup}
			type="button"
		>
			<Plus class={ICON_CLASS_DEFAULT} />

			<span>New group</span>
		</button>

		{#if groups.length > 0}
			<div aria-hidden="true" class="mx-1 my-1.5 h-px bg-border/40"></div>

			<div class="flex max-h-64 flex-col gap-0.5 overflow-y-auto">
				{#each groups as group (group.id)}
					<button
						class="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
						onclick={() => select(group.id)}
						type="button"
					>
						<span class="min-w-0 flex-1 truncate">{group.name}</span>

						{#if group.id === currentGroupId}
							<Check class="h-4 w-4 shrink-0 text-primary" />
						{/if}
					</button>
				{/each}
			</div>
		{/if}
	</Menu.SubContent>
</Menu.Sub>
