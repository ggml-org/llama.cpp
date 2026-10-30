<script lang="ts">
	import { MoreHorizontal, Pencil, Pin, PinOff, Trash2 } from '@lucide/svelte';
	import { ContextMenuActions, DialogConfirmation, DropdownMenuActions } from '$lib/components/app';
	import * as Card from '$lib/components/ui/card';
	import { projectsStore, uiStore } from '$lib/stores';
	import type { MenuAction, Project } from '$lib/types';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	const MINUTE = 60_000;
	const HOUR = 60 * MINUTE;
	const DAY = 24 * HOUR;

	let contextMenuOpen = $state(false);
	let confirmDeleteOpen = $state(false);

	function open() {
		uiStore.openProjectDialog({ projectId: project.id });
	}

	async function handleDelete() {
		confirmDeleteOpen = false;
		await projectsStore.remove(project.id);
	}

	// the "more actions" dropdown and the right-click menu offer the same actions
	const actions = $derived<MenuAction[]>([
		{ icon: Pencil, label: 'Edit project', onclick: open },
		{
			icon: project.pinned ? PinOff : Pin,
			label: project.pinned ? 'Unpin' : 'Pin',
			onclick: () => void projectsStore.setPinned(project.id, !project.pinned)
		},
		{
			icon: Trash2,
			label: 'Delete',
			onclick: () => (confirmDeleteOpen = true),
			separator: true,
			variant: 'destructive'
		}
	]);

	function formatUpdatedAt(timestamp: number): string {
		const elapsed = Date.now() - timestamp;

		if (elapsed < MINUTE) return 'just now';

		if (elapsed < HOUR) return `${Math.floor(elapsed / MINUTE)}m ago`;

		if (elapsed < DAY) return `${Math.floor(elapsed / HOUR)}h ago`;

		if (elapsed < 30 * DAY) return `${Math.floor(elapsed / DAY)}d ago`;

		return new Date(timestamp).toLocaleDateString();
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key !== 'Enter' && event.key !== ' ') return;

		event.preventDefault();
		open();
	}
</script>

<ContextMenuActions bind:open={contextMenuOpen} {actions}>
	<Card.Root
		class="group !gap-0 min-h-28 cursor-pointer p-5 transition-colors hover:bg-muted/50 {contextMenuOpen
			? 'bg-muted/50'
			: 'bg-muted/30'}"
		onclick={open}
		onkeydown={handleKeydown}
		role="button"
		tabindex={0}
	>
		<div class="flex items-start gap-2">
			<h3 class="min-w-0 flex-1 truncate font-medium">{project.name}</h3>

			<!-- the menu's clicks stay in it, not opening the card -->
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
			<div
				class="shrink-0 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
				onclick={(event) => event.stopPropagation()}
			>
				<DropdownMenuActions
					{actions}
					triggerIcon={MoreHorizontal}
					triggerTooltip="Project actions"
				/>
			</div>
		</div>

		{#if project.description}
			<p class="mt-1 line-clamp-2 text-sm text-muted-foreground">{project.description}</p>
		{/if}

		<div class="mt-auto flex items-center gap-2 pt-6 text-sm text-muted-foreground">
			<p class="min-w-0 flex-1">{formatUpdatedAt(project.updatedAt)}</p>

			{#if project.pinned}
				<Pin aria-label="Pinned" class="h-3.5 w-3.5 shrink-0 opacity-60" role="img" />
			{/if}
		</div>
	</Card.Root>
</ContextMenuActions>

<DialogConfirmation
	confirmText="Delete"
	description="Its chats stay in your history, outside any project."
	onCancel={() => (confirmDeleteOpen = false)}
	onConfirm={handleDelete}
	open={confirmDeleteOpen}
	title="Delete {project.name}?"
	variant="destructive"
/>
