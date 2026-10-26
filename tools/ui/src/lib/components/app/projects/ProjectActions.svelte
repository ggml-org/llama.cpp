<script lang="ts">
	import { Archive, ArchiveRestore, Download, MoreHorizontal, Pin, Trash2 } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { ActionIcon, DropdownMenuActions } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { projectsStore } from '$lib/stores';
	import type { MenuAction, Project } from '$lib/types';

	interface Props {
		project: Project;
	}

	let { project }: Props = $props();

	async function deleteProject() {
		const confirmed = window.confirm(`Delete "${project.name}"? This action cannot be undone.`);

		if (!confirmed) return;

		await projectsStore.remove(project.id);
		await goto(ROUTES.PROJECTS);
	}

	const actions = $derived<MenuAction[]>([
		{
			icon: project.archived ? ArchiveRestore : Archive,
			label: project.archived ? 'Unarchive' : 'Archive',
			onclick: (): void => void projectsStore.setArchived(project.id, !project.archived)
		},
		{
			icon: Trash2,
			label: 'Delete',
			onclick: (): void => void deleteProject(),
			variant: 'destructive'
		},
		{
			icon: Download,
			label: 'Export',
			onclick: (): void => void projectsStore.exportArchive(project.id),
			separator: true
		}
	]);
</script>

<!-- pin and the project menu sit by the title they act on -->
<div class="flex shrink-0 items-center gap-1">
	{#if !project.archived}
		<ActionIcon
			ariaLabel={project.pinned ? 'Unpin project' : 'Pin project'}
			class="h-8 w-8 shrink-0 {project.pinned ? 'text-foreground' : 'text-muted-foreground'}"
			icon={Pin}
			iconSize="h-4 w-4"
			onclick={() => void projectsStore.setPinned(project.id, !project.pinned)}
			tooltip={project.pinned ? 'Unpin project' : 'Pin project'}
			tooltipAsTitle
		/>
	{/if}

	<DropdownMenuActions {actions} triggerIcon={MoreHorizontal} triggerTooltip="Project actions" />
</div>
