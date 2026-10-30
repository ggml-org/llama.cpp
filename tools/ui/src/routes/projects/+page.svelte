<script lang="ts">
	import { Plus } from '@lucide/svelte';
	import { ProjectCard } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { APP_NAME, PROJECT_ICON } from '$lib/constants';
	import { projectsStore, uiStore } from '$lib/stores';

	/** Placeholder cards while the projects table is read. */
	const PROJECT_CARD_SKELETONS = [0, 1, 2];

	// the most recently edited first
	let projects = $derived([...projectsStore.projects].sort((a, b) => b.updatedAt - a.updatedAt));
</script>

<svelte:head>
	<title>Projects - {APP_NAME}</title>
</svelte:head>

<div class="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 pt-12 pb-16 md:px-8 md:pt-6">
	<div class="flex items-start gap-4">
		<div class="flex min-w-0 flex-col gap-1">
			<h1 class="text-2xl font-semibold tracking-tight">Projects</h1>

			<p class="text-sm text-muted-foreground">
				Instructions, context and a folder shared by every chat about the same work.
			</p>
		</div>

		<Button class="ml-auto shrink-0 gap-1.5" onclick={() => uiStore.openProjectDialog()} size="sm">
			<Plus class="h-3.5 w-3.5" />

			New project
		</Button>
	</div>

	{#if !projectsStore.isInitialized && projects.length === 0}
		<div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
			{#each PROJECT_CARD_SKELETONS as index (index)}
				<Skeleton class="h-28 rounded-xl" />
			{/each}
		</div>
	{:else if projects.length === 0}
		<Empty.Root class="border py-16">
			<Empty.Header>
				<Empty.Media variant="icon">
					<PROJECT_ICON />
				</Empty.Media>

				<Empty.Title>Create your first project</Empty.Title>

				<Empty.Description>
					Keep instructions and context in one place, shared by every chat about the same work.
				</Empty.Description>
			</Empty.Header>

			<Empty.Content>
				<Button onclick={() => uiStore.openProjectDialog()} size="sm">
					<Plus class="h-4 w-4" />
					New project
				</Button>
			</Empty.Content>
		</Empty.Root>
	{:else}
		<div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
			{#each projects as project (project.id)}
				<ProjectCard {project} />
			{/each}
		</div>
	{/if}
</div>
