<script lang="ts">
	import { Check, Plus } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app';
	import * as Dialog from '$lib/components/ui/dialog';
	import { projectsStore, uiStore } from '$lib/stores';

	let options = $derived(uiStore.projectPicker);
	let query = $state('');

	const projects = $derived.by(() => {
		const needle = query.trim().toLowerCase();
		// an archived project would take the chat off the sidebar with it
		const all = projectsStore.projects
			.filter((project) => !project.archived)
			.sort((a, b) => a.name.localeCompare(b.name));

		return needle ? all.filter((project) => project.name.toLowerCase().includes(needle)) : all;
	});

	const newProjectName = $derived(query.trim());

	function close() {
		uiStore.projectPicker = null;
		query = '';
	}

	function select(projectId: string) {
		const onSelect = options?.onSelect;

		close();
		onSelect?.(projectId);
	}

	async function createAndSelect() {
		if (!newProjectName) return;

		const project = await projectsStore.create({
			contextItems: [],
			cwd: null,
			description: '',
			instructions: '',
			name: newProjectName
		});

		if (project) select(project.id);
	}
</script>

<Dialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<Dialog.Content class="gap-0 overflow-hidden p-0 md:max-w-md!">
		<Dialog.Title class="sr-only">Move to project</Dialog.Title>

		<div class="p-2">
			<SearchInput
				bind:value={query}
				autofocus
				onKeyDown={(event) => {
					if (event.key === 'Enter' && newProjectName) void createAndSelect();
				}}
				placeholder="Search or create a project"
				size="sm"
			/>
		</div>

		<div class="flex max-h-80 flex-col gap-0.5 overflow-y-auto p-1">
			<button
				class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent disabled:pointer-events-none disabled:opacity-50"
				disabled={!newProjectName}
				onclick={createAndSelect}
				type="button"
			>
				<Plus class="h-4 w-4 shrink-0" />

				<span class="min-w-0 truncate">
					{newProjectName ? `Create "${newProjectName}"` : 'New project'}
				</span>
			</button>

			{#if projects.length > 0}
				<div aria-hidden="true" class="mx-2 my-1 h-px bg-border/40"></div>

				{#each projects as project (project.id)}
					<button
						class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
						onclick={() => select(project.id)}
						type="button"
					>
						<span class="min-w-0 flex-1 truncate">{project.name}</span>

						{#if project.id === options?.currentProjectId}
							<Check class="h-4 w-4 shrink-0 text-primary" />
						{/if}
					</button>
				{/each}
			{:else if newProjectName}
				<p class="px-2 py-2 text-sm text-muted-foreground">No matching projects</p>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
