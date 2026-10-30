<script lang="ts">
	import { Check, Plus } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app';
	import * as ContextMenu from '$lib/components/ui/context-menu';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { ICON_CLASS_DEFAULT, PROJECT_ICON } from '$lib/constants';
	import { MenuKind } from '$lib/enums';
	import { projectsStore, uiStore } from '$lib/stores';
	import { lastPathSegment } from '$lib/utils';
	import { tick } from 'svelte';

	interface Props {
		/** Project the row already belongs to, ticked in the list. */
		currentProjectId?: string | null;
		/** The menu the submenu sits in. */
		kind?: MenuKind;
		label: string;
		/** Closes the parent menu after a pick. */
		onClose?: () => void;
		/** Called with the picked project, or with the one created from here. */
		onSelect: (projectId: string) => void;
	}

	let {
		currentProjectId = null,
		kind = MenuKind.DROPDOWN,
		label,
		onClose,
		onSelect
	}: Props = $props();

	let Menu = $derived(kind === MenuKind.CONTEXT ? ContextMenu : DropdownMenu);

	let query = $state('');
	let searchInput = $state<HTMLInputElement | null>(null);

	const projects = $derived.by(() => {
		const needle = query.trim().toLowerCase();
		// an archived project would take the chat off the sidebar with it
		const all = projectsStore.projects
			.filter((project) => !project.archived)
			.sort((a, b) => a.name.localeCompare(b.name));

		return needle ? all.filter((project) => project.name.toLowerCase().includes(needle)) : all;
	});

	function select(projectId: string) {
		onSelect(projectId);
		onClose?.();
	}

	function handleNewProject() {
		uiStore.openProjectDialog({ onCreate: (project) => onSelect(project.id) });
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
		<PROJECT_ICON class={ICON_CLASS_DEFAULT} />

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
			placeholder="Search or create a project"
			size="sm"
		/>

		<div aria-hidden="true" class="mx-1 my-1.5 h-px bg-border/40"></div>

		<button
			class="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
			onclick={handleNewProject}
			type="button"
		>
			<Plus class={ICON_CLASS_DEFAULT} />

			<span>New project</span>
		</button>

		{#if projects.length > 0}
			<div aria-hidden="true" class="mx-1 my-1.5 h-px bg-border/40"></div>

			<div class="flex max-h-64 flex-col gap-0.5 overflow-y-auto">
				{#each projects as project (project.id)}
					<button
						class="flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-accent"
						onclick={() => select(project.id)}
						type="button"
					>
						<PROJECT_ICON class="{ICON_CLASS_DEFAULT} shrink-0" />

						<span class="min-w-0 flex-1">
							<span class="block truncate text-sm">{project.name}</span>

							{#if project.cwd}
								<span class="block truncate text-xs text-muted-foreground">
									{lastPathSegment(project.cwd)}
								</span>
							{/if}
						</span>

						{#if project.id === currentProjectId}
							<Check class="h-4 w-4 shrink-0 text-primary" />
						{/if}
					</button>
				{/each}
			</div>
		{/if}
	</Menu.SubContent>
</Menu.Sub>
