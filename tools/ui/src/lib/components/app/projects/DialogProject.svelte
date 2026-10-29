<script lang="ts">
	import ProjectContextList from './ProjectContextList.svelte';
	import ProjectFolderPicker from './ProjectFolderPicker.svelte';
	import { DialogConfirmation } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Textarea } from '$lib/components/ui/textarea';
	import { PROJECT_NAME_MAX_LENGTH } from '$lib/constants';
	import { projectsStore, uiStore } from '$lib/stores';
	import type { ProjectDraft } from '$lib/types';
	import { estimateProjectContext } from '$lib/utils';
	import { untrack } from 'svelte';

	const compactNumber = new Intl.NumberFormat('en', {
		maximumFractionDigits: 1,
		notation: 'compact'
	});

	let options = $derived(uiStore.projectDialog);
	let project = $derived(options?.projectId ? projectsStore.get(options.projectId) : undefined);

	let draft = $state<ProjectDraft>(emptyDraft());
	let confirmDeleteOpen = $state(false);
	let nameInput = $state<HTMLInputElement | null>(null);

	// a fresh draft every time the dialog opens, from the project when editing
	$effect.pre(() => {
		if (!options) return;

		untrack(() => (draft = project ? $state.snapshot(projectDraft(project)) : emptyDraft()));
	});

	const canSave = $derived(draft.name.trim().length > 0);
	// what every chat in the project starts with, so a small context window is not a surprise
	const budget = $derived(estimateProjectContext(draft));

	function emptyDraft(): ProjectDraft {
		return { contextItems: [], cwd: null, description: '', instructions: '', name: '' };
	}

	function projectDraft({ contextItems, cwd, description, instructions, name }: ProjectDraft) {
		return { contextItems, cwd, description, instructions, name };
	}

	function close() {
		uiStore.projectDialog = null;
	}

	// each handler reads the dialog's options before closing it, which clears them
	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();

		if (!canSave || !options) return;

		const { onCreate } = options;
		const editedId = project?.id;
		const changes = $state.snapshot(draft);

		close();

		if (editedId) {
			await projectsStore.update(editedId, changes);

			return;
		}

		const created = await projectsStore.create(changes);

		if (created) onCreate?.(created);
	}

	async function handleToggleHidden() {
		if (!project) return;

		const { hidden, id } = project;

		close();
		await projectsStore.setHidden(id, !hidden);
	}

	async function handleDelete() {
		const id = project?.id;

		if (!id) return;

		confirmDeleteOpen = false;
		close();
		await projectsStore.remove(id);
	}
</script>

<Dialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<Dialog.Content
		class="max-h-[90dvh] max-w-xl! overflow-y-auto"
		onOpenAutoFocus={(event) => {
			event.preventDefault();
			nameInput?.focus();
		}}
	>
		<Dialog.Header>
			<Dialog.Title class="text-2xl">{project ? 'Edit project' : 'New project'}</Dialog.Title>
		</Dialog.Header>

		<form class="contents" onsubmit={handleSubmit}>
			<div class="flex flex-col gap-5">
				<div class="flex flex-col gap-2">
					<Label for="project-name">Name</Label>

					<Input
						bind:ref={nameInput}
						bind:value={draft.name}
						autocomplete="off"
						id="project-name"
						maxlength={PROJECT_NAME_MAX_LENGTH}
						placeholder="What are you working on?"
					/>
				</div>

				<div class="flex flex-col gap-2">
					<Label for="project-description">Goal</Label>

					<Textarea
						bind:value={draft.description}
						class="field-sizing-content min-h-16"
						id="project-description"
						placeholder="What are you trying to achieve?"
					/>
				</div>

				<div class="flex flex-col gap-2">
					<Label>Working directory</Label>

					<ProjectFolderPicker cwd={draft.cwd} onChange={(cwd) => (draft.cwd = cwd)} />
				</div>

				<div class="flex flex-col gap-2">
					<Label for="project-instructions">Instructions</Label>

					<Textarea
						bind:value={draft.instructions}
						class="field-sizing-content max-h-64 min-h-24"
						id="project-instructions"
						placeholder="How should the model work in this project?"
					/>
				</div>

				<ProjectContextList bind:items={draft.contextItems} />

				{#if budget.tokens > 0}
					<p class="text-xs text-muted-foreground">
						~{compactNumber.format(budget.tokens)} tokens added to every chat{budget.readOnDemand >
						0
							? `, ${budget.readOnDemand} read on demand`
							: ''}
					</p>
				{/if}
			</div>

			<Dialog.Footer class="gap-2 sm:justify-between">
				{#if project}
					<div class="flex gap-2">
						<Button onclick={handleToggleHidden} size="sm" variant="ghost">
							{project.hidden ? 'Show in sidebar' : 'Hide'}
						</Button>

						<Button
							class="text-destructive hover:text-destructive"
							onclick={() => (confirmDeleteOpen = true)}
							size="sm"
							variant="ghost"
						>
							Delete
						</Button>
					</div>
				{/if}

				<div class="flex gap-2 sm:ml-auto">
					<Button onclick={close} size="sm" variant="secondary">Cancel</Button>

					<Button disabled={!canSave} size="sm" type="submit">
						{project ? 'Save' : 'Create project'}
					</Button>
				</div>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>

<DialogConfirmation
	confirmText="Delete"
	description="Its chats stay in your history, outside any project."
	onCancel={() => (confirmDeleteOpen = false)}
	onConfirm={handleDelete}
	open={confirmDeleteOpen}
	title="Delete {project?.name ?? 'project'}?"
	variant="destructive"
/>
