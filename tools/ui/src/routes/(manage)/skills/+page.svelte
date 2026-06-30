<script lang="ts">
	import { Plus, Trash2, Search } from '@lucide/svelte';

	import { Button } from '$lib/components/ui/button';
	import { ManageLayout, SkillCard, SearchInput } from '$lib/components/app';
	import { ROUTES } from '$lib/constants';
	import { SIDEBAR_ACTIONS_ITEMS } from '$lib/constants/ui';
	import { DialogSkillAddNew, DialogConfirmation } from '$lib/components/app/dialogs';

	import { skillsStore } from '$lib/stores/skills.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { isMobile } from '$lib/stores/viewport.svelte';

	let isAdding = $state(false);
	let showDeleteDialog = $state(false);
	let deleteTarget = $state<{ id: string; name: string } | null>(null);
	let searchQuery = $state('');

	let skills = $derived(skillsStore.searchSkills(searchQuery));
	let hasAnySkills = $derived(skillsStore.getSkills().length > 0);

	function handleDelete(id: string, name: string) {
		deleteTarget = { id, name };
		showDeleteDialog = true;
	}

	async function handleConfirmDelete() {
		if (deleteTarget) {
			await skillsStore.deleteSkill(deleteTarget.id);
			deleteTarget = null;
		}
		showDeleteDialog = false;
	}

	async function handleStartNewChat(id: string, name: string, content: string) {
		try {
			await conversationsStore.createConversation();
			await chatStore.addSystemPromptWithContent(content, id, name);
		} catch (error) {
			console.error('Failed to start new chat with skill:', error);
		}
	}
</script>

<ManageLayout title="Skills">
	{#snippet icon()}
		{@const Icon = SIDEBAR_ACTIONS_ITEMS.find((i) => i.route === ROUTES.SKILLS)?.icon}
		{#if Icon}
			<Icon class="h-5 w-5 md:h-6 md:w-6" />
		{/if}
	{/snippet}

	{#snippet actions()}
		{#if hasAnySkills}
			<Button
				variant="outline"
				size={isMobile.current ? 'lg' : 'default'}
				onclick={() => (isAdding = true)}
			>
				<Plus class="h-4 w-4" />
				Add new skill
			</Button>
		{/if}
	{/snippet}

	{#if !hasAnySkills}
		<div class="flex flex-col items-center justify-center py-16">
			<div class="mb-4 rounded-full bg-muted p-4">
				<Plus class="h-6 w-6 text-muted-foreground" />
			</div>
			<h2 class="text-lg font-medium text-foreground">No skills yet</h2>
			<p class="mt-1 text-sm text-muted-foreground">
				Create your first skill. You can insert it into any chat or mark it always-on to
				auto-include it in every new conversation's system prompt.
			</p>
			<Button class="mt-4" onclick={() => (isAdding = true)}>
				<Plus class="h-4 w-4" />
				Add new skill
			</Button>
		</div>
	{:else if skills.length === 0}
		<div class="flex flex-col items-center justify-center py-16">
			<div class="mb-4 rounded-full bg-muted p-4">
				<Search class="h-6 w-6 text-muted-foreground" />
			</div>
			<h2 class="text-lg font-medium text-foreground">No skills match your search</h2>
			<p class="mt-1 text-sm text-muted-foreground">Try a different name or content keyword.</p>
		</div>
	{:else}
		<div class="mb-6">
			<label for="skills-search" class="sr-only">Search skills</label>
			<SearchInput
				id="skills-search"
				bind:value={searchQuery}
				placeholder="Search skills by name, description, or content"
			/>
		</div>

		<div
			class="mt-4 grid gap-6"
			style="grid-template-columns: repeat(auto-fill, minmax(min(32rem, calc(100dvw - 2rem)), 1fr));"
		>
			{#each skills as skill (skill.id)}
				<SkillCard
					id={skill.id}
					name={skill.name}
					description={skill.description}
					content={skill.content}
					lastModified={skill.lastModified}
					path={skill.path}
					origin={skill.origin}
					onDelete={() => handleDelete(skill.id, skill.name)}
					onStartNewChat={() => handleStartNewChat(skill.id, skill.name, skill.content)}
				/>
			{/each}
		</div>
	{/if}

	<DialogSkillAddNew bind:open={isAdding} />

	<DialogConfirmation
		bind:open={showDeleteDialog}
		title="Delete skill?"
		description="This will permanently remove the skill. This action cannot be undone."
		confirmText="Delete"
		cancelText="Cancel"
		variant="destructive"
		icon={Trash2}
		onConfirm={handleConfirmDelete}
		onCancel={() => (showDeleteDialog = false)}
	/>
</ManageLayout>
