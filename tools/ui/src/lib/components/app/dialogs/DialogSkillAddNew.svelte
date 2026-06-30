<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Textarea } from '$lib/components/ui/textarea';
	import { skillsStore } from '$lib/stores/skills.svelte';
	import {
		validateSkillName,
		validateSkillDescription,
		normalizeSkillName
	} from '$lib/utils/skill-format';

	interface Props {
		open: boolean;
		onOpenChange?: (open: boolean) => void;
		initialContent?: string;
		onAddSkillComplete?: (skillId: string, name: string) => void;
	}

	let {
		open = $bindable(),
		onOpenChange,
		initialContent = '',
		onAddSkillComplete
	}: Props = $props();

	let name = $state('');
	let description = $state('');
	let content = $state('');

	let existingNames = $derived(skillsStore.getSkills().map((s) => ({ id: s.id, name: s.name })));
	let nameError = $derived.by(() => {
		const normalized = normalizeSkillName(name);
		const nameErr = validateSkillName(normalized);
		if (nameErr) return nameErr;
		const dup = existingNames.find(
			(s) => normalizeSkillName(s.name).toLowerCase() === normalized.toLowerCase()
		);
		return dup ? 'A skill with this name already exists' : null;
	});
	let descriptionError = $derived.by(() => validateSkillDescription(description));
	let contentError = $derived.by(() => (!content.trim() ? 'Content is required' : null));
	let saveError = $derived.by(() => nameError || contentError || descriptionError);

	$effect(() => {
		if (open && initialContent) {
			content = initialContent;
		}
	});

	function handleOpenChange(value: boolean) {
		if (!value) {
			name = '';
			description = '';
			content = '';
		}
		open = value;
		onOpenChange?.(value);
	}

	async function saveNewSkill() {
		if (saveError) return;

		const newSkill = await skillsStore.addSkill({
			name: normalizeSkillName(name),
			description: description.trim(),
			content: content.trim()
		});
		handleOpenChange(false);
		onAddSkillComplete?.(newSkill.id, newSkill.name);
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-2xl">
		<Dialog.Header>
			<Dialog.Title>Add skill</Dialog.Title>

			<Dialog.Description>
				Save a reusable instruction template. A skill can be inserted into any chat. Mark it
				always-on from the Skills page to auto-include it in every new conversation's system prompt.
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-6 py-2">
			<div class="space-y-2">
				<Label for="skill-name" class="text-sm font-medium">Name</Label>

				<Input
					id="skill-name"
					type="text"
					bind:value={name}
					placeholder="e.g. Code Review Assistant"
					class="w-full"
				/>

				{#if nameError}
					<p class="text-xs text-destructive">{nameError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="skill-description" class="text-sm font-medium">
					Description <span class="text-destructive">*</span>
				</Label>

				<Input
					id="skill-description"
					type="text"
					bind:value={description}
					placeholder="A short summary suitable for the system prompt"
					class="w-full"
				/>

				{#if descriptionError}
					<p class="text-xs text-destructive">{descriptionError}</p>
				{/if}
			</div>

			<div class="space-y-2">
				<Label for="skill-content" class="text-sm font-medium">Content</Label>

				<Textarea
					id="skill-content"
					bind:value={content}
					placeholder="Enter skill content..."
					class="min-h-[10rem] w-full"
				/>

				{#if contentError}
					<p class="text-xs text-destructive">{contentError}</p>
				{/if}
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			<Button size="sm" onclick={saveNewSkill} disabled={!!saveError}>Add</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
