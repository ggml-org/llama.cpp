<script lang="ts">
	import ProjectSection from './ProjectSection.svelte';
	import { Pencil } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Textarea } from '$lib/components/ui/textarea';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';

	interface Props {
		instructions: string;
		onSave: (instructions: string) => void;
	}

	let { instructions, onSave }: Props = $props();

	let draft = $state('');
	let isEditing = $state(false);

	const hasInstructions = $derived(instructions.trim().length > 0);

	function startEditing() {
		draft = instructions;
		isEditing = true;
	}

	function cancelEditing() {
		isEditing = false;
		draft = '';
	}

	function save() {
		onSave(draft.trim());
		isEditing = false;
	}
</script>

<ProjectSection title="Instructions">
	{#snippet action()}
		{#if !isEditing}
			<ActionIcon
				ariaLabel="Edit instructions"
				icon={Pencil}
				iconSize={ICON_CLASS_DEFAULT}
				onclick={startEditing}
				tooltip="Edit instructions"
			/>
		{/if}
	{/snippet}

	{#if isEditing}
		<div class="flex flex-col gap-2">
			<Textarea
				bind:value={draft}
				class="min-h-28 text-sm"
				placeholder="Add instructions to tailor the model's responses"
			/>

			<div class="flex justify-end gap-2">
				<Button onclick={cancelEditing} size="sm" type="button" variant="ghost">Cancel</Button>

				<Button onclick={save} size="sm" type="button">Save</Button>
			</div>
		</div>
	{:else if hasInstructions}
		<p class="line-clamp-[12] text-sm whitespace-pre-wrap text-muted-foreground">{instructions}</p>
	{:else}
		<p class="text-sm text-muted-foreground">Add instructions to tailor the model's responses</p>
	{/if}
</ProjectSection>
