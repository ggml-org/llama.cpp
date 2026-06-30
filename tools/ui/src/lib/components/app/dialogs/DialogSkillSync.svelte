<script lang="ts">
	import { ArrowRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import Label from '$lib/components/ui/label/label.svelte';
	import { CodeDiff } from '$lib/components/app/content';
	import {
		normalizeSkillName,
		validateSkillName,
		validateSkillDescription
	} from '$lib/utils/skill-format';

	interface Props {
		open: boolean;
		skillName?: string;
		currentTitle?: string;
		currentContent: string;
		updatedContent?: string;
		updatedTitle?: string;
		updatedDescription?: string;
		editableTitle?: boolean;
		editableDescription?: boolean;
		/** When true, the left side of the diff is the LIBRARY content
		 *  and the right side is the MESSAGE content (i.e. the message has
		 *  been edited away from the library). The primary action then
		 *  pushes the message content back to the library. */
		messageEdited?: boolean;
		/** When true, the left side of the diff is the MESSAGE content
		 *  and the right side is the LIBRARY content (i.e. the library has
		 *  been updated away from the message). The secondary action then
		 *  overwrites the message with the library content. */
		libraryEdited?: boolean;
		onOpenChange?: (open: boolean) => void;
		onUpdate?: (title?: string, description?: string) => void;
		onUseLibraryVersion?: () => void;
	}

	let {
		open = $bindable(),
		skillName = 'this skill',
		currentTitle,
		currentContent = '',
		updatedContent = '',
		updatedTitle = $bindable(),
		updatedDescription = $bindable(),
		editableTitle = false,
		editableDescription = false,
		messageEdited = false,
		libraryEdited = false,
		onOpenChange,
		onUpdate,
		onUseLibraryVersion
	}: Props = $props();

	let showBothDirections = $derived(messageEdited && libraryEdited);
	let showTitleRenamed = $derived(
		!!currentTitle && !!skillName && skillName !== 'this skill' && currentTitle !== skillName
	);

	// Initialise editable title with the library name on first open.
	$effect(() => {
		if (!open) return;
		if (editableTitle && (updatedTitle === undefined || updatedTitle === '')) {
			updatedTitle = skillName;
		}
	});

	let titleError = $derived.by(() => {
		if (!editableTitle) return null;
		const value = (updatedTitle ?? '').trim();
		if (!value) return 'Title is required';
		const normalized = normalizeSkillName(value);
		return validateSkillName(normalized);
	});

	let descriptionError = $derived.by(() => {
		if (!editableDescription) return null;
		return validateSkillDescription((updatedDescription ?? '').trim());
	});

	let hasErrors = $derived(!!titleError || !!descriptionError);

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
	}

	function handleUpdate() {
		if (hasErrors) return;
		onUpdate?.(
			editableTitle ? (updatedTitle ?? '').trim() : undefined,
			editableDescription ? (updatedDescription ?? '').trim() : undefined
		);
	}

	function handleUseLibraryVersion() {
		onUseLibraryVersion?.();
	}
</script>

<Dialog.Root {open} onOpenChange={handleOpenChange}>
	<Dialog.Content class="sm:max-w-3xl">
		<Dialog.Header>
			<Dialog.Title>Modified skill</Dialog.Title>

			<Dialog.Description>
				{showBothDirections
					? 'Both the library and the conversation version have diverged. Choose which one to keep, or edit the title/description before saving.'
					: messageEdited
						? 'This conversation has been edited away from the library skill. Save the changes back to the library, or restore the library version.'
						: 'The library version has been updated. Apply the new library version to this conversation, or keep your changes and save them back to the library.'}
			</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-4 py-2">
			{#if showTitleRenamed}
				<div
					class="flex items-center gap-2 rounded-md border border-dashed bg-muted/40 px-3 py-2 text-sm"
				>
					<span class="font-medium text-muted-foreground">Library title:</span>

					<span class="text-muted-foreground line-through">{currentTitle}</span>

					<ArrowRight class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

					<span class="font-medium">{skillName}</span>
				</div>
			{/if}

			{#if editableTitle}
				<div class="space-y-2">
					<Label for="sync-skill-title" class="text-sm font-medium">
						Title <span class="text-destructive">*</span>
					</Label>

					<Input id="sync-skill-title" type="text" bind:value={updatedTitle} class="w-full" />

					{#if titleError}
						<p class="text-xs text-destructive">{titleError}</p>
					{/if}
				</div>
			{/if}

			{#if editableDescription}
				<div class="space-y-2">
					<Label for="sync-skill-description" class="text-sm font-medium">
						Description <span class="text-destructive">*</span>
					</Label>

					<Input
						id="sync-skill-description"
						type="text"
						bind:value={updatedDescription}
						placeholder="A short summary suitable for the system prompt"
						class="w-full"
					/>

					{#if descriptionError}
						<p class="text-xs text-destructive">{descriptionError}</p>
					{/if}
				</div>
			{/if}

			{#if updatedContent && currentContent !== updatedContent}
				<div class="space-y-1">
					<div class="flex items-center justify-between text-xs text-muted-foreground">
						<span>{messageEdited ? 'Library' : 'Conversation'}</span>
						<span>{messageEdited ? 'Conversation' : 'Library'}</span>
					</div>

					<div class="overflow-x-auto">
						<CodeDiff oldContent={currentContent} newContent={updatedContent} maxHeight="50vh" />
					</div>
				</div>
			{/if}
		</div>

		<Dialog.Footer class="gap-2">
			<Button variant="secondary" size="sm" onclick={() => handleOpenChange(false)}>Cancel</Button>

			{#if libraryEdited || showBothDirections}
				<Button variant="outline" size="sm" onclick={handleUseLibraryVersion} disabled={hasErrors}>
					Use library version
				</Button>
			{/if}

			<Button variant="default" size="sm" onclick={handleUpdate} disabled={hasErrors}>
				{messageEdited || showBothDirections ? 'Save to library' : 'Update conversation'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
