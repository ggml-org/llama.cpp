<script lang="ts">
	import { FolderPlus, Pencil } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { GROUP_NAME_MAX_LENGTH } from '$lib/constants';
	import { groupsStore, uiStore } from '$lib/stores';
	import { untrack } from 'svelte';

	let options = $derived(uiStore.groupDialog);
	let group = $derived(options?.groupId ? groupsStore.get(options.groupId) : undefined);

	let name = $state('');
	let nameInput = $state<HTMLInputElement | null>(null);

	// a fresh name every time the dialog opens, the group's own when renaming
	$effect.pre(() => {
		if (!options) return;

		untrack(() => (name = group?.name ?? ''));
	});

	const canSave = $derived(name.trim().length > 0 && name.trim() !== group?.name);

	function close() {
		uiStore.groupDialog = null;
	}

	// reads the dialog's options before closing it, which clears them
	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();

		if (!canSave || !options) return;

		const { onCreate } = options;
		const renamedId = group?.id;

		close();

		if (renamedId) {
			await groupsStore.rename(renamedId, name);

			return;
		}

		const created = await groupsStore.create(name);

		if (created) onCreate?.(created);
	}
</script>

<!-- a short prompt, so a phone shows it as a centered card like an alert, not full screen -->
<AlertDialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<AlertDialog.Content
		class="sm:max-w-md"
		onOpenAutoFocus={(event) => {
			event.preventDefault();
			nameInput?.focus();
			nameInput?.select();
		}}
	>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				{#if group}
					<Pencil class="h-5 w-5" />
				{:else}
					<FolderPlus class="h-5 w-5" />
				{/if}

				{group ? 'Rename group' : 'New group'}
			</AlertDialog.Title>

			<AlertDialog.Description>
				{group ? 'Choose a new name for this group.' : 'Name a group to sort chats into.'}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<form class="contents" onsubmit={handleSubmit}>
			<div class="flex flex-col gap-2">
				<Label for="group-name">Name</Label>

				<Input
					bind:ref={nameInput}
					bind:value={name}
					autocomplete="off"
					id="group-name"
					maxlength={GROUP_NAME_MAX_LENGTH}
					placeholder="e.g. Research"
				/>
			</div>

			<AlertDialog.Footer>
				<!-- not a submit button, or Enter in the name field would cancel -->
				<AlertDialog.Cancel type="button">Cancel</AlertDialog.Cancel>

				<Button disabled={!canSave} type="submit">
					{group ? 'Save' : 'Create group'}
				</Button>
			</AlertDialog.Footer>
		</form>
	</AlertDialog.Content>
</AlertDialog.Root>
