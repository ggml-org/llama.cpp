<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
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

<Dialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<Dialog.Content
		class="max-w-md!"
		onOpenAutoFocus={(event) => {
			event.preventDefault();
			nameInput?.focus();
			nameInput?.select();
		}}
	>
		<Dialog.Header>
			<Dialog.Title>{group ? 'Rename group' : 'New group'}</Dialog.Title>
		</Dialog.Header>

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

			<Dialog.Footer class="gap-2">
				<Button onclick={close} size="sm" variant="secondary">Cancel</Button>

				<Button disabled={!canSave} size="sm" type="submit">
					{group ? 'Save' : 'Create group'}
				</Button>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>
