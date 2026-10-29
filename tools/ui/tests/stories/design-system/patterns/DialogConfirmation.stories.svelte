<script lang="ts" module>
	import { Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { DialogConfirmation } from '$lib/components/app';

	// the shared confirm dialog; Enter confirms
	const { Story } = defineMeta({
		component: DialogConfirmation,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/DialogConfirmation'
	});
</script>

<script lang="ts">
	let defaultOpen = $state(true);
	let destructiveOpen = $state(true);
</script>

<Story asChild name="Default">
	<DialogConfirmation
		bind:open={defaultOpen}
		confirmText="Continue"
		description="The model will be unloaded and the new one loaded in its place."
		onCancel={() => (defaultOpen = false)}
		onConfirm={() => (defaultOpen = false)}
		title="Switch model?"
	/>
</Story>

<Story asChild name="Destructive">
	<DialogConfirmation
		bind:open={destructiveOpen}
		confirmText="Delete"
		description="This conversation and its messages will be removed from this browser."
		icon={Trash2}
		onCancel={() => (destructiveOpen = false)}
		onConfirm={() => (destructiveOpen = false)}
		title="Delete conversation?"
		variant="destructive"
	/>
</Story>
