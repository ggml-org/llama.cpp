<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	const { Story } = defineMeta({
		component: Dialog.Root,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Design system/Overlays/Dialog'
	});
</script>

<Story asChild name="Default">
	<div class="grid h-screen place-items-center">
		<Dialog.Root open>
			<Dialog.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="outline">Rename conversation</Button>
				{/snippet}
			</Dialog.Trigger>

			<Dialog.Content class="sm:max-w-md">
				<Dialog.Header>
					<Dialog.Title>Rename conversation</Dialog.Title>

					<Dialog.Description>The new name shows in the sidebar and in search.</Dialog.Description>
				</Dialog.Header>

				<div class="grid gap-2">
					<Label for="story-dialog-name">Name</Label>

					<Input id="story-dialog-name" value="Rust borrow checker" />
				</div>

				<Dialog.Footer>
					<Dialog.Close>
						{#snippet child({ props })}
							<Button {...props} variant="outline">Cancel</Button>
						{/snippet}
					</Dialog.Close>

					<Button>Save</Button>
				</Dialog.Footer>
			</Dialog.Content>
		</Dialog.Root>
	</div>
</Story>
