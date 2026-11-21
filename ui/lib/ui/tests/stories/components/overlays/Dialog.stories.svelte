<script lang="ts" module>
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Dialog from '@ggml-org/llama-ui/ui/dialog';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Dialog.Root,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama UI/Components/Overlays/Dialog'
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
