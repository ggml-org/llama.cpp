<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';

	const { Story } = defineMeta({
		component: AlertDialog.Root,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Design system/Overlays/AlertDialog'
	});
</script>

<Story asChild name="Default">
	<div class="grid h-screen place-items-center">
		<AlertDialog.Root open>
			<AlertDialog.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="destructive">Delete conversation</Button>
				{/snippet}
			</AlertDialog.Trigger>

			<AlertDialog.Content>
				<AlertDialog.Header>
					<AlertDialog.Title>Delete this conversation?</AlertDialog.Title>

					<AlertDialog.Description>
						All of its messages are removed from this browser. This cannot be undone.
					</AlertDialog.Description>
				</AlertDialog.Header>

				<AlertDialog.Footer>
					<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

					<AlertDialog.Action>Delete</AlertDialog.Action>
				</AlertDialog.Footer>
			</AlertDialog.Content>
		</AlertDialog.Root>
	</div>
</Story>
