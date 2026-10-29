<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as Drawer from '$lib/components/ui/drawer';

	const { Story } = defineMeta({
		component: Drawer.Root,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Design system/Overlays/Drawer'
	});
</script>

<Story asChild name="Default">
	<div class="grid h-screen place-items-center">
		<Drawer.Root open>
			<Drawer.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="outline">Choose a model</Button>
				{/snippet}
			</Drawer.Trigger>

			<Drawer.Content>
				<div class="mx-auto w-full max-w-sm">
					<Drawer.Header>
						<Drawer.Title>Choose a model</Drawer.Title>

						<Drawer.Description>Used on phones in place of the dropdown.</Drawer.Description>
					</Drawer.Header>

					<div class="space-y-1 px-4 text-sm">
						<div class="rounded-md bg-accent px-3 py-2">Qwen3.8-27B</div>

						<div class="rounded-md px-3 py-2">Llama-3.2-3B</div>
					</div>

					<Drawer.Footer>
						<Drawer.Close>
							{#snippet child({ props })}
								<Button {...props} variant="outline">Close</Button>
							{/snippet}
						</Drawer.Close>
					</Drawer.Footer>
				</div>
			</Drawer.Content>
		</Drawer.Root>
	</div>
</Story>
