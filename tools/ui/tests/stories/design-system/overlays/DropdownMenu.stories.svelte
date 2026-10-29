<script lang="ts" module>
	import { Download, Pencil, Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';

	const { Story } = defineMeta({
		component: DropdownMenu.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Overlays/DropdownMenu'
	});
</script>

<Story asChild name="Default">
	<div class="h-96">
		<DropdownMenu.Root open>
			<DropdownMenu.Trigger>
				{#snippet child({ props })}
					<Button {...props} variant="outline">Conversation</Button>
				{/snippet}
			</DropdownMenu.Trigger>

			<DropdownMenu.Content class="w-56">
				<DropdownMenu.Group>
					<DropdownMenu.Label>Conversation</DropdownMenu.Label>

					<DropdownMenu.Item>
						<Pencil /> Rename
						<DropdownMenu.Shortcut>F2</DropdownMenu.Shortcut>
					</DropdownMenu.Item>

					<DropdownMenu.Item><Download /> Export</DropdownMenu.Item>
				</DropdownMenu.Group>

				<DropdownMenu.Separator />

				<DropdownMenu.CheckboxItem checked>Show statistics</DropdownMenu.CheckboxItem>

				<DropdownMenu.CheckboxItem>Render markdown in user messages</DropdownMenu.CheckboxItem>

				<DropdownMenu.Separator />

				<DropdownMenu.RadioGroup value="medium">
					<DropdownMenu.GroupHeading>Reasoning</DropdownMenu.GroupHeading>

					<DropdownMenu.RadioItem value="low">Low</DropdownMenu.RadioItem>

					<DropdownMenu.RadioItem value="medium">Medium</DropdownMenu.RadioItem>

					<DropdownMenu.RadioItem value="high">High</DropdownMenu.RadioItem>
				</DropdownMenu.RadioGroup>

				<DropdownMenu.Separator />

				<DropdownMenu.Sub>
					<DropdownMenu.SubTrigger>Move to</DropdownMenu.SubTrigger>

					<DropdownMenu.SubContent>
						<DropdownMenu.Item>Work</DropdownMenu.Item>

						<DropdownMenu.Item>Personal</DropdownMenu.Item>
					</DropdownMenu.SubContent>
				</DropdownMenu.Sub>

				<DropdownMenu.Item variant="destructive"><Trash2 /> Delete</DropdownMenu.Item>
			</DropdownMenu.Content>
		</DropdownMenu.Root>
	</div>
</Story>
