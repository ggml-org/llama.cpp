<script lang="ts" module>
	import * as ContextMenu from '@ggml-org/llama-ui/ui/context-menu';
	import * as DropdownMenu from '@ggml-org/llama-ui/ui/dropdown-menu';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect, screen, waitFor, within } from 'storybook/test';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Overlays/Submenus'
	});

	const ROWS = Array.from({ length: 12 }, (_, index) => `Recent chat ${index + 1}`);

	/**
	 * A submenu opens beside a menu that scrolls and draws as glass. Its first item has to be
	 * what the pointer hits at its centre: when the submenu stays inside the menu's clipping box,
	 * the item is laid out but cut off, and the point lands on whatever is behind the menu.
	 *
	 * It also has to look like the menu it opens from (the same surface and corners), with its
	 * first row level with the row that opens it.
	 */
	async function expectSubmenuLikeMenu(trigger: HTMLElement) {
		const submenu = await waitFor(() => {
			const element = document.querySelector<HTMLElement>('[data-slot$="menu-sub-content"]');

			expect(element).not.toBeNull();

			return element!;
		});
		const item = within(submenu).getByRole('menuitem', { name: 'Work' });

		await waitFor(() => {
			const { height, left, top, width } = item.getBoundingClientRect();
			const hit = document.elementFromPoint(left + width / 2, top + height / 2);

			expect(item.contains(hit), 'the submenu is cut off by the menu it opens from').toBe(true);
		});

		const menu = trigger.closest<HTMLElement>('[role="menu"]')!;
		const menuStyle = getComputedStyle(menu);
		const submenuStyle = getComputedStyle(submenu);

		expect(submenuStyle.borderRadius).toBe(menuStyle.borderRadius);
		expect(submenuStyle.backdropFilter).toBe(menuStyle.backdropFilter);
		expect(submenuStyle.backgroundColor).toBe(menuStyle.backgroundColor);
		expect(submenuStyle.boxShadow).toBe(menuStyle.boxShadow);
		// the submenu zooms and slides in, so its rows settle a moment after it shows
		await waitFor(() =>
			expect(item.getBoundingClientRect().top).toBeCloseTo(trigger.getBoundingClientRect().top, 0)
		);
	}
</script>

<Story
	asChild
	name="DropdownMenuSubmenu"
	play={async ({ userEvent }) => {
		const trigger = await screen.findByRole('menuitem', { name: 'Move to' });

		await userEvent.hover(trigger);

		await expectSubmenuLikeMenu(trigger);
	}}
>
	<div class="h-96 w-[32rem]">
		<DropdownMenu.Root open>
			<DropdownMenu.Trigger>Conversation</DropdownMenu.Trigger>

			<DropdownMenu.Content align="start" class="max-h-56 w-52" side="bottom">
				<DropdownMenu.Sub>
					<DropdownMenu.SubTrigger>Move to</DropdownMenu.SubTrigger>

					<DropdownMenu.SubContent>
						<DropdownMenu.Item>Work</DropdownMenu.Item>

						<DropdownMenu.Item>Personal</DropdownMenu.Item>
					</DropdownMenu.SubContent>
				</DropdownMenu.Sub>

				<DropdownMenu.Separator />

				{#each ROWS as row (row)}
					<DropdownMenu.Item>{row}</DropdownMenu.Item>
				{/each}
			</DropdownMenu.Content>
		</DropdownMenu.Root>
	</div>
</Story>

<Story
	asChild
	name="ContextMenuSubmenu"
	play={async ({ canvas, userEvent }) => {
		const area = await canvas.findByText('Right-click here');
		const { height, left, top, width } = area.getBoundingClientRect();

		await userEvent.pointer({
			coords: { clientX: left + width / 2, clientY: top + height / 2 },
			keys: '[MouseRight]',
			target: area
		});

		const trigger = await screen.findByRole('menuitem', { name: 'Move to' });

		await userEvent.hover(trigger);

		await expectSubmenuLikeMenu(trigger);
	}}
>
	<div class="h-96 w-[32rem]">
		<ContextMenu.Root>
			<ContextMenu.Trigger
				class="flex h-24 w-52 items-center justify-center rounded-lg border border-dashed text-body-sm text-muted-foreground"
			>
				Right-click here
			</ContextMenu.Trigger>

			<ContextMenu.Content class="max-h-56 w-52">
				<ContextMenu.Sub>
					<ContextMenu.SubTrigger>Move to</ContextMenu.SubTrigger>

					<ContextMenu.SubContent>
						<ContextMenu.Item>Work</ContextMenu.Item>

						<ContextMenu.Item>Personal</ContextMenu.Item>
					</ContextMenu.SubContent>
				</ContextMenu.Sub>

				<ContextMenu.Separator />

				{#each ROWS as row (row)}
					<ContextMenu.Item>{row}</ContextMenu.Item>
				{/each}
			</ContextMenu.Content>
		</ContextMenu.Root>
	</div>
</Story>
