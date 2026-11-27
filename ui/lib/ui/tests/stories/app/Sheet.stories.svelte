<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import {
		InsetList,
		InsetListRow,
		SheetHeader,
		SheetPages,
		useSheetPages
	} from '@ggml-org/llama-ui/app/sheets';
	import { SHEET_CONTENT_CLASS } from '@ggml-org/llama-ui/constants';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Drawer from '@ggml-org/llama-ui/ui/drawer';
	import { Switch } from '@ggml-org/llama-ui/ui/switch';
	import { Bell, Palette } from '@lucide/svelte';
	import { screen } from 'storybook/test';

	// a phone's sheet with pages: a Drawer whose content holds a SheetHeader (close on the first
	// page, back on the pages pushed over it) and SheetPages, which slides between the pages of a
	// useSheetPages stack
	const { Story } = defineMeta({
		globals: { viewport: { isRotated: false, value: 'mobile2' } },
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama UI/App components/Sheet'
	});

	const PAGES = {
		APPEARANCE: 'Appearance',
		NOTIFICATIONS: 'Notifications',
		ROOT: 'Settings'
	} as const;

	type Page = (typeof PAGES)[keyof typeof PAGES];
</script>

<script lang="ts">
	let open = $state(true);

	const pages = useSheetPages<Page>(PAGES.ROOT);

	$effect(() => {
		if (open) pages.reset();
	});
</script>

{#snippet sheet()}
	<div class="p-4">
		<Button onclick={() => (open = true)}>Open the sheet</Button>
	</div>

	<Drawer.Root bind:open>
		<Drawer.Content class="gap-0 {SHEET_CONTENT_CLASS}">
			<SheetHeader canGoBack={pages.canGoBack} onBack={pages.pop} title={pages.page} />

			<Drawer.Description class="sr-only">A sheet with pages.</Drawer.Description>

			<SheetPages class="gap-6 px-3 pb-6" keepFirstPageHeight pageKey={pages.page} {pages}>
				{#if pages.page === PAGES.ROOT}
					<InsetList caption="A row pushes its page; the corner key goes back.">
						<InsetListRow chevron icon={Palette} onclick={() => pages.push(PAGES.APPEARANCE)}>
							<span class="flex-1">{PAGES.APPEARANCE}</span>
						</InsetListRow>

						<InsetListRow chevron icon={Bell} onclick={() => pages.push(PAGES.NOTIFICATIONS)}>
							<span class="flex-1">{PAGES.NOTIFICATIONS}</span>
						</InsetListRow>
					</InsetList>
				{:else}
					<InsetList title={pages.page}>
						<InsetListRow as="label">
							<span class="flex-1">Enabled</span>

							<Switch checked size="lg" />
						</InsetListRow>
					</InsetList>
				{/if}
			</SheetPages>
		</Drawer.Content>
	</Drawer.Root>
{/snippet}

<Story asChild name="First page">
	{@render sheet()}
</Story>

<Story
	asChild
	name="Pushed page"
	play={async ({ userEvent }) => {
		await userEvent.click(await screen.findByRole('button', { name: PAGES.APPEARANCE }));
		await screen.findByRole('button', { name: 'Back' });
	}}
>
	{@render sheet()}
</Story>
