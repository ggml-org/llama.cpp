<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { SettingsSheet } from '#lib/components/app/settings/index.ts';
	import {
		SETTINGS_CHAT_SECTIONS,
		SETTINGS_SECTION_SLUGS,
		SETTINGS_SHEET_TEXT
	} from '#lib/constants/index.ts';
	import { screen } from 'storybook/test';

	// a phone's settings: a full-height sheet. Its first page has the theme tiles and a row per
	// section; a row pushes that section's page, whose Advanced row pushes the settings few people
	// need, and every change applies as it is made
	const { Story } = defineMeta({
		component: SettingsSheet,
		globals: { viewport: { isRotated: false, value: 'mobile2' } },
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Settings/SettingsSheet'
	});

	const titleOf = (slug: string) =>
		SETTINGS_CHAT_SECTIONS.find((section) => section.slug === slug)?.title ?? slug;

	/** Opens a section's page from the first page, as a tap on its row does. */
	function openSection(slug: string) {
		return async ({ userEvent }: { userEvent: { click: (element: Element) => Promise<void> } }) => {
			await userEvent.click(await screen.findByRole('button', { name: titleOf(slug) }));
			await screen.findByRole('button', { name: 'Back' });
		};
	}
</script>

<Story args={{ open: true }} name="First page" />

<Story args={{ open: true }} name="General" play={openSection(SETTINGS_SECTION_SLUGS.GENERAL)} />

<Story args={{ open: true }} name="Voice" play={openSection(SETTINGS_SECTION_SLUGS.VOICE)} />

<Story args={{ open: true }} name="Data" play={openSection(SETTINGS_SECTION_SLUGS.DATA)} />

<Story
	args={{ open: true }}
	name="Advanced page"
	play={async (context) => {
		await openSection(SETTINGS_SECTION_SLUGS.CHAT)(context);
		await context.userEvent.click(
			await screen.findByRole('button', { name: SETTINGS_SHEET_TEXT.ADVANCED })
		);
		await screen.findByRole('heading', { name: SETTINGS_SHEET_TEXT.ADVANCED });
	}}
/>

<Story
	args={{ open: true }}
	name="Back to the first page"
	play={async (context) => {
		await openSection(SETTINGS_SECTION_SLUGS.GENERAL)(context);
		await context.userEvent.click(screen.getByRole('button', { name: 'Back' }));
		await screen.findByText(SETTINGS_SHEET_TEXT.APPEARANCE);
	}}
/>
