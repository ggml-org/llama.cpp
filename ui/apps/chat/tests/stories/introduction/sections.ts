/**
 * The Storybook's top-level sections and the groups under each, in sidebar order, for the Introduction's Sections
 * figure. Storybook reads the sidebar order from `storySort` in .storybook/preview.ts, which it parses as written, so
 * this list cannot be imported there; tests/unit/storybook-sections.test.ts checks that the two match.
 */
export interface StorybookSection {
	/** What the section holds, in one line. */
	holds: string;
	name: string;
	/** The groups under it, as the sidebar lists them. */
	pages: string[];
}

export const STORYBOOK_SECTIONS: StorybookSection[] = [
	{
		holds: 'What this is, and how it fits together',
		name: 'Introduction',
		pages: []
	},
	{
		holds: 'The rules, for agents',
		name: 'Agents',
		pages: ['Design guidelines']
	},
	{
		holds: 'The library: its foundations, components and Model UI',
		name: 'Llama UI',
		pages: ['Foundations', 'Components', 'App components', 'Models', 'Mockups']
	},
	{
		holds: 'The app: its screens and the parts they are built from',
		name: 'Llama Chat',
		pages: [
			'Foundations',
			'Mobile',
			'Chat',
			'Content',
			'Models',
			'MCP',
			'Navigation',
			'Settings',
			'Dialogs',
			'Forms',
			'Actions',
			'Badges',
			'Misc',
			'Servers',
			'PWA'
		]
	}
];
