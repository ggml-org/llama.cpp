/** A page of a phone sheet, as the Mobile flows page draws it. */
export interface FlowPage {
	/** What the page holds, in a few words. */
	note: string;
	title: string;
}

/** One way through a sheet: the pages a tap pushes, in order. */
export interface Flow {
	pages: FlowPage[];
	/** What opens the sheet. */
	trigger: string;
}

export const SETTINGS_FLOW: Flow = {
	pages: [
		{ note: 'Theme tiles, then a row per section', title: 'Settings' },
		{ note: 'Switches, pickers and numbers in groups', title: 'A section' },
		{ note: 'From Enable MCP, in Agentic', title: 'MCP servers' }
	],
	trigger: 'Settings key, bottom left of the sidebar'
};

export const SETTINGS_TOOLS_FLOW: Flow = {
	pages: [
		{ note: 'The Tools row', title: 'Settings' },
		{ note: 'A row and a switch per group', title: 'Tools' },
		{ note: "The group's tools, each with a switch", title: 'A tool group' },
		{ note: 'Enabled, Always allow, Shown argument', title: 'A tool' }
	],
	trigger: 'Settings key, then Tools'
};

export const ADD_FLOW: Flow = {
	pages: [
		{ note: 'Camera, files, system message, working directory', title: 'Add to chat' },
		{ note: 'Built-in groups, then MCP servers', title: 'Tools' },
		{ note: 'All tools, then a switch per tool', title: 'A tool group' }
	],
	trigger: '+ in the chat form'
};

export const MODELS_FLOW: Flow = {
	pages: [
		{ note: "The chat's model and those in reach", title: 'Select model' },
		{ note: 'Loaded, Favorites, then the rest in groups', title: 'All models' }
	],
	trigger: 'The model name in the chat form'
};
