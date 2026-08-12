/**
 * Persistent memory tool constants.
 *
 * Three browser tools carry a memory for the model across conversations,
 * stored in IndexedDB: memory_open reads the index or the named entries,
 * memory_write creates and edits entries, memory_drop removes one. The
 * protocol matches a file-based memory store: entry names are <group>/<slug>,
 * the index lists name, size in bytes and description, and edits anchor on a
 * unique string of the body.
 */

import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const MEMORY_ENTRY_LIMIT_BYTES_DEFAULT = 49152;

export const MEMORY_GROUPS_DEFAULT_LIST = ['areas', 'people', 'topics'];

export const MEMORY_GROUPS_SEPARATOR = ',';

const MEMORY_GROUPS_JOIN = ', ';

export const MEMORY_GROUPS_DEFAULT = MEMORY_GROUPS_DEFAULT_LIST.join(MEMORY_GROUPS_JOIN);

export const MEMORY_NAME_SEPARATOR = '/';

/** Group names and slugs: lowercase alphanumerics and dashes, no leading dash */
export const MEMORY_NAME_PATTERN = /^[a-z0-9][a-z0-9-]*$/;

export const MEMORY_EMPTY_INDEX = 'Memory is empty';

export const MEMORY_EXPORT_FILENAME_PREFIX = 'llama_memory_';

export const MEMORY_TOOL_NAMES: ReadonlySet<string> = new Set([
	BuiltInTool.BROWSER_MEMORY_OPEN,
	BuiltInTool.BROWSER_MEMORY_WRITE,
	BuiltInTool.BROWSER_MEMORY_DROP
]);

/** Mutation labels a memory_write reports, joined into the result line */
export const MEMORY_DONE = {
	APPENDED: 'appended',
	DESCRIPTION_REFRESHED: 'description refreshed',
	REMOVED: 'removed',
	REPLACED: 'replaced'
} as const;

/** Meaning of the default taxonomy, shown to the model while the groups setting is untouched */
const MEMORY_GROUPS_GUIDANCE =
	'areas for ongoing projects, people for the ones the user works with, topics for lasting facts about them';

/**
 * Parse the comma separated groups setting. A value with no usable group
 * falls back to the default taxonomy, the tools always have at least one
 * group to file entries under.
 */
export function parseMemoryGroups(value: string): string[] {
	const groups = value
		.split(MEMORY_GROUPS_SEPARATOR)
		.map((group) => group.trim())
		.filter((group) => MEMORY_NAME_PATTERN.test(group));

	return groups.length > 0 ? groups : MEMORY_GROUPS_DEFAULT_LIST;
}

function memoryGroupsClause(groups: string[]): string {
	const joined = groups.join(MEMORY_GROUPS_JOIN);

	return joined === MEMORY_GROUPS_DEFAULT ? MEMORY_GROUPS_GUIDANCE : `groups are ${joined}`;
}

/** Build the three memory tool definitions, the write description carrying the configured groups */
export function buildMemoryToolDefinitions(groups: string[]): OpenAIToolDefinition[] {
	return [
		{
			function: {
				description:
					'Read persistent memory. No argument returns the index: one line per entry with its name, size in bytes and description. Pick from it and pass every entry you need in one call. Open the index at the start of a session, and whenever you need context on the user, their projects or the people around them.',
				name: BuiltInTool.BROWSER_MEMORY_OPEN,
				parameters: {
					properties: {
						names: {
							description: `Entries to open, for instance ["${groups[0]}${MEMORY_NAME_SEPARATOR}my-project"]. Omit for the index alone.`,
							items: { type: JsonSchemaType.STRING },
							type: JsonSchemaType.ARRAY
						}
					},
					required: [],
					type: JsonSchemaType.OBJECT
				}
			},
			type: ToolCallType.FUNCTION
		},
		{
			function: {
				description: `Write persistent memory. Name is <group>${MEMORY_NAME_SEPARATOR}<slug>: ${memoryGroupsClause(groups)}. old and new replace a unique string, new alone appends, old alone removes it along with the line break that follows it. new takes as many lines as you have to write. Pass description whenever the edit changes what the entry is about, and when creating one. Store what the user stated, not what you concluded. The index line of the entry comes back with the result.`,
				name: BuiltInTool.BROWSER_MEMORY_WRITE,
				parameters: {
					properties: {
						description: {
							description: 'One line on what this entry covers, shown in the index',
							type: JsonSchemaType.STRING
						},
						name: {
							description: `Entry name, <group>${MEMORY_NAME_SEPARATOR}<slug>`,
							type: JsonSchemaType.STRING
						},
						new: {
							description: 'Replacement text, or the lines to append',
							type: JsonSchemaType.STRING
						},
						old: {
							description: 'Text to replace or remove, must appear exactly once in the body',
							type: JsonSchemaType.STRING
						}
					},
					required: ['name'],
					type: JsonSchemaType.OBJECT
				}
			},
			type: ToolCallType.FUNCTION
		},
		{
			function: {
				description:
					'Remove a whole entry from persistent memory, only when the user asks to forget that subject. To remove a single fact, use memory_write with old alone.',
				name: BuiltInTool.BROWSER_MEMORY_DROP,
				parameters: {
					properties: {
						name: {
							description: `Entry name, <group>${MEMORY_NAME_SEPARATOR}<slug>`,
							type: JsonSchemaType.STRING
						}
					},
					required: ['name'],
					type: JsonSchemaType.OBJECT
				}
			},
			type: ToolCallType.FUNCTION
		}
	];
}
