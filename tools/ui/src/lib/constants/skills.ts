import { BookOpen, List } from '@lucide/svelte';
import { JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition, SkillToolSetting } from '$lib/types';

/** Working-directory header for Skills requests. */
export const X_SKILL_CWD_HEADER = 'x-skill-cwd';

/** Model-facing list tool name. */
export const SKILL_LIST_TOOL = 'list_skill' as const;

/** Model-facing read tool name. */
export const SKILL_READ_TOOL = 'read_skill' as const;

/** Skills source label used by consent UI. */
export const SKILL_SERVER_LABEL = 'Skills';

/** Model-facing read tool description. */
export const SKILL_READ_TOOL_DESCRIPTION =
	'Read the current base content of a skill by name, or one of its resources by a relative path.';

/** Model-facing list tool description. */
export const SKILL_LIST_TOOL_DESCRIPTION =
	'List the skills available in this run, with their descriptions.';

/** Build the read tool definition, optionally scoped to a name snapshot. */
export function buildSkillReadToolDefinition(names?: readonly string[]): OpenAIToolDefinition {
	return {
		function: {
			description: SKILL_READ_TOOL_DESCRIPTION,
			name: SKILL_READ_TOOL,
			parameters: {
				properties: {
					name: names ? { enum: [...names], type: 'string' } : { type: 'string' },
					path: { type: 'string' }
				},
				required: ['name'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}

/** Build the list tool definition. */
export function buildSkillListToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description: SKILL_LIST_TOOL_DESCRIPTION,
			name: SKILL_LIST_TOOL,
			parameters: {
				properties: {},
				required: [],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}

/** Deep-freeze a Skills tool definition. */
export function freezeSkillToolDefinition(def: OpenAIToolDefinition): OpenAIToolDefinition {
	return Object.freeze({
		...def,
		function: Object.freeze({
			...def.function,
			parameters: Object.freeze({ ...def.function.parameters })
		})
	});
}

/** Settings-only rows for the Skills adapters. */
export const SKILL_TOOL_SETTINGS: readonly SkillToolSetting[] = Object.freeze([
	{
		definition: freezeSkillToolDefinition(buildSkillReadToolDefinition()),
		description: SKILL_READ_TOOL_DESCRIPTION,
		icon: BookOpen,
		key: 'skill:read_skill',
		label: 'Read skill',
		toolName: SKILL_READ_TOOL
	},
	{
		definition: freezeSkillToolDefinition(buildSkillListToolDefinition()),
		description: SKILL_LIST_TOOL_DESCRIPTION,
		icon: List,
		key: 'skill:list_skill',
		label: 'List skills',
		toolName: SKILL_LIST_TOOL
	}
]);
