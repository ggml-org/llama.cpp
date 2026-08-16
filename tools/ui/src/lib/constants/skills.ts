import { BookOpen, List } from '@lucide/svelte';
import { JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition, SkillToolSetting } from '$lib/types';

/** HTTP header carrying the working directory Skills requests resolve under; an absent header means the canonical server process CWD. */
export const X_SKILL_CWD_HEADER = 'x-skill-cwd';

/** Model-facing tool name for the Skills list adapter. */
export const SKILL_LIST_TOOL = 'list_skill' as const;

/** Model-facing tool name for the Skills read adapter. */
export const SKILL_READ_TOOL = 'read_skill' as const;

/** Display label for the Skills tool source in the established consent UI. */
export const SKILL_SERVER_LABEL = 'Skills';

/** Model-facing description of `read_skill`, shared by run definitions and settings rows. */
export const SKILL_READ_TOOL_DESCRIPTION =
	'Read the current base content of a skill by name, or one of its resources by a relative path.';

/** Model-facing description of `list_skill`, shared by run definitions and settings rows. */
export const SKILL_LIST_TOOL_DESCRIPTION =
	'List the skills available in this run, with their descriptions.';

/**
 * Build the model-facing `read_skill` definition.
 *
 * The optional `names` argument adds the frozen snapshot name `enum` and is
 * used ONLY for run-scoped definitions. The no-argument form is the static
 * settings-only display definition and MUST never carry a static name enum
 * or be sent to a model.
 */
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

/** Build the no-argument `list_skill` definition. */
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

/** Deep-freeze a Skills tool definition (type, function, and parameters). */
export function freezeSkillToolDefinition(def: OpenAIToolDefinition): OpenAIToolDefinition {
	return Object.freeze({
		...def,
		function: Object.freeze({
			...def.function,
			parameters: Object.freeze({ ...def.function.parameters })
		})
	});
}

/**
 * Settings-only rows for the model-facing Skills adapters, consumed by the
 * Chat tool settings tab. Keys use the stable `skill:<tool>` form, distinct
 * from generic builtin/MCP/custom selection keys. Skills entries are NOT
 * ordinary model tools: they never enter `allTools`, `toolGroups`, or
 * `getEnabledToolsForLLM()`, and their definitions are display-only.
 */
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
