import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

/** Browser tools that read and edit the user's projects, offered once a project exists. */
export const PROJECT_TOOL_NAMES: readonly string[] = [
	BuiltInTool.BROWSER_LIST_PROJECTS,
	BuiltInTool.BROWSER_GET_PROJECT,
	BuiltInTool.BROWSER_UPDATE_PROJECT,
	BuiltInTool.BROWSER_ADD_PROJECT_CONTEXT
];

const PROJECT_ID_PROPERTY = {
	description:
		"Id of the project, from list_projects. Defaults to the current chat's project when the chat is in one.",
	type: JsonSchemaType.STRING
};

function tool(
	name: string,
	description: string,
	properties: Record<string, { description: string; type: JsonSchemaType }>
): OpenAIToolDefinition {
	return {
		function: {
			description,
			name,
			parameters: { properties, required: [], type: JsonSchemaType.OBJECT }
		},
		type: ToolCallType.FUNCTION
	};
}

export function buildProjectToolDefinitions(): OpenAIToolDefinition[] {
	return [
		tool(
			BuiltInTool.BROWSER_LIST_PROJECTS,
			"Lists the user's projects with their id, name, goal and number of chats. Projects group chats that share instructions and context files.",
			{}
		),
		tool(
			BuiltInTool.BROWSER_GET_PROJECT,
			"Returns a project's name, goal, instructions, context items and working directory.",
			{ project_id: PROJECT_ID_PROPERTY }
		),
		tool(
			BuiltInTool.BROWSER_UPDATE_PROJECT,
			'Updates a project: renames it, sets its goal, and replaces or extends its instructions. Instructions are added to every new chat in the project. Only the given fields change.',
			{
				append_instructions: {
					description: 'Text added after the current instructions, on a new line.',
					type: JsonSchemaType.STRING
				},
				goal: {
					description: 'What the project is trying to achieve, in a sentence.',
					type: JsonSchemaType.STRING
				},
				instructions: {
					description: 'Replaces all of the instructions. Use append_instructions to keep them.',
					type: JsonSchemaType.STRING
				},
				name: { description: 'New name of the project.', type: JsonSchemaType.STRING },
				project_id: PROJECT_ID_PROPERTY
			}
		),
		tool(
			BuiltInTool.BROWSER_ADD_PROJECT_CONTEXT,
			'Adds context to a project, which new chats in it start with: either a file on the server, referenced by its absolute path, or a text note saved with the project. Give either path, or note_name with note_content.',
			{
				note_content: { description: 'Text of the note.', type: JsonSchemaType.STRING },
				note_name: {
					description: 'File name for the note, e.g. "decisions.md".',
					type: JsonSchemaType.STRING
				},
				path: {
					description: 'Absolute path of a server file to reference.',
					type: JsonSchemaType.STRING
				},
				project_id: PROJECT_ID_PROPERTY
			}
		)
	];
}
