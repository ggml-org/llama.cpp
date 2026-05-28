import type { AgenticConfig } from '$lib/types/agentic';
import { JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const ATTACHMENT_SAVED_REGEX = /\[Attachment saved: ([^\]]+)\]/;

export const NEWLINE_SEPARATOR = '\n';

export const DEFAULT_AGENTIC_CONFIG: AgenticConfig = {
	enabled: true,
	maxTurns: 100,
	maxToolPreviewLines: 25
} as const;

export const AGENTIC_QUESTION_TOOL_NAME = 'question';
export const AGENTIC_TODO_WRITE_TOOL_NAME = 'todowrite';
export const AGENTIC_ARTIFACT_CREATE_TOOL_NAME = 'artifact_create';
export const AGENTIC_ARTIFACT_EDIT_TOOL_NAME = 'artifact_edit';

export const AGENTIC_QUESTION_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: AGENTIC_QUESTION_TOOL_NAME,
		description:
			'Ask the user one or more clarifying questions before continuing. Use this when the assistant needs a concrete user choice or answer to proceed.',
		parameters: {
			type: JsonSchemaType.OBJECT,
			properties: {
				questions: {
					type: 'array',
					description: 'Questions to ask',
					items: {
						type: JsonSchemaType.OBJECT,
						properties: {
							type: {
								type: 'string',
								enum: ['single_choice', 'multiple_choice', 'freeform']
							},
							question: { type: 'string' },
							header: { type: 'string' },
							options: {
								type: 'array',
								items: {
									type: JsonSchemaType.OBJECT,
									properties: {
										label: { type: 'string' },
										description: { type: 'string' }
									},
									required: ['label', 'description']
								}
							},
							multiple: { type: 'boolean' },
							custom: { type: 'boolean' }
						},
						required: ['question', 'header']
					}
				}
			},
			required: ['questions']
		}
	}
} as const;

export const AGENTIC_TODO_WRITE_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: AGENTIC_TODO_WRITE_TOOL_NAME,
		description:
			'Create or update the current task list. Use this to track multi-step work, mark progress, and keep task status current.',
		parameters: {
			type: JsonSchemaType.OBJECT,
			properties: {
				todos: {
					type: 'array',
					description: 'The updated todo list.',
					items: {
						type: JsonSchemaType.OBJECT,
						properties: {
							content: { type: 'string' },
							status: {
								type: 'string',
								enum: ['pending', 'in_progress', 'completed', 'cancelled']
							}
						},
						required: ['content', 'status']
					}
				}
			},
			required: ['todos']
		}
	}
} as const;

const AGENTIC_PRESENTABLE_CONTENT_SCHEMA = {
	name: { type: 'string' },
	mime_type: { type: 'string' },
	content: { type: 'string' },
	content_base64: { type: 'string' }
} as const;

export const AGENTIC_ARTIFACT_CREATE_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: AGENTIC_ARTIFACT_CREATE_TOOL_NAME,
		description:
			'Create a file-like artifact for content the user asked to have as a file or document. Prefer this over replying only with inline text when the user wants a file to open or revise later.',
		parameters: {
			type: JsonSchemaType.OBJECT,
			properties: AGENTIC_PRESENTABLE_CONTENT_SCHEMA,
			required: ['name', 'mime_type']
		}
	}
} as const;

export const AGENTIC_ARTIFACT_EDIT_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: AGENTIC_ARTIFACT_EDIT_TOOL_NAME,
		description:
			'Edit an existing artifact by artifact_id when revising a file previously created with artifact_create. Provide replacement content/content_base64 and optionally a new name or MIME type.',
		parameters: {
			type: JsonSchemaType.OBJECT,
			properties: {
				artifact_id: { type: 'string' },
				...AGENTIC_PRESENTABLE_CONTENT_SCHEMA
			},
			required: ['artifact_id']
		}
	}
} as const;

export const REASONING_TAGS = {
	START: '<think>',
	END: '</think>'
} as const;

/**
 * @deprecated Legacy marker tags - only used for migration of old stored messages.
 * New messages use structured fields (reasoningContent, toolCalls, toolCallId).
 */
export const LEGACY_AGENTIC_TAGS = {
	TOOL_CALL_START: '<<<AGENTIC_TOOL_CALL_START>>>',
	TOOL_CALL_END: '<<<AGENTIC_TOOL_CALL_END>>>',
	TOOL_NAME_PREFIX: '<<<TOOL_NAME:',
	TOOL_ARGS_START: '<<<TOOL_ARGS_START>>>',
	TOOL_ARGS_END: '<<<TOOL_ARGS_END>>>',
	TAG_SUFFIX: '>>>'
} as const;

/**
 * @deprecated Legacy reasoning tags - only used for migration of old stored messages.
 * New messages use the dedicated reasoningContent field.
 */
export const LEGACY_REASONING_TAGS = {
	START: '<<<reasoning_content_start>>>',
	END: '<<<reasoning_content_end>>>'
} as const;

/**
 * @deprecated Legacy regex patterns - only used for migration of old stored messages.
 */
export const LEGACY_AGENTIC_REGEX = {
	COMPLETED_TOOL_CALL:
		/<<<AGENTIC_TOOL_CALL_START>>>\n<<<TOOL_NAME:(.+?)>>>\n<<<TOOL_ARGS_START>>>([\s\S]*?)<<<TOOL_ARGS_END>>>([\s\S]*?)<<<AGENTIC_TOOL_CALL_END>>>/g,
	REASONING_BLOCK: /<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g,
	REASONING_EXTRACT: /<<<reasoning_content_start>>>([\s\S]*?)<<<reasoning_content_end>>>/,
	REASONING_OPEN: /<<<reasoning_content_start>>>[\s\S]*$/,
	AGENTIC_TOOL_CALL_BLOCK: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*?<<<AGENTIC_TOOL_CALL_END>>>/g,
	AGENTIC_TOOL_CALL_OPEN: /\n*<<<AGENTIC_TOOL_CALL_START>>>[\s\S]*$/,
	HAS_LEGACY_MARKERS: /<<<(?:AGENTIC_TOOL_CALL_START|reasoning_content_start)>>>/
} as const;
