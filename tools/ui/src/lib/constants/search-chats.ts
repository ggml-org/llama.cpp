import {
	BuiltInTool,
	JsonSchemaType,
	SearchChatsMessageSelection,
	SearchChatsOrder,
	ToolCallType,
	ToolSource
} from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const SEARCH_CHATS_TOOL_NAME = BuiltInTool.BROWSER_SEARCH_CHATS;

/** Key of the tool in the disabled tools lists, same format as toolsStore.toolKey */
export const SEARCH_CHATS_TOOL_KEY = `${ToolSource.BROWSER}:${SEARCH_CHATS_TOOL_NAME}`;

export const SEARCH_CHATS_OUTPUT_MAX_CHARS = 8192;

export const SEARCH_CHATS_LIMIT_CHATS_DEFAULT = 10;
export const SEARCH_CHATS_LIMIT_CHATS_MAX = 1000;

export const SEARCH_CHATS_LIMIT_MESSAGES_DEFAULT = 5;
export const SEARCH_CHATS_LIMIT_MESSAGES_MAX = 100;

export const SEARCH_CHATS_MESSAGE_CHARS_DEFAULT = 600;
export const SEARCH_CHATS_MESSAGE_CHARS_MIN = 50;
export const SEARCH_CHATS_MESSAGE_CHARS_MAX = 4000;

export const SEARCH_CHATS_MAX_PATTERNS = 10;
export const SEARCH_CHATS_MAX_IDS = 100;

/** Matching cost grows with pattern length times text length, and runs on the main thread */
export const SEARCH_CHATS_PATTERN_MAX_CHARS = 200;
export const SEARCH_CHATS_ID_MAX_CHARS = 200;

/** A match past this share of the excerpt length moves the excerpt window to the match */
export const SEARCH_CHATS_EXCERPT_RECENTER_RATIO = 0.6;

/** Share of the excerpt shown before the match when the window is moved */
export const SEARCH_CHATS_EXCERPT_LEAD_RATIO = 1 / 3;

export const SEARCH_CHATS_ELLIPSIS = '...';

export const SEARCH_CHATS_FIELD_SEPARATOR = ' | ';

export const SEARCH_CHATS_CUT_NOTICE = '...[cut: lower limit_messages or max_message_chars]';

const DESCRIPTION = `Find and read the user's other saved chats (read-only). The current chat is hidden, you already see it.
Returns chats, each with messages from what the user saw: user and assistant text only (tool output is skipped), numbered #1..#N in chat order. Dates are in the user's local time.
- List chats: limit_messages=0 gives one line per chat (title, id, date); raise limit_chats and page with offset_chats.
- Find chats: by_matches=["%paris%", "%eiffel%"] matches the title or any message (case-insensitive, % = any text, _ = one char, no % means "contains"). A chat matches if any pattern does.
- Read chats: by_ids=["<id>"] with message_selection "latest" (default without by_matches) or "earliest".
Example, "what was my last question in the chat about paris?": by_matches=["%paris%"], message_selection="latest", limit_messages=4, then answer from the last user message.`;

export function buildSearchChatsToolDefinition(): OpenAIToolDefinition {
	return {
		function: {
			description: DESCRIPTION,
			name: SEARCH_CHATS_TOOL_NAME,
			parameters: {
				properties: {
					by_ids: {
						description: 'Only these chat ids',
						items: { type: JsonSchemaType.STRING },
						type: JsonSchemaType.ARRAY
					},
					by_matches: {
						description: `LIKE patterns matched against titles and message text, any of them, max ${SEARCH_CHATS_PATTERN_MAX_CHARS} chars each`,
						items: { type: JsonSchemaType.STRING },
						type: JsonSchemaType.ARRAY
					},
					include_current: {
						description: 'Also include the current chat, default false',
						type: JsonSchemaType.BOOLEAN
					},
					limit_chats: {
						description: `Max chats, default ${SEARCH_CHATS_LIMIT_CHATS_DEFAULT}, max ${SEARCH_CHATS_LIMIT_CHATS_MAX}`,
						type: JsonSchemaType.NUMBER
					},
					limit_messages: {
						description: `Messages per chat, default ${SEARCH_CHATS_LIMIT_MESSAGES_DEFAULT}, max ${SEARCH_CHATS_LIMIT_MESSAGES_MAX}; 0 lists chats only`,
						type: JsonSchemaType.NUMBER
					},
					max_message_chars: {
						description: `Characters shown per message, default ${SEARCH_CHATS_MESSAGE_CHARS_DEFAULT}, max ${SEARCH_CHATS_MESSAGE_CHARS_MAX}`,
						type: JsonSchemaType.NUMBER
					},
					message_selection: {
						description: `Which messages per chat, "${SearchChatsMessageSelection.MATCHING}" is the default with by_matches`,
						enum: Object.values(SearchChatsMessageSelection),
						type: JsonSchemaType.STRING
					},
					offset_chats: {
						description: 'Skip this many chats, for paging',
						type: JsonSchemaType.NUMBER
					},
					order_timestamp: {
						description: `Chat order by last activity, default "${SearchChatsOrder.DESC}" (newest first)`,
						enum: Object.values(SearchChatsOrder),
						type: JsonSchemaType.STRING
					}
				},
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}
