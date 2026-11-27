import { CWD_COMMAND_NAME, SET_WORKING_DIRECTORY_LABEL } from '#lib/constants/index.ts';
import { ChatFormCommandAction } from '#lib/enums/index.ts';
import type { ChatCommandsOptions, ChatFormCommand } from '#lib/types/index.ts';

/**
 * The slash commands surfaced by the `/` command picker, in display order.
 *
 * Availability is supplied as predicates rather than store imports: this
 * module is re-exported through the `#lib/utils/index.ts` barrel, and importing
 * stores at module load would create a circular dependency (the stores
 * themselves import from `#lib/utils/index.ts`).
 */
export function getChatCommands(options: ChatCommandsOptions): ChatFormCommand[] {
	return [
		{
			action: ChatFormCommandAction.PROMPT,
			description: 'Insert an MCP prompt',
			disabled: !options.hasPrompts(),
			name: 'prompt',
			takesArgument: true
		},
		{
			action: ChatFormCommandAction.CWD,
			description: SET_WORKING_DIRECTORY_LABEL,
			disabled: !options.hasCwdTools(),
			keywords: ['current working directory'],
			name: CWD_COMMAND_NAME,
			takesArgument: true
		},
		{
			action: ChatFormCommandAction.MODEL,
			description: 'Select model',
			disabled: !options.showModelSelector,
			name: 'model'
		},
		{
			action: ChatFormCommandAction.COMPACT,
			description: 'Summarize to free up context',
			disabled: !options.canCompact(),
			keywords: ['summarize', 'context'],
			name: 'compact'
		},
		{
			action: ChatFormCommandAction.SYSTEM_MESSAGE,
			description: 'Edit the system message',
			disabled: !options.canEditSystemMessage(),
			keywords: ['system prompt', 'instructions'],
			name: 'system-message'
		}
	];
}
