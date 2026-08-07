import { SET_WORKING_DIRECTORY_LABEL } from '$lib/constants/working-directory';
import { ChatFormCommandAction } from '$lib/enums';
import type { ChatFormCommand } from '$lib/types';

interface ChatCommandsOptions {
	/** Gates `/model`. */
	showModelSelector: boolean;
	/** Gates `/prompt`. */
	hasPrompts: () => boolean;
	/** Gates `/cwd`. */
	hasBuiltinTools: () => boolean;
}

/**
 * The slash commands surfaced by the `/` command picker, in display order.
 *
 * Availability is supplied as predicates rather than store imports: this
 * module is re-exported through the `$lib/constants` barrel, and importing
 * stores at module load would create a circular dependency (the stores
 * themselves import from `$lib/constants`).
 */
export function getChatCommands(options: ChatCommandsOptions): ChatFormCommand[] {
	return [
		{
			name: 'prompt',
			description: 'Insert an MCP prompt',
			action: ChatFormCommandAction.PROMPT,
			disabled: !options.hasPrompts()
		},
		{
			name: 'cwd',
			description: SET_WORKING_DIRECTORY_LABEL,
			keywords: ['current working directory'],
			action: ChatFormCommandAction.CWD,
			disabled: !options.hasBuiltinTools()
		},
		{
			name: 'model',
			description: 'Select model',
			action: ChatFormCommandAction.MODEL,
			disabled: !options.showModelSelector
		}
	];
}
