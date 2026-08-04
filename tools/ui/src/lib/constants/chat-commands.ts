import { SET_WORKING_DIRECTORY_LABEL } from '$lib/constants/working-directory';
import type { ChatFormCommand } from '$lib/types';

interface ChatCommandsOptions {
	/** Whether the model selector is rendered in the chat form actions. */
	showModelSelector: boolean;
	/** Whether MCP prompts are reachable (gates `/prompt`). */
	hasPrompts: () => boolean;
	/** Whether built-in tools are present (gates `/cwd`). */
	hasBuiltinTools: () => boolean;
}

/**
 * The slash commands surfaced by the `/` command picker, in display order.
 *
 * Availability is supplied as predicates (rather than importing stores here)
 * so this module stays free of store imports - it is re-exported through the
 * `$lib/constants` barrel, and importing stores at module load would create a
 * circular dependency (the stores themselves import from `$lib/constants`).
 * The host evaluates the predicates per-render so a command's `disabled`
 * state tracks its backing capability live.
 */
export function getChatCommands(options: ChatCommandsOptions): ChatFormCommand[] {
	return [
		{
			name: 'prompt',
			description: 'Insert an MCP prompt',
			action: 'prompt',
			disabled: !options.hasPrompts()
		},
		{
			name: 'cwd',
			description: SET_WORKING_DIRECTORY_LABEL,
			keywords: ['current working directory'],
			action: 'cwd',
			disabled: !options.hasBuiltinTools()
		},
		{
			name: 'model',
			description: 'Select model',
			action: 'model',
			disabled: !options.showModelSelector
		}
	];
}
