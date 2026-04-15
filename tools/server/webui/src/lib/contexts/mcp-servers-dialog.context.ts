import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_MCP_SERVERS_DIALOG } from '$lib/constants';

export interface McpServersDialogContext {
	open: () => void;
	isActive: () => boolean;
}

export function setMcpServersDialogContext(ctx: McpServersDialogContext): McpServersDialogContext {
	return setContext(CONTEXT_KEY_MCP_SERVERS_DIALOG, ctx);
}

export function getMcpServersDialogContext(): McpServersDialogContext {
	return getContext(CONTEXT_KEY_MCP_SERVERS_DIALOG);
}
