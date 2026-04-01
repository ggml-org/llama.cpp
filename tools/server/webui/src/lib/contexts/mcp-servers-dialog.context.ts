import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_MCP_SERVERS_DIALOG } from '$lib/constants';

export interface McpServersDialogContext {
	open: () => void;
	isActive: () => boolean;
}

const MCP_SERVERS_DIALOG_KEY = Symbol.for(CONTEXT_KEY_MCP_SERVERS_DIALOG);

export function setMcpServersDialogContext(ctx: McpServersDialogContext): McpServersDialogContext {
	return setContext(MCP_SERVERS_DIALOG_KEY, ctx);
}

export function getMcpServersDialogContext(): McpServersDialogContext {
	return getContext(MCP_SERVERS_DIALOG_KEY);
}
