export const APP_NAME = import.meta.env?.VITE_PUBLIC_APP_NAME || 'Llama Chat';

/**
 * False in a chat-only build (`VITE_PUBLIC_AGENTIC=false`): no tools, no MCP and
 * no agentic loop, only a message and its reply. Such a build leaves Llama
 * Harness's loop out of the bundle (see vite.config.ts). A full build still
 * turns the agentic features on only for a server that offers them: read
 * `agenticMode.enabled`, not this constant, to decide what to show.
 */
export const AGENTIC_ENABLED = import.meta.env?.VITE_PUBLIC_AGENTIC !== 'false';
