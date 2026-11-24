/**
 * agenticMode - Whether the agentic features are on
 *
 * They are on when the build has them (see AGENTIC_ENABLED) and llama-server
 * offers what they need: its built-in tools or its proxy, which `--agent`
 * turns on together. Without them Llama Chat is a plain chat: a message and
 * its reply, with no tools, no MCP and no agentic loop.
 *
 * The server says so in two places: `/props` reports the proxy, and `/tools`
 * lists the built-in tools, which the tools store reports here.
 */

import { AGENTIC_ENABLED } from '#lib/constants/app.constants.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';

class AgenticMode {
	/** Set by the tools store once the server has listed its built-in tools. */
	serverHasTools = $state(false);

	/** Set by the tools store once the server has answered about its tools, either way. */
	serverToolsKnown = $state(false);

	get enabled(): boolean {
		return (
			AGENTIC_ENABLED && (this.serverHasTools || (serverStore.props?.cors_proxy_enabled ?? false))
		);
	}

	/** True once the server has said what it offers, so a page can act on `enabled` being false. */
	get settled(): boolean {
		return !AGENTIC_ENABLED || (serverStore.props !== null && this.serverToolsKnown);
	}
}

export const agenticMode = new AgenticMode();
