import type { ServerType } from '$lib/enums';

/**
 * Server registry types
 */

/**
 * A server the app can talk to. Entries are discriminated by `type` so the
 * registry accepts new kinds of servers without changing the stored shape.
 */
export interface StoredTunnelServer {
	type: ServerType.TUNNEL;
	/** Share code of the llama-connect host, room code followed by pass code */
	code: string;
}

/** Entry of the persisted server registry. */
export type StoredServer = StoredTunnelServer;
