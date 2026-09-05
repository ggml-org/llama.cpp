/**
 * Stored server type enum - discriminates the entries of the server registry
 */
export enum ServerType {
	/** Reached through a WebRTC tunnel, addressed by its share code */
	TUNNEL = 'tunnel'
}

/**
 * Remote access mode enum - whether requests go through the tunnel
 */
export enum RemoteAccessMode {
	/** Requests are tunneled to a remote llama-connect host */
	CLIENT = 'client',
	/** No remote session, requests reach the server serving this page */
	OFF = 'off'
}

/**
 * Tunnel connection status enum
 */
export enum TunnelStatus {
	/** Data channel open and authenticated */
	CONNECTED = 'connected',
	/** Handshake in progress, requests are queued */
	CONNECTING = 'connecting',
	/** Connection failed or dropped, requests are rejected */
	ERROR = 'error',
	/** No connection attempted */
	IDLE = 'idle'
}
