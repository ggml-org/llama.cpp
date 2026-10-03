/**
 * WebRTC tunnel configuration constants
 */

/**
 * Public WebTorrent trackers used as the signaling rendezvous. The offer is
 * announced on all of them at once and the first answer wins.
 */
export const TRACKER_URLS = [
	'wss://tracker.openwebtorrent.com',
	'wss://tracker.webtorrent.dev'
] as const;

/** STUN servers used to discover the public address for NAT traversal. */
export const STUN_URLS = ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302'] as const;

/**
 * Deadlines for each step of establishing the tunnel.
 */
export const WEBRTC_TIMEOUTS = {
	/** Deadline for the host to answer and the pass code to be accepted */
	CONNECT_MS: 30_000,
	/** Deadline for gathering ICE candidates before the offer goes out */
	ICE_GATHER_MS: 10_000,
	/** Deadline for the WebSocket handshake with a tracker */
	TRACKER_CONNECT_MS: 10_000
} as const;

/**
 * Lengths of the codes shared with the user. The share code is the room code
 * followed by the pass code.
 */
export const CODE_LENGTHS = {
	/** WebTorrent trackers require an info_hash of exactly this length */
	INFO_HASH: 20,
	/** Secret authenticating the client on the data channel */
	PASS: 32,
	/** Rendezvous key, padded to INFO_HASH */
	ROOM: 8,
	/** Room and pass code concatenated, what the user pastes */
	SHARE: 40
} as const;

/**
 * Lengths of the random identifiers exchanged over the wire.
 */
export const ID_LENGTHS = {
	/** Correlates an offer with its answer */
	OFFER: 20,
	/** Identifies this peer to the trackers */
	PEER: 20,
	/** Correlates a tunneled request with its response */
	REQUEST: 16
} as const;

/** Characters that are unambiguous to read aloud or type. */
export const CODE_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
