/**
 * WebRTC tunnel for remote llama.cpp access.
 *
 * Signaling uses the WebTorrent tracker WebSocket protocol (no external deps).
 * The room code is used as the info_hash rendezvous key; the pass code
 * authenticates the client on the data channel after the WebRTC handshake.
 *
 * Host:   announces periodically, accepts incoming offers, relays HTTP requests
 *         made by connected clients back to its own local server.
 * Client: sends an offer, authenticates via pass code, then all same-origin
 *         fetch calls are transparently forwarded through the data channel.
 */

const STUN_CONFIG: RTCConfiguration = {
	iceServers: [
		{ urls: 'stun:stun.l.google.com:19302' },
		{ urls: 'stun:stun1.l.google.com:19302' }
	]
};

const TRACKER_URLS = [
	'wss://tracker.openwebtorrent.com',
	'wss://tracker.btorrent.xyz'
];

const ICE_GATHER_TIMEOUT_MS = 10_000;
const ANNOUNCE_INTERVAL_MS = 30_000;
const CONNECT_TIMEOUT_MS = 30_000;
const TRACKER_CONNECT_TIMEOUT_MS = 10_000;

// Characters that are unambiguous to read aloud or type
const CODE_CHARS = 'ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';

function randomStr(len: number): string {
	const bytes = new Uint8Array(len);
	crypto.getRandomValues(bytes);
	let result = '';
	for (const b of bytes) result += CODE_CHARS[b % CODE_CHARS.length];
	return result;
}

export function generateRoomCode(): string {
	return randomStr(8);
}

export function generatePassCode(): string {
	return randomStr(32);
}

// info_hash must be exactly 20 chars for WebTorrent trackers
function roomToInfoHash(roomCode: string): string {
	return roomCode.padEnd(20, '0').slice(0, 20);
}

function waitForIceComplete(pc: RTCPeerConnection): Promise<void> {
	return new Promise((resolve, reject) => {
		if (pc.iceGatheringState === 'complete') {
			resolve();
			return;
		}
		const timer = setTimeout(
			() => reject(new Error('ICE gathering timed out')),
			ICE_GATHER_TIMEOUT_MS
		);
		pc.addEventListener('icegatheringstatechange', () => {
			if (pc.iceGatheringState === 'complete') {
				clearTimeout(timer);
				resolve();
			}
		});
	});
}

// ---------------------------------------------------------------------------
// Tracker signaling (WebTorrent WS tracker protocol)
// ---------------------------------------------------------------------------

type TrackerMsg = Record<string, unknown>;

class Tracker {
	private ws: WebSocket | null = null;
	private readonly infoHash: string;
	private readonly peerId: string;

	onOffer?: (fromPeerId: string, offerId: string, offer: RTCSessionDescriptionInit) => void;
	onAnswer?: (offerId: string, answer: RTCSessionDescriptionInit) => void;
	onClose?: () => void;

	constructor(infoHash: string, peerId: string) {
		this.infoHash = infoHash;
		this.peerId = peerId;
	}

	connect(url: string): Promise<void> {
		return new Promise((resolve, reject) => {
			const ws = new WebSocket(url);
			this.ws = ws;
			const timer = setTimeout(
				() => reject(new Error('tracker connect timeout')),
				TRACKER_CONNECT_TIMEOUT_MS
			);

			ws.onopen = () => {
				clearTimeout(timer);
				resolve();
			};
			ws.onerror = () => {
				clearTimeout(timer);
				reject(new Error('tracker WebSocket error'));
			};
			ws.onclose = () => {
				this.onClose?.();
			};
			ws.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data as string) as TrackerMsg;
					if (msg.offer && msg.peer_id && msg.offer_id) {
						this.onOffer?.(
							msg.peer_id as string,
							msg.offer_id as string,
							msg.offer as RTCSessionDescriptionInit
						);
					} else if (msg.answer && msg.offer_id) {
						this.onAnswer?.(msg.offer_id as string, msg.answer as RTCSessionDescriptionInit);
					}
				} catch {
					// ignore malformed tracker messages
				}
			};
		});
	}

	private send(msg: TrackerMsg): void {
		if (this.ws?.readyState === WebSocket.OPEN) {
			this.ws.send(JSON.stringify(msg));
		}
	}

	announce(opts: {
		numwant?: number;
		offers?: Array<{ offer_id: string; offer: RTCSessionDescriptionInit }>;
	} = {}): void {
		const msg: TrackerMsg = {
			action: 'announce',
			info_hash: this.infoHash,
			peer_id: this.peerId,
			numwant: opts.numwant ?? 0
		};
		if (opts.offers) msg.offers = opts.offers;
		this.send(msg);
	}

	sendAnswer(toPeerId: string, offerId: string, answer: RTCSessionDescriptionInit): void {
		this.send({
			action: 'announce',
			info_hash: this.infoHash,
			peer_id: this.peerId,
			to_peer_id: toPeerId,
			answer,
			offer_id: offerId
		});
	}

	close(): void {
		this.ws?.close();
		this.ws = null;
	}
}

// ---------------------------------------------------------------------------
// Tunnel message types (JSON, sent over RTCDataChannel)
// ---------------------------------------------------------------------------

interface ReqMsg {
	type: 'req';
	id: string;
	method: string;
	path: string;
	headers: Record<string, string>;
	body: string | null; // base64 or null
}
interface ResStartMsg {
	type: 'res_start';
	id: string;
	status: number;
	headers: Record<string, string>;
}
interface ResChunkMsg {
	type: 'res_chunk';
	id: string;
	data: string; // base64
}
interface ResEndMsg {
	type: 'res_end';
	id: string;
}
interface ResErrMsg {
	type: 'res_err';
	id: string;
	message: string;
}
interface CancelMsg {
	type: 'cancel';
	id: string;
}
interface AuthMsg {
	type: 'auth';
	pass: string;
}
interface AuthOkMsg {
	type: 'auth_ok';
}
interface AuthFailMsg {
	type: 'auth_fail';
}

type TunnelMsg =
	| ReqMsg
	| ResStartMsg
	| ResChunkMsg
	| ResEndMsg
	| ResErrMsg
	| CancelMsg
	| AuthMsg
	| AuthOkMsg
	| AuthFailMsg;

// Max bytes per res_chunk message (keeps data channel messages well below limits)
const CHUNK_BYTES = 8192;

function uint8ToBase64(bytes: Uint8Array): string {
	let binary = '';
	for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
	return btoa(binary);
}

function base64ToUint8(b64: string): Uint8Array {
	return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

// ---------------------------------------------------------------------------
// HostTunnel
// ---------------------------------------------------------------------------

export type HostCallbacks = {
	onPeerCountChange?: (count: number) => void;
};

export class HostTunnel {
	private readonly passCode: string;
	private readonly infoHash: string;
	private readonly peerId: string;
	private readonly callbacks: HostCallbacks;

	private trackers: Tracker[] = [];
	private peers = new Map<string, { pc: RTCPeerConnection; channel: RTCDataChannel }>();
	// AbortControllers for in-flight host-side fetch() calls, keyed by request id.
	private activeRequests = new Map<string, AbortController>();
	private announceTimer: ReturnType<typeof setInterval> | null = null;
	private stopped = false;

	constructor(roomCode: string, passCode: string, callbacks: HostCallbacks = {}) {
		this.passCode = passCode;
		this.infoHash = roomToInfoHash(roomCode);
		this.peerId = randomStr(20);
		this.callbacks = callbacks;
	}

	get peerCount(): number {
		return this.peers.size;
	}

	async start(): Promise<void> {
		await this.connectTrackers();
		this.announceTimer = setInterval(() => {
			for (const t of this.trackers) t.announce({ numwant: 0 });
		}, ANNOUNCE_INTERVAL_MS);
	}

	private async connectTrackers(): Promise<void> {
		for (const url of TRACKER_URLS) {
			try {
				await this.connectOneTracker(url);
			} catch {
				// try next
			}
		}
	}

	private async connectOneTracker(url: string): Promise<void> {
		const tracker = new Tracker(this.infoHash, this.peerId);
		tracker.onOffer = (fromPeerId, offerId, offer) => {
			void this.handleOffer(tracker, fromPeerId, offerId, offer);
		};
		tracker.onClose = () => {
			this.trackers = this.trackers.filter((t) => t !== tracker);
			if (!this.stopped) {
				setTimeout(() => void this.connectOneTracker(url), 5000);
			}
		};
		await tracker.connect(url);
		tracker.announce({ numwant: 0 });
		this.trackers.push(tracker);
	}

	private async handleOffer(
		tracker: Tracker,
		fromPeerId: string,
		offerId: string,
		offer: RTCSessionDescriptionInit
	): Promise<void> {
		try {
			const pc = new RTCPeerConnection(STUN_CONFIG);
			await pc.setRemoteDescription(new RTCSessionDescription(offer));
			const answer = await pc.createAnswer();
			await pc.setLocalDescription(answer);
			await waitForIceComplete(pc);
			tracker.sendAnswer(fromPeerId, offerId, pc.localDescription!);

			pc.ondatachannel = (event) => this.setupChannel(pc, fromPeerId, event.channel);
		} catch {
			// ignore failed handshakes
		}
	}

	private setupChannel(pc: RTCPeerConnection, peerId: string, channel: RTCDataChannel): void {
		let authenticated = false;

		channel.onclose = () => {
			if (this.peers.has(peerId)) {
				this.peers.delete(peerId);
				this.callbacks.onPeerCountChange?.(this.peers.size);
			}
			pc.close();
		};

		channel.onmessage = (event) => {
			try {
				const msg = JSON.parse(event.data as string) as TunnelMsg;

				if (!authenticated) {
					if (msg.type === 'auth') {
						if (msg.pass === this.passCode) {
							authenticated = true;
							channel.send(JSON.stringify({ type: 'auth_ok' } satisfies AuthOkMsg));
							this.peers.set(peerId, { pc, channel });
							this.callbacks.onPeerCountChange?.(this.peers.size);
						} else {
							channel.send(JSON.stringify({ type: 'auth_fail' } satisfies AuthFailMsg));
							channel.close();
						}
					}
					return;
				}

				if (msg.type === 'req') {
					void this.handleRequest(channel, msg);
				} else if (msg.type === 'cancel') {
					this.activeRequests.get(msg.id)?.abort();
				}
			} catch {
				// ignore malformed messages
			}
		};
	}

	private async handleRequest(channel: RTCDataChannel, msg: ReqMsg): Promise<void> {
		const { id, method, path, headers, body } = msg;
		const t0 = performance.now();
		const ac = new AbortController();
		this.activeRequests.set(id, ac);

		try {
			const init: RequestInit = { method, headers, signal: ac.signal };
			if (body !== null) init.body = base64ToUint8(body).buffer as ArrayBuffer;

			const response = await fetch(path, init);

			const resHeaders: Record<string, string> = {};
			response.headers.forEach((v, k) => {
				resHeaders[k] = v;
			});

			console.log(`[rtc] ${method} ${path} -> ${response.status} (${Math.round(performance.now() - t0)}ms)`);

			channel.send(
				JSON.stringify({
					type: 'res_start',
					id,
					status: response.status,
					headers: resHeaders
				} satisfies ResStartMsg)
			);

			const reader = response.body?.getReader();
			if (reader) {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					for (let i = 0; i < value.length; i += CHUNK_BYTES) {
						const slice = value.subarray(i, i + CHUNK_BYTES);
						channel.send(
							JSON.stringify({
								type: 'res_chunk',
								id,
								data: uint8ToBase64(slice)
							} satisfies ResChunkMsg)
						);
					}
				}
			}

			channel.send(JSON.stringify({ type: 'res_end', id } satisfies ResEndMsg));
		} catch (e) {
			// AbortError means the client cancelled — no need to send an error back.
			if (!(e instanceof DOMException && e.name === 'AbortError')) {
				const errMsg = e instanceof Error ? e.message : String(e);
				console.error(`[rtc] ${method} ${path} -> error: ${errMsg}`);
				channel.send(
					JSON.stringify({ type: 'res_err', id, message: errMsg } satisfies ResErrMsg)
				);
			}
		} finally {
			this.activeRequests.delete(id);
		}
	}

	stop(): void {
		this.stopped = true;
		if (this.announceTimer) clearInterval(this.announceTimer);
		for (const t of this.trackers) t.close();
		for (const { pc, channel } of this.peers.values()) {
			channel.close();
			pc.close();
		}
		for (const ac of this.activeRequests.values()) ac.abort();
		this.trackers = [];
		this.peers.clear();
		this.activeRequests.clear();
	}
}

// ---------------------------------------------------------------------------
// ClientTunnel
// ---------------------------------------------------------------------------

export type ClientCallbacks = {
	onConnected?: () => void;
	onDisconnected?: () => void;
};

type PendingReq = {
	onStart: (status: number, headers: Record<string, string>) => void;
	onChunk: (data: string) => void;
	onEnd: () => void;
	onError: (message: string) => void;
};

export class ClientTunnel {
	private readonly passCode: string;
	private readonly infoHash: string;
	private readonly peerId: string;
	private readonly callbacks: ClientCallbacks;

	private pc: RTCPeerConnection | null = null;
	private channel: RTCDataChannel | null = null;
	private tracker: Tracker | null = null;
	private pending = new Map<string, PendingReq>();

	constructor(roomCode: string, passCode: string, callbacks: ClientCallbacks = {}) {
		this.passCode = passCode;
		this.infoHash = roomToInfoHash(roomCode);
		this.peerId = randomStr(20);
		this.callbacks = callbacks;
	}

	get isConnected(): boolean {
		return this.channel?.readyState === 'open';
	}

	async connect(): Promise<void> {
		let lastError: Error = new Error('no trackers available');
		for (const url of TRACKER_URLS) {
			try {
				await this.connectViaTracker(url);
				return;
			} catch (e) {
				lastError = e instanceof Error ? e : new Error(String(e));
				this.cleanupConnection();
			}
		}
		throw lastError;
	}

	private async connectViaTracker(trackerUrl: string): Promise<void> {
		const offerId = randomStr(20);
		const pc = new RTCPeerConnection(STUN_CONFIG);
		this.pc = pc;

		const channel = pc.createDataChannel('tunnel', { ordered: true });
		this.channel = channel;

		const offer = await pc.createOffer();
		await pc.setLocalDescription(offer);
		await waitForIceComplete(pc);

		const tracker = new Tracker(this.infoHash, this.peerId);
		this.tracker = tracker;
		await tracker.connect(trackerUrl);

		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				reject(new Error('connection timed out waiting for host'));
			}, CONNECT_TIMEOUT_MS);

			tracker.onAnswer = async (_offerId, answer) => {
				if (_offerId !== offerId) return;
				try {
					await pc.setRemoteDescription(new RTCSessionDescription(answer));
				} catch (e) {
					clearTimeout(timer);
					reject(e);
				}
			};

			channel.onopen = () => {
				channel.send(JSON.stringify({ type: 'auth', pass: this.passCode } satisfies AuthMsg));
			};

			channel.onmessage = (event) => {
				try {
					const msg = JSON.parse(event.data as string) as TunnelMsg;
					if (msg.type === 'auth_ok') {
						clearTimeout(timer);
						this.callbacks.onConnected?.();
						resolve();
					} else if (msg.type === 'auth_fail') {
						clearTimeout(timer);
						reject(new Error('authentication failed: invalid passcode'));
					} else {
						this.routeResponseMsg(msg);
					}
				} catch {
					// ignore
				}
			};

			channel.onclose = () => {
				this.callbacks.onDisconnected?.();
				this.rejectAllPending('connection closed');
			};

			channel.onerror = () => {
				clearTimeout(timer);
				reject(new Error('data channel error'));
			};

			tracker.announce({
				numwant: 1,
				offers: [{ offer_id: offerId, offer: pc.localDescription! }]
			});
		});
	}

	private routeResponseMsg(msg: TunnelMsg): void {
		if (
			msg.type !== 'res_start' &&
			msg.type !== 'res_chunk' &&
			msg.type !== 'res_end' &&
			msg.type !== 'res_err'
		)
			return;
		const req = this.pending.get(msg.id);
		if (!req) return;

		if (msg.type === 'res_start') {
			req.onStart(msg.status, msg.headers);
		} else if (msg.type === 'res_chunk') {
			req.onChunk(msg.data);
		} else if (msg.type === 'res_end') {
			req.onEnd();
			this.pending.delete(msg.id);
		} else if (msg.type === 'res_err') {
			req.onError(msg.message);
			this.pending.delete(msg.id);
		}
	}

	private rejectAllPending(reason: string): void {
		for (const req of this.pending.values()) req.onError(reason);
		this.pending.clear();
	}

	private cleanupConnection(): void {
		this.channel?.close();
		this.pc?.close();
		this.tracker?.close();
		this.channel = null;
		this.pc = null;
		this.tracker = null;
	}

	async fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		if (!this.channel || this.channel.readyState !== 'open') {
			throw new Error('tunnel not connected');
		}

		const request = new Request(input, init);
		const signal = init?.signal ?? (input instanceof Request ? input.signal : undefined);
		const id = randomStr(16);

		if (signal?.aborted) {
			return Promise.reject(new DOMException('Aborted', 'AbortError'));
		}

		// Extract path+query so the host fetches relative to its own origin
		const reqUrl = new URL(request.url);
		const path = reqUrl.pathname + reqUrl.search;

		const headers: Record<string, string> = {};
		request.headers.forEach((v, k) => {
			headers[k] = v;
		});

		let bodyB64: string | null = null;
		const bodyBytes = await request.arrayBuffer();
		if (bodyBytes.byteLength > 0) {
			bodyB64 = uint8ToBase64(new Uint8Array(bodyBytes));
		}

		return new Promise((resolve, reject) => {
			let streamController!: ReadableStreamDefaultController<Uint8Array>;
			const stream = new ReadableStream<Uint8Array>({
				start(ctrl) {
					streamController = ctrl;
				}
			});

			const abortHandler = () => {
				this.pending.delete(id);
				try {
					streamController.error(new DOMException('Aborted', 'AbortError'));
				} catch {
					// stream may already be closed
				}
				reject(new DOMException('Aborted', 'AbortError'));
				// Tell the host to stop the in-flight fetch
				if (this.channel?.readyState === 'open') {
					this.channel.send(JSON.stringify({ type: 'cancel', id } satisfies CancelMsg));
				}
			};

			signal?.addEventListener('abort', abortHandler, { once: true });

			this.pending.set(id, {
				onStart: (status, resHeaders) => {
					resolve(new Response(stream, { status, headers: resHeaders }));
				},
				onChunk: (data) => {
					streamController.enqueue(base64ToUint8(data));
				},
				onEnd: () => {
					signal?.removeEventListener('abort', abortHandler);
					streamController.close();
				},
				onError: (message) => {
					signal?.removeEventListener('abort', abortHandler);
					try {
						streamController.error(new Error(message));
					} catch {
						// stream may already be closed
					}
					reject(new Error(message));
					this.pending.delete(id);
				}
			});

			this.channel!.send(
				JSON.stringify({
					type: 'req',
					id,
					method: request.method,
					path,
					headers,
					body: bodyB64
				} satisfies ReqMsg)
			);
		});
	}

	disconnect(): void {
		this.rejectAllPending('disconnected');
		this.cleanupConnection();
	}
}
