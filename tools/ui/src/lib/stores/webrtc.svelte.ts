import { browser } from '$app/environment';
import { ClientTunnel, HostTunnel, generatePassCode, generateRoomCode } from '$lib/utils/webrtc-tunnel';

// Stores the generated host codes; persists until explicitly regenerated.
const HOST_CODES_KEY = 'llama_webrtc_host_codes';
// Stores the active session (mode + codes) for auto-reconnect on reload.
const SESSION_KEY = 'llama_webrtc_session';

type HostCodes = { roomCode: string; passCode: string };
type SessionData = { mode: 'host' | 'client'; roomCode: string; passCode: string };
type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'error';

class WebRTCStore {
	mode = $state<'off' | 'host' | 'client'>('off');
	status = $state<ConnectionStatus>('idle');
	peerCount = $state(0);
	errorMessage = $state('');

	// Reflect the saved host codes; populated on init even when mode is 'off'.
	private _roomCode = $state('');
	private _passCode = $state('');

	private hostTunnel: HostTunnel | null = null;
	private clientTunnel: ClientTunnel | null = null;
	// Requests that arrive while mode='client' but tunnel not yet open are held here.
	private connectionWaiters: Array<{ resolve: () => void; reject: (e: Error) => void }> = [];
	// The original window.fetch saved before the interceptor is installed.
	private originalFetch: typeof window.fetch | null = null;

	constructor() {
		if (browser) {
			// Load persisted host codes so the UI can show them before host is enabled.
			const saved = this.readHostCodes();
			if (saved) {
				this._roomCode = saved.roomCode;
				this._passCode = saved.passCode;
			}
			this.restoreSession();
		}
	}

	get roomCode(): string {
		return this._roomCode;
	}

	get passCode(): string {
		return this._passCode;
	}

	// Full 40-char code shared with remote clients.
	get shareCode(): string {
		return this._roomCode + this._passCode;
	}

	get isConnected(): boolean {
		return this.status === 'connected';
	}

	get hasHostCodes(): boolean {
		return this._roomCode !== '' && this._passCode !== '';
	}

	// -------------------------------------------------------------------------
	// Host
	// -------------------------------------------------------------------------

	async startHost(): Promise<void> {
		if (this.mode !== 'off') return;

		// Reuse the persisted codes; generate once if none exist yet.
		let roomCode = this._roomCode;
		let passCode = this._passCode;

		if (!roomCode || !passCode) {
			roomCode = generateRoomCode();
			passCode = generatePassCode();
			this._roomCode = roomCode;
			this._passCode = passCode;
			this.writeHostCodes({ roomCode, passCode });
		}

		await this.activateHost(roomCode, passCode);
	}

	/** Generate a fresh room + pass code. Restarts the tunnel if currently active. */
	async regenerateCodes(): Promise<void> {
		const roomCode = generateRoomCode();
		const passCode = generatePassCode();
		this._roomCode = roomCode;
		this._passCode = passCode;
		this.writeHostCodes({ roomCode, passCode });

		if (this.mode === 'host') {
			this.hostTunnel?.stop();
			this.hostTunnel = null;
			await this.activateHost(roomCode, passCode);
		}
	}

	private async activateHost(roomCode: string, passCode: string): Promise<void> {
		this.mode = 'host';
		this.status = 'connecting';
		this.errorMessage = '';
		this.peerCount = 0;

		const tunnel = new HostTunnel(roomCode, passCode, {
			onPeerCountChange: (count) => {
				this.peerCount = count;
			}
		});

		try {
			await tunnel.start();
			this.hostTunnel = tunnel;
			this.status = 'connected';
			this.writeSession({ mode: 'host', roomCode, passCode });
		} catch (e) {
			this.hostTunnel = null;
			this.status = 'error';
			this.errorMessage = e instanceof Error ? e.message : String(e);
		}
	}

	stopHost(): void {
		this.hostTunnel?.stop();
		this.hostTunnel = null;
		this.mode = 'off';
		this.status = 'idle';
		this.peerCount = 0;
		// Codes are intentionally kept: _roomCode/_passCode and HOST_CODES_KEY
		// remain so the user can re-enable without a new code.
		this.clearSession();
	}

	// -------------------------------------------------------------------------
	// Client
	// -------------------------------------------------------------------------

	async joinAsClient(shareCode: string): Promise<void> {
		if (shareCode.length < 40) throw new Error('Invalid code: must be 40 characters');

		const roomCode = shareCode.slice(0, 8);
		const passCode = shareCode.slice(8);
		await this.activateClient(roomCode, passCode);
	}

	private async activateClient(roomCode: string, passCode: string): Promise<void> {
		this.mode = 'client';
		this.status = 'connecting';
		this.errorMessage = '';
		// Install the fetch interceptor synchronously (before any await) so that
		// requests fired by layout effects on the same tick are already captured.
		this.installInterceptor();

		const tunnel = new ClientTunnel(roomCode, passCode, {
			onConnected: () => {
				this.status = 'connected';
			},
			onDisconnected: () => {
				this.status = 'error';
				this.errorMessage = 'Disconnected from host';
			}
		});

		try {
			await tunnel.connect();
			this.clientTunnel = tunnel;
			this.writeSession({ mode: 'client', roomCode, passCode });
			// Release any requests that were queued while connecting.
			const waiters = this.connectionWaiters.splice(0);
			for (const w of waiters) w.resolve();
		} catch (e) {
			this.clientTunnel = null;
			this.mode = 'off';
			this.status = 'error';
			this.errorMessage = e instanceof Error ? e.message : String(e);
			this.uninstallInterceptor();
			// Reject queued requests.
			const waiters = this.connectionWaiters.splice(0);
			const err = e instanceof Error ? e : new Error(String(e));
			for (const w of waiters) w.reject(err);
			throw e;
		}
	}

	leaveAsClient(): void {
		this.uninstallInterceptor();
		this.clientTunnel?.disconnect();
		this.clientTunnel = null;
		this.mode = 'off';
		this.status = 'idle';
		this.clearSession();
	}

	// -------------------------------------------------------------------------
	// Fetch interceptor (installed synchronously when client mode activates)
	// -------------------------------------------------------------------------

	private installInterceptor(): void {
		if (this.originalFetch) return; // already installed
		this.originalFetch = window.fetch.bind(window);
		const store = this;
		window.fetch = function (input: RequestInfo | URL, init?: RequestInit) {
			try {
				const url =
					input instanceof Request
						? input.url
						: input instanceof URL
							? input.href
							: String(input);
				const parsed = new URL(url, window.location.href);
				if (parsed.origin === window.location.origin) {
					return store.tunnelFetch(input, init);
				}
			} catch {
				// not a parseable URL — fall through
			}
			return store.originalFetch!(input, init);
		};
	}

	private uninstallInterceptor(): void {
		if (!this.originalFetch) return;
		window.fetch = this.originalFetch;
		this.originalFetch = null;
	}

	// -------------------------------------------------------------------------
	// Fetch proxy
	// -------------------------------------------------------------------------

	tunnelFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		// If the tunnel is open, forward immediately.
		if (this.clientTunnel?.isConnected) {
			return this.clientTunnel.fetch(input, init);
		}
		// If we are still connecting, queue the request until the tunnel opens.
		if (this.mode === 'client' && this.status === 'connecting') {
			return new Promise<void>((resolve, reject) => {
				this.connectionWaiters.push({ resolve, reject });
			}).then(() => this.clientTunnel!.fetch(input, init));
		}
		throw new Error('tunnel not connected');
	}

	// -------------------------------------------------------------------------
	// Persistence helpers
	// -------------------------------------------------------------------------

	private readHostCodes(): HostCodes | null {
		try {
			const raw = localStorage.getItem(HOST_CODES_KEY);
			return raw ? (JSON.parse(raw) as HostCodes) : null;
		} catch {
			return null;
		}
	}

	private writeHostCodes(codes: HostCodes): void {
		localStorage.setItem(HOST_CODES_KEY, JSON.stringify(codes));
	}

	private restoreSession(): void {
		try {
			const raw = localStorage.getItem(SESSION_KEY);
			if (!raw) return;
			const session = JSON.parse(raw) as SessionData;
			if (session.mode === 'host') {
				void this.activateHost(session.roomCode, session.passCode);
			} else if (session.mode === 'client') {
				void this.activateClient(session.roomCode, session.passCode);
			}
		} catch {
			// ignore corrupt storage
		}
	}

	private writeSession(data: SessionData): void {
		localStorage.setItem(SESSION_KEY, JSON.stringify(data));
	}

	private clearSession(): void {
		localStorage.removeItem(SESSION_KEY);
	}
}

export const webrtcStore = new WebRTCStore();
