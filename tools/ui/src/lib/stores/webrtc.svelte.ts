import { browser } from '$app/environment';
import { CODE_LENGTHS } from '$lib/constants';
import { IS_WEB_ONLY } from '$lib/constants/web-only.constants';
import { RemoteAccessMode, ServerType, TunnelStatus } from '$lib/enums';
import type { StoredServer } from '$lib/types';
import { ClientTunnel } from '$lib/utils/webrtc-tunnel';

// Registry of the servers the app can talk to, reconnected to on reload.
const SERVERS_KEY = 'llama_servers';

class WebRTCStore {
	errorMessage = $state('');
	mode = $state<RemoteAccessMode>(RemoteAccessMode.OFF);
	status = $state<TunnelStatus>(TunnelStatus.IDLE);

	private clientTunnel: ClientTunnel | null = null;
	// Requests that arrive while mode='client' but tunnel not yet open are held here.
	private connectionWaiters: Array<{ resolve: () => void; reject: (e: Error) => void }> = [];
	// The original window.fetch saved before the interceptor is installed.
	private originalFetch: typeof window.fetch | null = null;

	/**
	 * A web-only build cannot render the app before the tunnel carries its
	 * requests, so the splash stays up while a restored session connects and
	 * when it fails, where it offers another code.
	 */
	get isBlocked(): boolean {
		return IS_WEB_ONLY && this.status !== TunnelStatus.CONNECTED;
	}

	get isConnected(): boolean {
		return this.status === TunnelStatus.CONNECTED;
	}

	/**
	 * A web-only build has no server of its own, so it stays unusable until a
	 * code is stored. Restoring a saved session already sets mode='client', even
	 * when the tunnel later fails, so a failed reconnect does not ask again.
	 */
	get needsCode(): boolean {
		return IS_WEB_ONLY && this.mode === RemoteAccessMode.OFF;
	}

	constructor() {
		if (browser) {
			this.restoreServers();
		}
	}

	async joinAsClient(shareCode: string): Promise<void> {
		if (shareCode.length < CODE_LENGTHS.SHARE) {
			throw new Error(`Invalid code: must be ${CODE_LENGTHS.SHARE} characters`);
		}

		const roomCode = shareCode.slice(0, CODE_LENGTHS.ROOM);
		const passCode = shareCode.slice(CODE_LENGTHS.ROOM);

		await this.activateClient(roomCode, passCode);
	}

	leaveAsClient(): void {
		this.uninstallInterceptor();
		this.clientTunnel?.disconnect();
		this.clientTunnel = null;
		this.mode = RemoteAccessMode.OFF;
		this.status = TunnelStatus.IDLE;
		this.clearServers();
	}

	/** Retry the saved server after a failed or dropped connection. */
	async reconnect(): Promise<void> {
		const server = this.readTunnelServer();

		if (!server) return;

		this.clientTunnel?.disconnect();
		this.clientTunnel = null;

		await this.activateClient(
			server.code.slice(0, CODE_LENGTHS.ROOM),
			server.code.slice(CODE_LENGTHS.ROOM),
			true
		);
	}

	tunnelFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
		// If the tunnel is open, forward immediately.
		if (this.clientTunnel?.isConnected) {
			return this.clientTunnel.fetch(input, init);
		}

		// If we are still connecting, queue the request until the tunnel opens.
		if (this.mode === RemoteAccessMode.CLIENT && this.status === TunnelStatus.CONNECTING) {
			return new Promise<void>((resolve, reject) => {
				this.connectionWaiters.push({ reject, resolve });
			}).then(() => this.clientTunnel!.fetch(input, init));
		}

		throw new Error(
			'Remote access is enabled but the tunnel is not connected. ' +
				'Requests are not sent to the server hosting this page. ' +
				'Reconnect or leave remote access in settings.'
		);
	}

	// -------------------------------------------------------------------------
	// Fetch interceptor (installed synchronously when client mode activates)
	// -------------------------------------------------------------------------

	private async activateClient(
		roomCode: string,
		passCode: string,
		keepOnError = false
	): Promise<void> {
		this.mode = RemoteAccessMode.CLIENT;
		this.status = TunnelStatus.CONNECTING;
		this.errorMessage = '';
		// Install the fetch interceptor synchronously (before any await) so that
		// requests fired by layout effects on the same tick are already captured.
		this.installInterceptor();

		const tunnel = new ClientTunnel(roomCode, passCode, {
			onConnected: () => {
				this.status = TunnelStatus.CONNECTED;
			},
			onDisconnected: () => {
				this.status = TunnelStatus.ERROR;
				this.errorMessage = 'Disconnected from host';
			}
		});

		try {
			await tunnel.connect();
			this.clientTunnel = tunnel;
			this.writeServers([{ code: roomCode + passCode, type: ServerType.TUNNEL }]);
			// Release any requests that were queued while connecting.
			const waiters = this.connectionWaiters.splice(0);

			for (const w of waiters) w.resolve();
		} catch (e) {
			this.clientTunnel = null;
			this.status = TunnelStatus.ERROR;
			this.errorMessage = e instanceof Error ? e.message : String(e);
			// Reject queued requests.
			const waiters = this.connectionWaiters.splice(0);
			const err = e instanceof Error ? e : new Error(String(e));

			for (const w of waiters) w.reject(err);

			// A restored session stays in client mode with the interceptor in
			// place, so requests fail loudly instead of silently reaching the
			// server that happens to serve this page.
			if (!keepOnError) {
				this.mode = RemoteAccessMode.OFF;
				this.uninstallInterceptor();
			}

			throw e;
		}
	}

	private clearServers(): void {
		localStorage.removeItem(SERVERS_KEY);
	}

	// -------------------------------------------------------------------------
	// Fetch proxy
	// -------------------------------------------------------------------------

	private installInterceptor(): void {
		if (this.originalFetch) return; // already installed

		this.originalFetch = window.fetch.bind(window);
		window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
			try {
				const url =
					input instanceof Request ? input.url : input instanceof URL ? input.href : String(input);
				const parsed = new URL(url, window.location.href);

				if (parsed.origin === window.location.origin) {
					return this.tunnelFetch(input, init);
				}
			} catch {
				// not a parseable URL — fall through
			}

			return this.originalFetch!(input, init);
		};
	}

	// -------------------------------------------------------------------------
	// Persistence helpers
	// -------------------------------------------------------------------------

	/** First tunnel entry of the registry, the only one reachable for now. */
	private readTunnelServer(): StoredServer | null {
		try {
			const raw = localStorage.getItem(SERVERS_KEY);

			if (!raw) return null;

			const servers = JSON.parse(raw) as StoredServer[];

			return servers.find((server) => server.type === ServerType.TUNNEL) ?? null;
		} catch {
			// ignore corrupt storage
			return null;
		}
	}

	private restoreServers(): void {
		const server = this.readTunnelServer();

		if (!server) return;

		void this.activateClient(
			server.code.slice(0, CODE_LENGTHS.ROOM),
			server.code.slice(CODE_LENGTHS.ROOM),
			true
		).catch(() => {
			// state is already reflected in status and errorMessage
		});
	}

	private uninstallInterceptor(): void {
		if (!this.originalFetch) return;

		window.fetch = this.originalFetch;
		this.originalFetch = null;
	}

	private writeServers(servers: StoredServer[]): void {
		localStorage.setItem(SERVERS_KEY, JSON.stringify(servers));
	}
}

export const webrtcStore = new WebRTCStore();
