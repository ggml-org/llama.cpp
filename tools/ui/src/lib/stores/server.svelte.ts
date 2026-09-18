/**
 * serverStore - Server connection state, configuration and role detection
 *
 * Owns the connection state and properties fetched from /props, plus MODEL
 * vs ROUTER role detection and server-wide generation defaults. Uses
 * PropsService for the /props fetch.
 */

import { BACKEND_CAPABILITIES, LOCAL_BACKEND_ID } from '$lib/constants';
import { ServerRole } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
import type { BackendCapabilities } from '$lib/types';
import { ApiError } from '$lib/utils';
import { getBackend } from '$lib/utils/api-base';
import { getBackendCapabilities } from '$lib/utils/backend';

const LOADING_RETRY_INTERVAL_MS = 1000;

class ServerStore {
	error = $state<string | null>(null);
	loading = $state(false);
	props = $state<ApiLlamaCppServerProps | null>(null);
	role = $state<ServerRole | null>(null);
	status = $state<number | null>(null);
	private fetchBackendId: string | undefined;
	private fetchPromise: Promise<void> | null = null;
	/**
	 * Whether a local llama.cpp server answered. Null until the first probe
	 * resolves: the bundled build is served by one, the hosted PWA is not.
	 */
	private localAvailable: boolean | null = null;
	/** Local server state kept alive while an external backend is active. */
	private localState: { props: ApiLlamaCppServerProps | null; role: ServerRole | null } | null =
		null;
	private retryTimer: ReturnType<typeof setTimeout> | null = null;

	/** Features of the active backend. Defaults to full llama.cpp support. */
	get capabilities(): BackendCapabilities {
		const backend = getBackend();

		return backend ? getBackendCapabilities(backend) : BACKEND_CAPABILITIES['llama.cpp'];
	}

	get contextSize(): number | null {
		const nCtx = this.props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	get defaultParams(): ApiLlamaCppServerProps['default_generation_settings']['params'] | null {
		return this.localProps?.default_generation_settings?.params || null;
	}

	/** A local llama.cpp server answered, either now or earlier this session. */
	get hasLocalServer(): boolean {
		return this.localAvailable === true;
	}

	get isModelMode(): boolean {
		return this.role === ServerRole.MODEL;
	}

	get isRouterMode(): boolean {
		return this.role === ServerRole.ROUTER;
	}

	/**
	 * Local server state. The local server is the base of the installation, so
	 * its props and role come from the mirror while another backend is active
	 * instead of following the selected model.
	 */
	private get localFacts(): { props: ApiLlamaCppServerProps | null; role: ServerRole | null } {
		if (getBackend()?.id === LOCAL_BACKEND_ID) return { props: this.props, role: this.role };

		return this.localState ?? { props: null, role: null };
	}

	/** The local server runs as a router, independent of the active backend. */
	get localIsRouter(): boolean {
		return this.localFacts.role === ServerRole.ROUTER;
	}

	/** Props of the local server, from the active backend or the cache. */
	get localProps(): ApiLlamaCppServerProps | null {
		return this.localFacts.props;
	}

	/** The local server is known to be unreachable: local-only UI stays out. */
	get localServerMissing(): boolean {
		return this.localAvailable === false;
	}

	get uiSettings(): Record<string, string | number | boolean> | undefined {
		return this.props?.ui_settings ?? this.props?.webui_settings;
	}

	/**
	 * Keep the local server state before switching to an external backend, so
	 * switching back restores it instead of asking the server again.
	 */
	cacheLocalState(): void {
		// props only exist while a llama.cpp server is active; an external to
		// external switch must not overwrite the kept local state with blanks
		if (!this.props) return;

		this.localState = { props: this.props, role: this.role };
	}

	clear(): void {
		this.clearRetryTimer();
		this.props = null;
		this.error = null;
		this.status = null;
		this.loading = false;
		this.role = null;
		this.fetchPromise = null;
		this.fetchBackendId = undefined;
	}

	/**
	 * @param background - Set by the automatic "still loading" poll. Skips the
	 * `loading` flag flip so the UI doesn't bounce between the full loading
	 * splash and the chat screen every retry tick.
	 */
	async fetch({ background = false }: { background?: boolean } = {}): Promise<void> {
		// props and role describe one server. a fetch started for another backend
		// must not be reused, and its response must not commit once the active
		// backend has changed while it was in flight
		const backendId = getBackend()?.id;

		if (this.fetchPromise && this.fetchBackendId === backendId) return this.fetchPromise;

		this.clearRetryTimer();

		// External backends expose no /props endpoint. Keep MODEL-mode defaults so
		// role detection and generation defaults degrade instead of failing.
		if (!this.capabilities.props) {
			this.clear();
			this.role = ServerRole.MODEL;

			return;
		}

		if (!background) {
			this.loading = true;
		}

		// Don't clear an existing "still loading" error before a retry -
		// doing so would unmount/remount the error banner every second.
		if (this.status !== 503) {
			this.error = null;
		}

		const promise = (async () => {
			try {
				const props = await PropsService.fetch();

				// the active backend changed while this request was in flight
				if (getBackend()?.id !== backendId) return;

				this.props = props;
				this.error = null;
				this.status = null;
				this.detectRole(props);

				if (backendId === LOCAL_BACKEND_ID) {
					this.localAvailable = true;
					// mirror the local facts as they arrive: a later clear() must not
					// be able to wipe what the local server told us
					this.localState = { props, role: this.role };
				}
			} catch (error: unknown) {
				if (getBackend()?.id !== backendId) return;

				this.error = error instanceof Error ? error.message : String(error);
				this.status = error instanceof ApiError ? error.status : null;
				console.error('Error fetching server properties:', error);

				if (backendId === LOCAL_BACKEND_ID) this.localAvailable = false;

				if (this.status === 503) {
					this.scheduleRetry();
				}
			} finally {
				if (!background) {
					this.loading = false;
				}
			}
		})();

		this.fetchPromise = promise;
		this.fetchBackendId = backendId;

		// a backend switch clears the in-flight handle; only the fetch that is
		// still the current one may release it
		void promise
			.catch(() => {})
			.finally(() => {
				if (this.fetchPromise === promise) {
					this.fetchPromise = null;
					this.fetchBackendId = undefined;
				}
			});

		await promise;
	}

	/**
	 * Load the local server state in the background at startup. The local tab
	 * then opens from memory instead of asking for props on the first click.
	 */
	async prefetchLocalState(): Promise<void> {
		if (this.localState) return;

		try {
			const props = await PropsService.fetch(false, LOCAL_BACKEND_ID);

			this.localState = {
				props,
				role: props?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL
			};
			this.localAvailable = true;
		} catch {
			// no local server in this deployment: mark it missing so the local
			// backend drops out of the enabled list
			this.localAvailable = false;
		}
	}

	/** Restore the state kept by {@link cacheLocalState}; no request is made. */
	restoreLocalState(): void {
		if (!this.localState) return;

		this.props = this.localState.props;
		this.role = this.localState.role;
		this.error = null;
		this.status = null;
	}

	private clearRetryTimer(): void {
		if (this.retryTimer) {
			clearTimeout(this.retryTimer);
			this.retryTimer = null;
		}
	}

	private detectRole(props: ApiLlamaCppServerProps): void {
		const newRole = props?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;

		if (this.role !== newRole) {
			this.role = newRole;
			console.info(`Server running in ${newRole === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
		}
	}

	private scheduleRetry(): void {
		if (this.retryTimer) return;

		this.retryTimer = setTimeout(() => {
			this.retryTimer = null;
			this.fetch({ background: true });
		}, LOADING_RETRY_INTERVAL_MS);
	}
}

export const serverStore = new ServerStore();
