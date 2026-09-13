/**
 * serverStore - Server connection state, configuration and role detection
 *
 * Owns the connection state and properties fetched from /props, plus MODEL
 * vs ROUTER role detection and server-wide generation defaults. Uses
 * PropsService for the /props fetch.
 */

import { ServerRole } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
import { ApiError } from '$lib/utils';

const LOADING_RETRY_INTERVAL_MS = 1000;

class ServerStore {
	error = $state<string | null>(null);
	loading = $state(false);
	props = $state<ApiLlamaCppServerProps | null>(null);
	role = $state<ServerRole | null>(null);
	status = $state<number | null>(null);
	private fetchPromise: Promise<void> | null = null;
	private retryTimer: ReturnType<typeof setTimeout> | null = null;

	get contextSize(): number | null {
		const nCtx = this.props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	get defaultParams(): ApiLlamaCppServerProps['default_generation_settings']['params'] | null {
		return this.props?.default_generation_settings?.params || null;
	}

	get isModelMode(): boolean {
		return this.role === ServerRole.MODEL;
	}

	get isRouterMode(): boolean {
		return this.role === ServerRole.ROUTER;
	}

	get uiSettings(): Record<string, string | number | boolean> | undefined {
		return this.props?.ui_settings ?? this.props?.webui_settings;
	}

	clear(): void {
		this.clearRetryTimer();
		this.props = null;
		this.error = null;
		this.status = null;
		this.loading = false;
		this.role = null;
		this.fetchPromise = null;
	}

	/**
	 * @param background - Set by the automatic "still loading" poll. Skips the
	 * `loading` flag flip so the UI doesn't bounce between the full loading
	 * splash and the chat screen every retry tick.
	 */
	async fetch({ background = false }: { background?: boolean } = {}): Promise<void> {
		if (this.fetchPromise) return this.fetchPromise;

		this.clearRetryTimer();

		if (!background) {
			this.loading = true;
		}

		// Don't clear an existing "still loading" error before a retry -
		// doing so would unmount/remount the error banner every second.
		if (this.status !== 503) {
			this.error = null;
		}

		const fetchPromise = (async () => {
			try {
				const props = await PropsService.fetch();

				this.props = props;
				this.error = null;
				this.status = null;
				this.detectRole(props);
			} catch (error: unknown) {
				this.error = error instanceof Error ? error.message : String(error);
				this.status = error instanceof ApiError ? error.status : null;
				console.error('Error fetching server properties:', error);

				if (this.status === 503) {
					this.scheduleRetry();
				}
			} finally {
				if (!background) {
					this.loading = false;
				}

				this.fetchPromise = null;
			}
		})();

		this.fetchPromise = fetchPromise;
		await fetchPromise;
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
