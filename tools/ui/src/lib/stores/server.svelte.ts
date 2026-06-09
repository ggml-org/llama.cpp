import { PropsService } from '$lib/services/props.service';
import { ServerRole } from '$lib/enums';

/**
 * serverStore - Server connection state, configuration, and role detection
 *
 * This store manages the server connection state and properties fetched from `/props`.
 * It provides reactive state for server configuration and role detection.
 *
 * **Architecture & Relationships:**
 * - **PropsService**: Stateless service for fetching `/props` data
 * - **serverStore** (this class): Reactive store for server state
 * - **modelsStore**: Independent store for model management (uses PropsService directly)
 *
 * **Key Features:**
 * - **Server State**: Connection status, loading, error handling
 * - **Role Detection**: MODEL (single model) vs ROUTER (multi-model)
 * - **Default Params**: Server-wide generation defaults
 */
class ServerStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	props = $state<ApiLlamaCppServerProps | null>(null);
	loading = $state(false);
	error = $state<string | null>(null);
	role = $state<ServerRole | null>(null);
	private fetchPromise: Promise<void> | null = null;

	/**
	 *
	 *
	 * Getters
	 *
	 *
	 */

	get defaultParams(): ApiLlamaCppServerProps['default_generation_settings']['params'] | null {
		return this.props?.default_generation_settings?.params || null;
	}

	get contextSize(): number | null {
		const nCtx = this.props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	get uiSettings(): Record<string, string | number | boolean> | undefined {
		return this.props?.ui_settings ?? this.props?.webui_settings;
	}

	get isRouterMode(): boolean {
		return this.role === ServerRole.ROUTER;
	}

	get isModelMode(): boolean {
		return this.role === ServerRole.MODEL;
	}

	/**
	 *
	 *
	 * Data Handling
	 *
	 *
	 */

	async fetch(): Promise<void> {
		if (this.fetchPromise) return this.fetchPromise;

		this.loading = true;
		this.error = null;

		const fetchPromise = (async () => {
			try {
				const props = await PropsService.fetch();
				this.props = props;
				this.error = null;
				this.detectRole(props);
			} catch (error: unknown) {
				this.error = error instanceof Error ? error.message : String(error);
				console.error('Error fetching server properties:', error);
			} finally {
				this.loading = false;
				this.fetchPromise = null;
			}
		})();

		this.fetchPromise = fetchPromise;
		await fetchPromise;
	}

	clear(): void {
		this.props = null;
		this.error = null;
		this.loading = false;
		this.role = null;
		this.fetchPromise = null;
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	private detectRole(props: ApiLlamaCppServerProps): void {
		const newRole = props?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;
		if (this.role !== newRole) {
			this.role = newRole;
			console.info(`Server running in ${newRole === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
		}
	}
}

export const serverStore = new ServerStore();

export const serverProps = () => serverStore.props;
export const serverLoading = () => serverStore.loading;
export const serverError = () => serverStore.error;
export const serverRole = () => serverStore.role;
export const defaultParams = () => serverStore.defaultParams;
export const contextSize = () => serverStore.contextSize;
export const isRouterMode = () => serverStore.isRouterMode;
export const isModelMode = () => serverStore.isModelMode;
