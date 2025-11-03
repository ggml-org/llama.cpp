import { browser } from '$app/environment';
import { SERVER_PROPS_LOCALSTORAGE_KEY } from '$lib/constants/localstorage-keys';
import { ChatService } from '$lib/services/chat';
import { config } from '$lib/stores/settings.svelte';

/**
 * ServerStore - Server state management and capability detection
 *
 * This store manages communication with the llama.cpp server to retrieve and maintain
 * server properties, model information, and capability detection. It provides reactive
 * state for server connectivity, model capabilities, and endpoint availability.
 *
 * **Architecture & Relationships:**
 * - **ServerStore** (this class): Server state and capability management
 *   - Fetches and caches server properties from `/props` endpoint
 *   - Detects model capabilities (vision, audio support)
 *   - Tests endpoint availability (slots endpoint)
 *   - Provides reactive server state for UI components
 *
 * - **ChatService**: Uses server properties for request validation
 * - **SlotsService**: Depends on slots endpoint availability detection
 * - **UI Components**: Subscribe to server state for capability-based rendering
 *
 * **Key Features:**
 * - **Server Properties**: Model path, context size, build information
 * - **Capability Detection**: Vision and audio modality support
 * - **Endpoint Testing**: Slots endpoint availability checking
 * - **Error Handling**: User-friendly error messages for connection issues
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 * - **State Management**: Loading states and error recovery
 *
 * **Server Capabilities Detected:**
 * - Model name extraction from file path
 * - Vision support (multimodal image processing)
 * - Audio support (speech processing)
 * - Slots endpoint availability (for processing state monitoring)
 * - Context window size and token limits
 */

class ServerStore {
	constructor() {
		if (!browser) return;

		const cachedProps = this.readCachedServerProps();
		if (cachedProps) {
			this._serverProps = cachedProps;
		}
	}

	private _serverProps = $state<ApiLlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _serverWarning = $state<string | null>(null);
	private _slotsEndpointAvailable = $state<boolean | null>(null);
	private fetchServerPropsPromise: Promise<void> | null = null;

	private readCachedServerProps(): ApiLlamaCppServerProps | null {
		if (!browser) return null;

		try {
			const raw = localStorage.getItem(SERVER_PROPS_LOCALSTORAGE_KEY);
			if (!raw) return null;

			return JSON.parse(raw) as ApiLlamaCppServerProps;
		} catch (error) {
			console.warn('Failed to read cached server props from localStorage:', error);
			return null;
		}
	}

	private persistServerProps(props: ApiLlamaCppServerProps | null): void {
		if (!browser) return;

		try {
			if (props) {
				localStorage.setItem(SERVER_PROPS_LOCALSTORAGE_KEY, JSON.stringify(props));
			} else {
				localStorage.removeItem(SERVER_PROPS_LOCALSTORAGE_KEY);
			}
		} catch (error) {
			console.warn('Failed to persist server props to localStorage:', error);
		}
	}

	get serverProps(): ApiLlamaCppServerProps | null {
		return this._serverProps;
	}

	get loading(): boolean {
		return this._loading;
	}

	get error(): string | null {
		return this._error;
	}

	get serverWarning(): string | null {
		return this._serverWarning;
	}

	get modelName(): string | null {
		if (this._serverProps?.model_alias) {
			return this._serverProps.model_alias;
		}
		if (!this._serverProps?.model_path) return null;
		return this._serverProps.model_path.split(/(\\|\/)/).pop() || null;
	}

	get supportedModalities(): string[] {
		const modalities: string[] = [];
		if (this._serverProps?.modalities?.audio) {
			modalities.push('audio');
		}
		if (this._serverProps?.modalities?.vision) {
			modalities.push('vision');
		}
		return modalities;
	}

	get supportsVision(): boolean {
		return this._serverProps?.modalities?.vision ?? false;
	}

	get supportsAudio(): boolean {
		return this._serverProps?.modalities?.audio ?? false;
	}

	get slotsEndpointAvailable(): boolean | null {
		return this._slotsEndpointAvailable;
	}

	get serverDefaultParams():
		| ApiLlamaCppServerProps['default_generation_settings']['params']
		| null {
		return this._serverProps?.default_generation_settings?.params || null;
	}

	/**
	 * Check if slots endpoint is available based on server properties and endpoint support
	 */
	private async checkSlotsEndpointAvailability(): Promise<void> {
		if (!this._serverProps) {
			this._slotsEndpointAvailable = false;
			return;
		}

		if (this._serverProps.total_slots <= 0) {
			this._slotsEndpointAvailable = false;
			return;
		}

		try {
			const currentConfig = config();
			const apiKey = currentConfig.apiKey?.toString().trim();

			const response = await fetch(`./slots`, {
				headers: {
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
				}
			});

			if (response.status === 501) {
				console.info('Slots endpoint not implemented - server started without --slots flag');
				this._slotsEndpointAvailable = false;
				return;
			}

			this._slotsEndpointAvailable = true;
		} catch (error) {
			console.warn('Unable to test slots endpoint availability:', error);
			this._slotsEndpointAvailable = false;
		}
	}

	/**
	 * Fetches server properties from the server
	 */
	async fetchServerProps(options: { silent?: boolean } = {}): Promise<void> {
		const { silent = false } = options;
		const isSilent = silent && this._serverProps !== null;

		if (this.fetchServerPropsPromise) {
			return this.fetchServerPropsPromise;
		}

		if (!isSilent) {
			this._loading = true;
			this._error = null;
			this._serverWarning = null;
		}

		const hadProps = this._serverProps !== null;

		const fetchPromise = (async () => {
			try {
				const props = await ChatService.getServerProps();
				this._serverProps = props;
				this.persistServerProps(props);
				this._error = null;
				this._serverWarning = null;
				await this.checkSlotsEndpointAvailability();
			} catch (error) {
				if (isSilent && hadProps) {
					console.warn('Silent server props refresh failed, keeping cached data:', error);
					return;
				}

				this.handleFetchServerPropsError(error, hadProps);
			} finally {
				if (!isSilent) {
					this._loading = false;
				}

				this.fetchServerPropsPromise = null;
			}
		})();

		this.fetchServerPropsPromise = fetchPromise;

		await fetchPromise;
	}

	/**
	 * Handles fetch failures by attempting to recover cached server props and
	 * updating the user-facing error or warning state appropriately.
	 */
	private handleFetchServerPropsError(error: unknown, hadProps: boolean): void {
		const { errorMessage, isOfflineLikeError, isServerSideError } = this.normalizeFetchError(error);

		let cachedProps: ApiLlamaCppServerProps | null = null;

		if (!hadProps) {
			cachedProps = this.readCachedServerProps();

			if (cachedProps) {
				this._serverProps = cachedProps;
				this._error = null;

				if (isOfflineLikeError || isServerSideError) {
					this._serverWarning = errorMessage;
				}

				console.warn(
					'Failed to refresh server properties, using cached values from localStorage:',
					errorMessage
				);
			} else {
				this._error = errorMessage;
			}
		} else {
			this._error = null;

			if (isOfflineLikeError || isServerSideError) {
				this._serverWarning = errorMessage;
			}

			console.warn(
				'Failed to refresh server properties, continuing with cached values:',
				errorMessage
			);
		}

		console.error('Error fetching server properties:', error);
	}

	private normalizeFetchError(error: unknown): {
		errorMessage: string;
		isOfflineLikeError: boolean;
		isServerSideError: boolean;
	} {
		let errorMessage = 'Failed to connect to server';
		let isOfflineLikeError = false;
		let isServerSideError = false;

		if (error instanceof Error) {
			const message = error.message || '';

			if (error.name === 'TypeError' && message.includes('fetch')) {
				errorMessage = 'Server is not running or unreachable';
				isOfflineLikeError = true;
			} else if (message.includes('ECONNREFUSED')) {
				errorMessage = 'Connection refused - server may be offline';
				isOfflineLikeError = true;
			} else if (message.includes('ENOTFOUND')) {
				errorMessage = 'Server not found - check server address';
				isOfflineLikeError = true;
			} else if (message.includes('ETIMEDOUT')) {
				errorMessage = 'Request timed out - the server took too long to respond';
				isOfflineLikeError = true;
			} else if (message.includes('503')) {
				errorMessage = 'Server temporarily unavailable - try again shortly';
				isServerSideError = true;
			} else if (message.includes('500')) {
				errorMessage = 'Server error - check server logs';
				isServerSideError = true;
			} else if (message.includes('404')) {
				errorMessage = 'Server endpoint not found';
			} else if (message.includes('403') || message.includes('401')) {
				errorMessage = 'Access denied';
			}
		}

		return { errorMessage, isOfflineLikeError, isServerSideError };
	}

	/**
	 * Clears the server state
	 */
	clear(): void {
		this._serverProps = null;
		this._error = null;
		this._serverWarning = null;
		this._loading = false;
		this._slotsEndpointAvailable = null;
		this.fetchServerPropsPromise = null;
		this.persistServerProps(null);
	}
}

export const serverStore = new ServerStore();

export const serverProps = () => serverStore.serverProps;
export const serverLoading = () => serverStore.loading;
export const serverError = () => serverStore.error;
export const serverWarning = () => serverStore.serverWarning;
export const modelName = () => serverStore.modelName;
export const supportedModalities = () => serverStore.supportedModalities;
export const supportsVision = () => serverStore.supportsVision;
export const supportsAudio = () => serverStore.supportsAudio;
export const slotsEndpointAvailable = () => serverStore.slotsEndpointAvailable;
export const serverDefaultParams = () => serverStore.serverDefaultParams;
