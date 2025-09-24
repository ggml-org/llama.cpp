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
	private _serverProps = $state<ApiLlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);
	private _slotsEndpointAvailable = $state<boolean | null>(null);

	get serverProps(): ApiLlamaCppServerProps | null {
		return this._serverProps;
	}

	get loading(): boolean {
		return this._loading;
	}

	get error(): string | null {
		return this._error;
	}

	get modelName(): string | null {
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

			const response = await fetch('/slots', {
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
	async fetchServerProps(): Promise<void> {
		this._loading = true;
		this._error = null;

		try {
			console.log('Fetching server properties...');
			const props = await ChatService.getServerProps();
			this._serverProps = props;
			console.log('Server properties loaded:', props);

			// Check slots endpoint availability after server props are loaded
			await this.checkSlotsEndpointAvailability();
		} catch (error) {
			let errorMessage = 'Failed to connect to server';

			if (error instanceof Error) {
				// Handle specific error types with user-friendly messages
				if (error.name === 'TypeError' && error.message.includes('fetch')) {
					errorMessage = 'Server is not running or unreachable';
				} else if (error.message.includes('ECONNREFUSED')) {
					errorMessage = 'Connection refused - server may be offline';
				} else if (error.message.includes('ENOTFOUND')) {
					errorMessage = 'Server not found - check server address';
				} else if (error.message.includes('ETIMEDOUT')) {
					errorMessage = 'Connection timeout - server may be overloaded';
				} else if (error.message.includes('500')) {
					errorMessage = 'Server error - check server logs';
				} else if (error.message.includes('404')) {
					errorMessage = 'Server endpoint not found';
				} else if (error.message.includes('403') || error.message.includes('401')) {
					errorMessage = 'Access denied';
				}
			}

			this._error = errorMessage;
			console.error('Error fetching server properties:', error);
		} finally {
			this._loading = false;
		}
	}

	/**
	 * Clears the server state
	 */
	clear(): void {
		this._serverProps = null;
		this._error = null;
		this._loading = false;
		this._slotsEndpointAvailable = null;
	}
}

export const serverStore = new ServerStore();

export const serverProps = () => serverStore.serverProps;
export const serverLoading = () => serverStore.loading;
export const serverError = () => serverStore.error;
export const modelName = () => serverStore.modelName;
export const supportedModalities = () => serverStore.supportedModalities;
export const supportsVision = () => serverStore.supportsVision;
export const supportsAudio = () => serverStore.supportsAudio;
export const slotsEndpointAvailable = () => serverStore.slotsEndpointAvailable;
