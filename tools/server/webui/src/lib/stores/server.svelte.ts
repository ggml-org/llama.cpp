import { ApiService, type LlamaCppServerProps } from '$lib/services/api';

/**
 * Server properties store
 * Manages server information including model, build info, and supported modalities
 */
class ServerStore {
	private _serverProps = $state<LlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);

	/**
	 * Get current server properties
	 */
	get serverProps(): LlamaCppServerProps | null {
		return this._serverProps;
	}

	/**
	 * Get loading state
	 */
	get loading(): boolean {
		return this._loading;
	}

	/**
	 * Get error state
	 */
	get error(): string | null {
		return this._error;
	}

	/**
	 * Get model name from model path
	 */
	get modelName(): string | null {
		if (!this._serverProps?.model_path) return null;
		return this._serverProps.model_path.split(/(\\|\/)/).pop() || null;
	}

	/**
	 * Get supported modalities
	 */
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

	/**
	 * Check if vision is supported
	 */
	get supportsVision(): boolean {
		return this._serverProps?.modalities?.vision ?? false;
	}

	/**
	 * Check if audio is supported
	 */
	get supportsAudio(): boolean {
		return this._serverProps?.modalities?.audio ?? false;
	}

	/**
	 * Fetch server properties from API
	 */
	async fetchServerProps(): Promise<void> {
		this._loading = true;
		this._error = null;

		try {
			console.log('Fetching server properties...');
			const props = await ApiService.getServerProps();
			this._serverProps = props;
			console.log('Server properties loaded:', props);
		} catch (error) {
			const errorMessage = error instanceof Error ? error.message : 'Failed to fetch server properties';
			this._error = errorMessage;
			console.error('Error fetching server properties:', error);
		} finally {
			this._loading = false;
		}
	}

	/**
	 * Clear server properties
	 */
	clear(): void {
		this._serverProps = null;
		this._error = null;
		this._loading = false;
	}
}

// Export singleton instance
export const serverStore = new ServerStore();

// Export reactive getters for easy access
export const serverProps = () => serverStore.serverProps;
export const serverLoading = () => serverStore.loading;
export const serverError = () => serverStore.error;
export const modelName = () => serverStore.modelName;
export const supportedModalities = () => serverStore.supportedModalities;
export const supportsVision = () => serverStore.supportsVision;
export const supportsAudio = () => serverStore.supportsAudio;
