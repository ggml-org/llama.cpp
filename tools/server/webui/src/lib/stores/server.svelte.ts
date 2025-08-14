import { ChatService } from '$lib/services/chat';

/**
 * Server properties store
 * Manages server information including model, build info, and supported modalities
 */
class ServerStore {
	private _serverProps = $state<ApiLlamaCppServerProps | null>(null);
	private _loading = $state(false);
	private _error = $state<string | null>(null);

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

	async fetchServerProps(): Promise<void> {
		this._loading = true;
		this._error = null;

		try {
			console.log('Fetching server properties...');
			const props = await ChatService.getServerProps();
			this._serverProps = props;
			console.log('Server properties loaded:', props);
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
					errorMessage = 'Access denied - check server permissions';
				}
			}
			
			this._error = errorMessage;
			console.error('Error fetching server properties:', error);
		} finally {
			this._loading = false;
		}
	}

	clear(): void {
		this._serverProps = null;
		this._error = null;
		this._loading = false;
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
