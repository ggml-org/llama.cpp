import { apiFetchWithParams } from '$lib/utils';

/**
 * PropsService - Server properties management
 *
 * This service handles communication with the /props endpoint to retrieve
 * server configuration, model information, and capabilities.
 *
 * **Responsibilities:**
 * - Fetch server properties from /props endpoint
 * - Handle API authentication
 * - Parse and validate server response
 *
 * **Used by:**
 * - serverStore: Primary consumer for server state management
 */
export class PropsService {
	/**
	 *
	 *
	 * Fetching
	 *
	 *
	 */

	/**
	 * Fetches server properties from the /props endpoint
	 *
	 * @param autoload - If false, prevents automatic model loading (default: false)
	 * @returns {Promise<ApiLlamaCppServerProps>} Server properties
	 * @throws {Error} If the request fails or returns invalid data
	 */
	static async fetch(autoload = false): Promise<ApiLlamaCppServerProps> {
		const params: Record<string, string> = {};
		if (!autoload) {
			params.autoload = 'false';
		}

		return apiFetchWithParams<ApiLlamaCppServerProps>('./props', params, { authOnly: true });
	}

	/**
	 * Fetches server properties for a specific model (ROUTER mode)
	 *
	 * @param modelId - The model ID to fetch properties for
	 * @param autoload - If false, prevents automatic model loading (default: false)
	 * @returns {Promise<ApiLlamaCppServerProps>} Server properties for the model
	 * @throws {Error} If the request fails or returns invalid data
	 */
	static async fetchForModel(modelId: string, autoload = false): Promise<ApiLlamaCppServerProps> {
		const params: Record<string, string> = { model: modelId };
		if (!autoload) {
			params.autoload = 'false';
		}

		return apiFetchWithParams<ApiLlamaCppServerProps>('./props', params, { authOnly: true });
	}
}
