import { getAuthHeaders } from '$lib/utils';

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
	// ─────────────────────────────────────────────────────────────────────────────
	// Fetching
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetches server properties from the /props endpoint
	 *
	 * @param autoload - If false, prevents automatic model loading (default: false)
	 * @returns {Promise<ApiLlamaCppServerProps>} Server properties
	 * @throws {Error} If the request fails or returns invalid data
	 */
	static async fetch(autoload = false): Promise<ApiLlamaCppServerProps> {
		const url = new URL('./props', window.location.href);
		if (!autoload) {
			url.searchParams.set('autoload', 'false');
		}

		const response = await fetch(url.toString(), {
			headers: getAuthHeaders()
		});

		if (!response.ok) {
			throw new Error(
				`Failed to fetch server properties: ${response.status} ${response.statusText}`
			);
		}

		const data = await response.json();
		return data as ApiLlamaCppServerProps;
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
		const url = new URL('./props', window.location.href);
		url.searchParams.set('model', modelId);
		if (!autoload) {
			url.searchParams.set('autoload', 'false');
		}

		const response = await fetch(url.toString(), {
			headers: getAuthHeaders()
		});

		if (!response.ok) {
			throw new Error(
				`Failed to fetch model properties: ${response.status} ${response.statusText}`
			);
		}

		const data = await response.json();
		return data as ApiLlamaCppServerProps;
	}
}
