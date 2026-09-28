/**
 * Backend availability for connection error states.
 *
 * The server error banner and the offline affordances should only appear when
 * no backend can serve requests. The active backend failing is not enough:
 * with an external backend configured, the UI stays usable.
 */

import { LOCAL_BACKEND_ID } from '$lib/constants';
import { backendsModelsStore, backendsStore, serverStore } from '$lib/stores';

export interface BackendAvailability {
	readonly hasAvailableBackend: boolean;
	readonly isOffline: boolean;
}

export function useBackendAvailability(): BackendAvailability {
	const hasAvailableBackend = $derived.by(() => {
		// no connection error at all: nothing to report
		if (!serverStore.error) return true;

		// the local server failed, but an external backend may still be usable
		return backendsStore.enabled.some(
			(backend) =>
				backend.id !== LOCAL_BACKEND_ID && backendsModelsStore.get(backend.id).error === null
		);
	});

	return {
		get hasAvailableBackend() {
			return hasAvailableBackend;
		},

		get isOffline() {
			return !hasAvailableBackend;
		}
	};
}
