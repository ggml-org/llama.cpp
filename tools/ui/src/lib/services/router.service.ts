/**
 * RouterService - Builds app route paths
 *
 * Returns chat and settings route strings from a single source of truth
 * (ROUTES). No state.
 */

import { ROUTES } from '$lib/constants';

export class RouterService {
	static chat(id: string): string {
		return `${ROUTES.CHAT}/${id}`;
	}

	static settings(section: string): string {
		return `${ROUTES.SETTINGS}/${section}`;
	}
}
