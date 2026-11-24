import type { PageLoad } from './$types';
import { redirect } from '@sveltejs/kit';
import { AGENTIC_ENABLED, ROUTES } from '#lib/constants/index.ts';
import { loadAppRoute } from '#lib/stores/init.ts';

export const load: PageLoad = ({ fetch }) => {
	// a chat-only build has no MCP
	if (!AGENTIC_ENABLED) redirect(307, ROUTES.START);

	return loadAppRoute(fetch);
};
