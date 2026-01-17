import type { PageLoad } from './$types';
import { validateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch }) => {
	await validateApiKey(fetch);
};
