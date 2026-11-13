import type { PageLoad } from './$types';
import { loadAppRoute } from '$lib/stores/init';

export const load: PageLoad = ({ fetch }) => loadAppRoute(fetch);
