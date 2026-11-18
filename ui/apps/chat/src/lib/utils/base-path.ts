import { asset } from '$app/paths';
import type { AssetPath } from '$app/types';

/**
 * Prefix a server path (e.g. `/props`) with the app base path.
 * asset() only types files from `static/`, but its runtime prefix is the base path for any file.
 */
export function withBase(path: string): string {
	return asset(path.replace(/^\//, '') as AssetPath);
}
