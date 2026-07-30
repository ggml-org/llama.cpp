import { apiFetch, apiPost } from '$lib/utils';
import { API_FILESYSTEM } from '$lib/constants';
import type {
	ApiFilesystemSearchRequest,
	ApiFilesystemSearchResponse,
	ApiFilesystemRootsResponse,
	ApiFilesystemGitRequest,
	ApiFilesystemGitResponse
} from '$lib/types';

export class FilesystemService {
	/**
	 * Query the server filesystem for entries matching `body.query`.
	 * The endpoint is gated on `--tools` / `--agent` on the server side;
	 * calls made without those flags enabled return a 501.
	 *
	 * Pass an `AbortSignal` to cancel an in-flight query (e.g. when the
	 * user keeps typing and we supersede the previous fetch with a newer
	 * one). On abort the underlying `fetch` is rejected, surfaced as an
	 * `ApiError`; callers should detect it via `signal.aborted` rather
	 * than parsing the message because `apiPost` wraps the rejection.
	 */
	static async search(
		body: ApiFilesystemSearchRequest,
		signal?: AbortSignal
	): Promise<ApiFilesystemSearchResponse> {
		return apiPost<ApiFilesystemSearchResponse>(API_FILESYSTEM.SEARCH, body, { signal });
	}

	/**
	 * Discover the browse roots the server has configured. The first entry
	 * is the implicit default (server falls back to `$HOME` when the user
	 * did not pass any `--browse-root`); subsequent entries are additional
	 * scopes to pick from when the user has multiple roots.
	 */
	static async getRoots(signal?: AbortSignal): Promise<ApiFilesystemRootsResponse> {
		return apiFetch<ApiFilesystemRootsResponse>(API_FILESYSTEM.ROOTS, { signal });
	}

	/**
	 * Probe `path` for git-repository metadata. Server walks up from the
	 * path looking for `.git/`, capped at 8 ancestors and bounded to the
	 * configured browse roots so a malicious input cannot probe outside
	 * the sandbox. A non-repo path is reported with `is_repo=false` rather
	 * than an HTTP error, so callers can just check that flag.
	 *
	 * Pass an `AbortSignal` to cancel an in-flight request (e.g. when the
	 * user rapidly switches the working directory and we supersede the
	 * previous fetch with a newer one).
	 */
	static async getGitInfo(
		body: ApiFilesystemGitRequest,
		signal?: AbortSignal
	): Promise<ApiFilesystemGitResponse> {
		return apiPost<ApiFilesystemGitResponse>(API_FILESYSTEM.GIT, body, { signal });
	}
}
