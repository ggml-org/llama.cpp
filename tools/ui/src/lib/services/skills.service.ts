import { API_SKILLS, X_SKILL_CWD_HEADER } from '$lib/constants';
import type { SkillCatalogResponse, SkillReadRequest, SkillReadResult } from '$lib/types';
import { apiFetch } from '$lib/utils';

/**
 * SkillsService - stateless transport for the dedicated llama-server Skills API.
 *
 * The server owns host-filesystem discovery, resolution, identity,
 * containment, parsing, and XML. The client sends only `name` and an optional
 * `path` on reads, plus an optional validated CWD header; origin fields are
 * never client-supplied. Errors and aborts propagate unchanged through the
 * shared `apiFetch` envelope.
 */
export class SkillsService {
	/**
	 * Fetch the deterministic catalog for an effective CWD.
	 *
	 * An absent `cwd` means the canonical server process CWD; a selected
	 * non-whitespace CWD is sent as `X-Skill-Cwd`. Whitespace-only values are
	 * treated as absent before transport.
	 */
	static async list(cwd?: string, signal?: AbortSignal): Promise<SkillCatalogResponse> {
		return apiFetch<SkillCatalogResponse>(API_SKILLS.LIST, {
			headers: skillCwdHeaders(cwd),
			signal
		});
	}

	/**
	 * Read a skill's current base content or a resource.
	 *
	 * Sends exactly `{ name }` or `{ name, path }` — never identity, scope,
	 * provider, root, or absolute path. The server re-resolves the name for the
	 * effective CWD on every request.
	 */
	static async read(
		request: SkillReadRequest,
		cwd?: string,
		signal?: AbortSignal
	): Promise<SkillReadResult> {
		return apiFetch<SkillReadResult>(API_SKILLS.READ, {
			body: JSON.stringify(
				request.path !== undefined
					? { name: request.name, path: request.path }
					: { name: request.name }
			),
			headers: skillCwdHeaders(cwd),
			method: 'POST',
			signal
		});
	}
}

/**
 * Map a selected CWD to the Skills request header, or absent when no
 * non-whitespace CWD is selected (the server then uses its process CWD).
 */
function skillCwdHeaders(cwd: string | undefined): Record<string, string> | undefined {
	if (cwd === undefined || cwd.trim().length === 0) {
		return undefined;
	}

	return { [X_SKILL_CWD_HEADER]: cwd };
}
