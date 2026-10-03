/**
 * Backend-qualified model option ids.
 *
 * The selector lists models from every enabled backend, so a bare model id can
 * collide. Option ids are qualified as `<backendId>::<rawModelId>`; selection
 * strips the prefix to find the row in the active backend's list.
 */

const MODEL_OPTION_ID_SEPARATOR = '::';

/** Prefix a raw model id with the backend that serves it. */
export function qualifyModelId(backendId: string, modelId: string): string {
	return `${backendId}${MODEL_OPTION_ID_SEPARATOR}${modelId}`;
}

/** Backend id of a qualified model option id, or null when unqualified. */
export function backendIdFromModelId(qualifiedId: string): string | null {
	const index = qualifiedId.indexOf(MODEL_OPTION_ID_SEPARATOR);

	return index === -1 ? null : qualifiedId.slice(0, index);
}

/** Strip the backend prefix from a qualified model option id. */
export function rawModelId(qualifiedId: string): string {
	const index = qualifiedId.indexOf(MODEL_OPTION_ID_SEPARATOR);

	return index === -1 ? qualifiedId : qualifiedId.slice(index + MODEL_OPTION_ID_SEPARATOR.length);
}
