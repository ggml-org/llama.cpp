/**
 * The Hub API calls of an account: who it is, its datasets, and committing a trace to one.
 * Stateless, and callable from the browser: the Hub API allows cross-origin requests.
 * Reference: https://huggingface.co/docs/hub/api
 */

import {
	HF_API_CREATE_REPO_URL,
	HF_API_DATASETS_URL,
	HF_API_WHOAMI_URL,
	HF_BEARER_PREFIX,
	HF_COMMIT_ENCODING,
	HF_COMMIT_KEY_FILE,
	HF_COMMIT_KEY_HEADER,
	HF_COMMIT_LINE_SEPARATOR,
	HF_CONTENT_TYPE_JSON,
	HF_CONTENT_TYPE_NDJSON,
	HF_DATASETS_AUTHOR_PARAM,
	HF_DATASETS_LIST_QUERY,
	HF_DATASETS_URL,
	HF_HEADER_AUTHORIZATION,
	HF_HEADER_CONTENT_TYPE,
	HF_HTTP_CONFLICT,
	HF_HTTP_FORBIDDEN,
	HF_HTTP_NOT_FOUND,
	HF_HTTP_POST,
	HF_HTTP_UNAUTHORIZED,
	HF_MAIN_BRANCH,
	HF_MS_PER_SECOND,
	HF_PREUPLOAD_SAMPLE_BYTES,
	HF_REGULAR_FILE_MAX_BYTES,
	HF_REPO_ID_REGEX,
	HF_REPO_TYPE_DATASET,
	HF_REQUEST_TIMEOUT_MS,
	HF_TIMEOUT_ERROR_NAME,
	HF_UPLOAD_MIN_BYTES_PER_SECOND,
	HF_UPLOAD_MODE_LFS
} from './constants.js';
import { HfNotWritableError } from './errors.js';
import { HF_ACTION, type HfAction, HF_MESSAGE } from './messages.js';
import type { HfDataset, HfTraceUpload, HfUser } from './types.js';
import { bytesToBase64, splitRepoId } from './util.js';

/** The headers of a Hub call: the token, and what the body is when there is one. */
function headers(token: string, contentType?: string): Record<string, string> {
	return {
		[HF_HEADER_AUTHORIZATION]: `${HF_BEARER_PREFIX}${token}`,
		...(contentType ? { [HF_HEADER_CONTENT_TYPE]: contentType } : {})
	};
}

/**
 * How long a call may take: the usual limit, and the time its body needs to go out on the
 * slowest link it is given time for. The timeout counts the upload too.
 */
function timeoutFor(init: RequestInit): number {
	const bodyBytes = typeof init.body === 'string' ? init.body.length : 0;

	return (
		HF_REQUEST_TIMEOUT_MS + Math.ceil(bodyBytes / HF_UPLOAD_MIN_BYTES_PER_SECOND) * HF_MS_PER_SECOND
	);
}

/** A request to the Hub that is given up after its timeout instead of waiting for good. */
async function hubFetch(url: string, init: RequestInit = {}): Promise<Response> {
	try {
		return await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutFor(init)) });
	} catch (error) {
		if (error instanceof Error && error.name === HF_TIMEOUT_ERROR_NAME) {
			throw new Error(HF_MESSAGE.REQUEST_TIMED_OUT, { cause: error });
		}

		throw error;
	}
}

/** Throws with the Hub's own error text when it gives one. */
async function ensureOk(response: Response, action: HfAction): Promise<void> {
	if (response.ok) return;

	let detail = response.statusText;

	try {
		const body = await response.json();

		if (typeof body?.error === 'string') detail = body.error;
	} catch {
		// not JSON; keep the status text
	}

	throw new Error(HF_MESSAGE.requestFailed(action, response.status, detail));
}

/**
 * A sign-in with Hugging Face reaches only the datasets the app created, so the Hub turns down
 * a share to any other dataset: a 403, or a 409 for creating one the sign-in cannot see.
 */
function ensureWritable(response: Response, repoId: string): void {
	if (response.status !== HF_HTTP_FORBIDDEN && response.status !== HF_HTTP_CONFLICT) return;

	throw new HfNotWritableError(repoId);
}

/** Throws unless the ID is `owner/name`: the Hub's URLs and the creation call both read it. */
function ensureRepoId(repoId: string): void {
	if (!HF_REPO_ID_REGEX.test(repoId)) throw new Error(HF_MESSAGE.invalidRepoId(repoId));
}

/**
 * Reads the account a token belongs to.
 * @returns Null when the Hub does not accept the token: it is wrong, revoked or expired
 * @throws When the Hub cannot be asked
 */
export async function whoAmI(token: string): Promise<HfUser | null> {
	const response = await hubFetch(HF_API_WHOAMI_URL, { headers: headers(token) });

	if (response.status === HF_HTTP_UNAUTHORIZED) return null;

	await ensureOk(response, HF_ACTION.ACCOUNT_CHECK);

	const body = await response.json();

	return {
		fullname: body.fullname,
		name: body.name,
		tokenRole: body.auth?.accessToken?.role
	};
}

/**
 * Lists the datasets of an account, most recently changed first; the token brings in the
 * private ones it may read.
 */
export async function listDatasets(token: string, author: string): Promise<HfDataset[]> {
	const query = new URLSearchParams({
		...HF_DATASETS_LIST_QUERY,
		[HF_DATASETS_AUTHOR_PARAM]: author
	});
	const response = await hubFetch(`${HF_API_DATASETS_URL}?${query}`, {
		headers: headers(token)
	});

	await ensureOk(response, HF_ACTION.DATASET_LIST);

	const body: unknown = await response.json();

	if (!Array.isArray(body)) return [];

	return body
		.filter((dataset) => typeof dataset?.id === 'string')
		.map((dataset) => ({ id: dataset.id, isPrivate: dataset.private === true }));
}

/** Creates the dataset unless it already exists; an existing one keeps its visibility. */
export async function ensureDataset(
	token: string,
	repoId: string,
	isPrivate: boolean
): Promise<void> {
	ensureRepoId(repoId);

	const existing = await hubFetch(`${HF_API_DATASETS_URL}/${repoId}`, {
		headers: headers(token)
	});

	if (existing.ok) return;

	if (existing.status !== HF_HTTP_NOT_FOUND) await ensureOk(existing, HF_ACTION.DATASET_LOOKUP);

	const [organization, name] = splitRepoId(repoId);
	const response = await hubFetch(HF_API_CREATE_REPO_URL, {
		body: JSON.stringify({ name, organization, private: isPrivate, type: HF_REPO_TYPE_DATASET }),
		headers: headers(token, HF_CONTENT_TYPE_JSON),
		method: HF_HTTP_POST
	});

	ensureWritable(response, repoId);
	await ensureOk(response, HF_ACTION.DATASET_CREATION);
}

/**
 * Commits one trace file to a dataset, creating the dataset first when needed.
 * @returns The URL of the file on the Hub
 */
export async function uploadTrace(token: string, upload: HfTraceUpload): Promise<string> {
	const { commitMessage, content, isPrivate, path, repoId } = upload;
	const bytes = new TextEncoder().encode(content);

	if (bytes.length > HF_REGULAR_FILE_MAX_BYTES) throw new Error(HF_MESSAGE.TRACE_TOO_LARGE);

	await ensureDataset(token, repoId, isPrivate);

	const repoApi = `${HF_API_DATASETS_URL}/${repoId}`;
	// a dataset whose .gitattributes sends .jsonl to LFS needs an upload this share does not do
	const preupload = await hubFetch(`${repoApi}/preupload/${HF_MAIN_BRANCH}`, {
		body: JSON.stringify({
			files: [
				{
					path,
					sample: bytesToBase64(bytes.subarray(0, HF_PREUPLOAD_SAMPLE_BYTES)),
					size: bytes.length
				}
			]
		}),
		headers: headers(token, HF_CONTENT_TYPE_JSON),
		method: HF_HTTP_POST
	});

	ensureWritable(preupload, repoId);
	await ensureOk(preupload, HF_ACTION.UPLOAD_CHECK);

	const { files } = await preupload.json();

	if (files?.[0]?.uploadMode === HF_UPLOAD_MODE_LFS) {
		throw new Error(HF_MESSAGE.lfsNotSupported(repoId));
	}

	const body = [
		{ key: HF_COMMIT_KEY_HEADER, value: { description: '', summary: commitMessage } },
		{
			key: HF_COMMIT_KEY_FILE,
			value: { content: bytesToBase64(bytes), encoding: HF_COMMIT_ENCODING, path }
		}
	]
		.map((line) => JSON.stringify(line))
		.join(HF_COMMIT_LINE_SEPARATOR);
	const commit = await hubFetch(`${repoApi}/commit/${HF_MAIN_BRANCH}`, {
		body,
		headers: headers(token, HF_CONTENT_TYPE_NDJSON),
		method: HF_HTTP_POST
	});

	ensureWritable(commit, repoId);
	await ensureOk(commit, HF_ACTION.UPLOAD);

	return `${HF_DATASETS_URL}/${repoId}/blob/${HF_MAIN_BRANCH}/${encodeURIComponent(path)}`;
}
