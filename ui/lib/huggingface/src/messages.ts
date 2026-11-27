/**
 * What the package says when something goes wrong, and what it calls the calls that can.
 * Named here, so a wording is changed in one place and a test can read the same text.
 */

/** What an error names a failed call as: "Dataset list failed (500): ...". */
export const HF_ACTION = {
	ACCOUNT_CHECK: 'Sign in',
	DATASET_CREATION: 'Dataset creation',
	DATASET_LIST: 'Dataset list',
	DATASET_LOOKUP: 'Dataset lookup',
	SIGN_IN: 'The sign-in',
	SIGN_IN_RENEWAL: 'Renewing the sign-in',
	UPLOAD: 'Upload',
	UPLOAD_CHECK: 'Upload check'
} as const;

export type HfAction = (typeof HF_ACTION)[keyof typeof HF_ACTION];

export const HF_MESSAGE = {
	ACCESS_TOKEN_MISSING: 'Hugging Face sent no access token back.',
	APP_NOT_REGISTERED: 'Could not reach Hugging Face to start the sign-in.',
	CLIENT_ID_MISSING: 'Hugging Face registered the app but sent no client ID back.',
	CODE_MISSING: 'Hugging Face sent no code back.',
	REQUEST_TIMED_OUT: 'Hugging Face did not answer in time. Try again.',
	SCOPE_NOT_GRANTED: 'Hugging Face did not grant the access the app asked for.',
	SECURE_PAGE_NEEDED: 'Signing in with Hugging Face needs a secure page: https, or localhost.',
	// not the error's description: it comes from the address bar, anyone could have written it
	SIGN_IN_CANCELLED: 'The sign-in was cancelled.',
	SIGN_IN_FAILED_FALLBACK: 'Sign in with an access token instead.',
	SIGN_IN_REFUSED: 'Hugging Face refused the sign-in.',
	SIGN_IN_REQUIRED: 'Sign in to Hugging Face first.',
	TOKEN_REJECTED: 'Hugging Face did not accept this token. Check it and try again.',
	TRACE_TOO_LARGE:
		'This trace is too large to share. Leave attachments out or share fewer messages.',
	invalidRepoId: (repoId: string) => `${repoId} is not a dataset name of the form owner/name.`,
	lfsNotSupported: (repoId: string) =>
		`${repoId} stores .jsonl files with LFS, which sharing does not support yet.`,
	notWritable: (repoId: string) =>
		`You cannot add traces to ${repoId}. Signing in with Hugging Face lets the app write only to datasets it created: pick a new dataset name, or sign in with an access token that has write access.`,
	registrationRefused: (origin: string, detail: string) =>
		`Hugging Face does not allow signing in from ${origin} (${detail}).`,
	requestFailed: (action: HfAction, status: number, detail: string) =>
		`${action} failed (${status}): ${detail}`,
	status: (status: number) => `HTTP ${status}`,
	tokenRequestFailed: (action: HfAction, detail: string) => `${action} failed: ${detail}.`,
	tokenRequestUnreachable: (action: HfAction) => `${action} could not reach Hugging Face.`
} as const;
