/**
 * Llama Hugging Face: a Hugging Face account for an app that runs in the browser, with no
 * server of its own. Signing in ("Sign in with Hugging Face", or a pasted access token) and
 * sharing a conversation to a dataset as an agent trace.
 *
 * This entry is plain TypeScript: the OAuth flow and the Hub calls. The account as reactive
 * state and the components that show it are in @ggml-org/llama-huggingface/svelte.
 * It knows no app: the app says who it is (HfAccountOptions) and hands over what to share.
 */

export {
	authorizationUrl,
	createCodeChallenge,
	createRandomString,
	exchangeCode,
	hasScope,
	refreshTokens,
	registerClient
} from './client.js';
export {
	HF_BASE_URL,
	HF_NEW_TOKEN_URL,
	HF_OAUTH_CALLBACK_PARAM,
	HF_PREVIEW_LINES_MAX,
	HF_PREVIEW_TEXT_MAX,
	HF_REGULAR_FILE_MAX_BYTES,
	HF_REPO_ID_REGEX,
	HF_REPO_ID_SEPARATOR,
	HF_STORAGE_KEY_DATASET,
	HF_STORAGE_KEY_OAUTH,
	HF_STORAGE_KEY_SEPARATOR,
	HF_STORAGE_KEY_TOKEN,
	HF_TRACES_DOCS_URL
} from './constants.js';
export {
	HfOAuthErrorCode,
	HfOAuthFlowMode,
	HfOAuthGrantType,
	HfOAuthParam,
	HfTokenRole
} from './enums.js';
export { HfNotWritableError, HfTokenRejectedError } from './errors.js';
export { ensureDataset, listDatasets, uploadTrace, whoAmI } from './hub.js';
export { HF_ACTION, HF_MESSAGE } from './messages.js';
export { HfOAuthSignIn } from './sign-in.js';
export { errorMessage, splitRepoId } from './util.js';
export type {
	HfAccountOptions,
	HfDataset,
	HfOAuthCallbackOutcome,
	HfOAuthClient,
	HfOAuthPendingFlow,
	HfOAuthRecord,
	HfOAuthResult,
	HfOAuthSignInOptions,
	HfOAuthTokens,
	HfShareScope,
	HfTraceFile,
	HfTracePreviewLine,
	HfTraceUpload,
	HfUser
} from './types.js';
