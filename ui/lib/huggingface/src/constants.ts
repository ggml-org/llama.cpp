/**
 * Hugging Face endpoints and the values the Hub API and the OAuth flow take.
 * References: https://huggingface.co/docs/hub/oauth, https://huggingface.co/docs/hub/api,
 * https://huggingface.co/docs/hub/session-traces-format
 */

export const HF_BASE_URL = 'https://huggingface.co';

// OAuth

export const HF_OAUTH_AUTHORIZE_URL = `${HF_BASE_URL}/oauth/authorize`;
export const HF_OAUTH_REGISTER_URL = `${HF_BASE_URL}/oauth/register`;
export const HF_OAUTH_TOKEN_URL = `${HF_BASE_URL}/oauth/token`;

/** Query parameter that marks a page load as Hugging Face sending the browser back. */
export const HF_OAUTH_CALLBACK_PARAM = 'hf_oauth';
export const HF_OAUTH_CALLBACK_VALUE = '1';

export const HF_OAUTH_SCOPE_SEPARATOR = ' ';
export const HF_OAUTH_RESPONSE_TYPE = 'code';
export const HF_OAUTH_CODE_CHALLENGE_METHOD = 'S256';
export const HF_OAUTH_CODE_CHALLENGE_HASH = 'SHA-256';
/** A public client: the browser holds no secret. */
export const HF_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD = 'none';
/** Bytes of randomness in a PKCE verifier and in a state. */
export const HF_OAUTH_RANDOM_BYTES = 32;

export const HF_MS_PER_SECOND = 1000;
/** A sign-in that has not come back by then is dropped. */
export const HF_OAUTH_PENDING_TTL_MS = 15 * 60 * HF_MS_PER_SECOND;
/** An access token is renewed this long before it expires. */
export const HF_OAUTH_EXPIRY_MARGIN_MS = 60 * HF_MS_PER_SECOND;

/** The window a sign-in opens the authorization page in. */
export const HF_OAUTH_POPUP_NAME = 'hf-oauth';
export const HF_OAUTH_POPUP_FEATURES = 'popup,width=520,height=720';
/** Where the popup waits while the app registers with Hugging Face. */
export const HF_OAUTH_POPUP_BLANK_URL = 'about:blank';

// HTTP

export const HF_HEADER_AUTHORIZATION = 'Authorization';
export const HF_HEADER_CONTENT_TYPE = 'Content-Type';
export const HF_BEARER_PREFIX = 'Bearer ';
export const HF_CONTENT_TYPE_JSON = 'application/json';
export const HF_CONTENT_TYPE_FORM = 'application/x-www-form-urlencoded';
/** Commit API body: one JSON operation per line. */
export const HF_CONTENT_TYPE_NDJSON = 'application/x-ndjson';
export const HF_HTTP_POST = 'POST';
/** A Hub call that has not answered by then is given up, so it cannot hold a dialog open. */
export const HF_REQUEST_TIMEOUT_MS = 60 * HF_MS_PER_SECOND;
/**
 * The slowest uplink a call is given time for: its timeout grows by the time its body takes to
 * send at this speed, so a large trace on a slow link is not given up while it is going out.
 */
export const HF_UPLOAD_MIN_BYTES_PER_SECOND = 32 * 1024;
/** The name of the error `AbortSignal.timeout` aborts a request with. */
export const HF_TIMEOUT_ERROR_NAME = 'TimeoutError';
/** The first status of a request the Hub turned down, and of one it failed to answer. */
export const HF_HTTP_CLIENT_ERROR_MIN = 400;
export const HF_HTTP_SERVER_ERROR_MIN = 500;
export const HF_HTTP_UNAUTHORIZED = 401;
export const HF_HTTP_FORBIDDEN = 403;
export const HF_HTTP_NOT_FOUND = 404;
export const HF_HTTP_CONFLICT = 409;

// Hub API

export const HF_API_WHOAMI_URL = `${HF_BASE_URL}/api/whoami-v2`;
export const HF_API_CREATE_REPO_URL = `${HF_BASE_URL}/api/repos/create`;
export const HF_API_DATASETS_URL = `${HF_BASE_URL}/api/datasets`;
export const HF_DATASETS_URL = `${HF_BASE_URL}/datasets`;
export const HF_TRACES_DOCS_URL = `${HF_BASE_URL}/docs/hub/agent-traces`;
/** Opens the token form with write access picked, the access a share needs. */
export const HF_NEW_TOKEN_URL = `${HF_BASE_URL}/settings/tokens/new?tokenType=write`;

export const HF_REPO_TYPE_DATASET = 'dataset';
export const HF_MAIN_BRANCH = 'main';
/** Separates the owner from the name in a repo ID. */
export const HF_REPO_ID_SEPARATOR = '/';
/** `owner/name`, each part made of letters, digits, `-`, `_` and `.`. */
export const HF_REPO_ID_REGEX = /^[\w.-]+\/[\w.-]+$/;

/** Datasets listed to pick from. */
export const HF_DATASETS_LIST_LIMIT = 100;
/** Query of the dataset list: an account's datasets, most recently changed first. */
export const HF_DATASETS_LIST_QUERY = {
	direction: '-1',
	limit: String(HF_DATASETS_LIST_LIMIT),
	sort: 'lastModified'
} as const;
export const HF_DATASETS_AUTHOR_PARAM = 'author';

export const HF_COMMIT_LINE_SEPARATOR = '\n';
export const HF_COMMIT_KEY_HEADER = 'header';
export const HF_COMMIT_KEY_FILE = 'file';
export const HF_COMMIT_ENCODING = 'base64';
/** Bytes turned to base64 at a time, under the argument limit of String.fromCharCode. */
export const HF_BASE64_CHUNK_BYTES = 0x8000;
/** Bytes of the file the preupload check reads to pick regular or LFS upload. */
export const HF_PREUPLOAD_SAMPLE_BYTES = 512;
export const HF_UPLOAD_MODE_LFS = 'lfs';
/** The Hub takes files up to this size in a commit; larger ones need LFS, which a share does not do. */
export const HF_REGULAR_FILE_MAX_BYTES = 10 * 1024 * 1024;

// What the account keeps in localStorage, after the app's own prefix

export const HF_STORAGE_KEY_OAUTH = 'huggingFaceOAuth';
export const HF_STORAGE_KEY_TOKEN = 'huggingFaceToken';
export const HF_STORAGE_KEY_DATASET = 'huggingFaceTracesDataset';
export const HF_STORAGE_KEY_SHARED_DATASETS = 'huggingFaceSharedDatasets';
export const HF_STORAGE_KEY_SEPARATOR = '.';

/** The name of the error for a dataset the account cannot write to. */
export const HF_NOT_WRITABLE_ERROR_NAME = 'HfNotWritableError';
/** The name of the error for a token the Hugging Face turned down. */
export const HF_TOKEN_REJECTED_ERROR_NAME = 'HfTokenRejectedError';
/** Messages of a conversation a finished share shows faded, and the text kept of each. */
export const HF_PREVIEW_LINES_MAX = 4;
export const HF_PREVIEW_TEXT_MAX = 200;

/** Choice value of the "New dataset" tile; repo IDs always hold a slash, so it never clashes. */
export const HF_NEW_DATASET_VALUE = 'new';
/** Datasets a sign-in remembers having shared to, per account: the ones it can write to again. */
export const HF_SHARED_DATASETS_MAX = 20;
/** Dataset tiles a share shows before "Show all". */
export const HF_DATASET_TILES_SHOWN = 3;
