/**
 * Sharing conversations to Hugging Face as agent traces.
 *
 * A trace is the JSONL session export, which the Hub reads as its Session Traces Format.
 * Reference: https://huggingface.co/docs/hub/session-traces-format
 */

import { HF_BASE_URL, HF_MAIN_BRANCH } from './huggingface.constants';
import { STORAGE_APP_NAME } from './storage.constants';

export const HF_API_WHOAMI_URL = `${HF_BASE_URL}/api/whoami-v2`;
export const HF_API_CREATE_REPO_URL = `${HF_BASE_URL}/api/repos/create`;
export const HF_API_DATASETS_URL = `${HF_BASE_URL}/api/datasets`;
export const HF_DATASETS_URL = `${HF_BASE_URL}/datasets`;
export const HF_TRACES_DOCS_URL = `${HF_BASE_URL}/docs/hub/agent-traces`;
/** Opens the token form with write access picked, the access a share needs. */
export const HF_NEW_TOKEN_URL = `${HF_BASE_URL}/settings/tokens/new?tokenType=write`;

export const HF_HTTP_UNAUTHORIZED = 401;

export const HF_REPO_TYPE_DATASET = 'dataset';
export const HF_TRACES_REVISION = HF_MAIN_BRANCH;
export const HF_TRACES_DEFAULT_DATASET_NAME = 'llama-chat-traces';

/** Commit API body: one JSON operation per line. */
export const HF_COMMIT_CONTENT_TYPE = 'application/x-ndjson';
/** Bytes of the file the preupload check reads to pick regular or LFS upload. */
export const HF_PREUPLOAD_SAMPLE_BYTES = 512;
export const HF_UPLOAD_MODE_LFS = 'lfs';
/** The Hub takes files up to this size in a commit; larger ones need LFS, which a share does not do. */
export const HF_REGULAR_FILE_MAX_BYTES = 10 * 1024 * 1024;

/** `owner/name`, each part made of letters, digits, `-`, `_` and `.`. */
export const HF_REPO_ID_REGEX = /^[\w.-]+\/[\w.-]+$/;

export const HF_TOKEN_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.huggingFaceToken`;
export const HF_TRACES_DATASET_LOCALSTORAGE_KEY = `${STORAGE_APP_NAME}.huggingFaceTracesDataset`;
