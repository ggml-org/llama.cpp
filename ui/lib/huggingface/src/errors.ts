/** The errors a caller can tell from the rest and act on. */

import { HF_NOT_WRITABLE_ERROR_NAME, HF_TOKEN_REJECTED_ERROR_NAME } from './constants.js';
import { HF_MESSAGE } from './messages.js';

/** The Hub turned down a share because the account cannot write to the dataset. */
export class HfNotWritableError extends Error {
	readonly repoId: string;

	constructor(repoId: string) {
		super(HF_MESSAGE.notWritable(repoId));
		this.name = HF_NOT_WRITABLE_ERROR_NAME;
		this.repoId = repoId;
	}
}

/**
 * The token endpoint turned a request down (a 4xx), as it does a refresh token that was
 * revoked or has expired for good. An endpoint that could not be reached, or failed (a 5xx),
 * is a plain Error: asking again later may work.
 */
export class HfTokenRejectedError extends Error {
	constructor(message: string) {
		super(message);
		this.name = HF_TOKEN_REJECTED_ERROR_NAME;
	}
}
