/**
 * huggingFace - the Hugging Face account Llama Chat is signed in with
 *
 * The account, its sign-in and the upload of a trace live in @ggml-org/llama-huggingface;
 * this file says who the app is. Sharing a conversation is what asks for the sign-in, so it
 * starts in the share dialog, and Settings > General shows the account once there is one.
 */

import { HfAccount } from '@ggml-org/llama-huggingface/svelte';
import { HUGGING_FACE_ENABLED } from '#lib/constants/app.constants.ts';
import {
	HUGGING_FACE_CLIENT_NAME,
	HUGGING_FACE_DEFAULT_DATASET_NAME,
	HUGGING_FACE_LOGO_URL,
	HUGGING_FACE_POPUP_CLOSE_GRACE_MS,
	HUGGING_FACE_REQUIRED_SCOPE,
	HUGGING_FACE_SCOPES,
	HUGGING_FACE_SIGN_IN_TEXT
} from '#lib/constants/huggingface-account.constants.ts';
import { STORAGE_APP_NAME } from '#lib/constants/storage.constants.ts';
import { withBase } from '#lib/utils/base-path.ts';
import { toast } from 'svelte-sonner';

export const huggingFace = new HfAccount({
	callbackUrl: () => `${window.location.origin}${withBase('/')}`,
	clientName: HUGGING_FACE_CLIENT_NAME,
	defaultDatasetName: HUGGING_FACE_DEFAULT_DATASET_NAME,
	logoUrl: HUGGING_FACE_LOGO_URL,
	requiredScope: HUGGING_FACE_REQUIRED_SCOPE,
	scopes: HUGGING_FACE_SCOPES,
	storagePrefix: STORAGE_APP_NAME
});

const { CLOSE_WINDOW, FAILED, SIGNED_IN } = HUGGING_FACE_SIGN_IN_TEXT;

/** Finishes a sign-in when this page load is Hugging Face sending the browser back. */
async function completeSignIn(): Promise<void> {
	const outcome = await huggingFace.completeSignInFromLocation();

	if (!outcome?.result) return;

	const { result } = outcome;

	if (outcome.closeWindow) {
		window.close();

		// a browser may keep a window its script did not open in this document: say so
		setTimeout(() => {
			if (result.ok) toast.success(`${SIGNED_IN} ${CLOSE_WINDOW}`);
			else toast.error(`${result.error ?? FAILED} ${CLOSE_WINDOW}`);
		}, HUGGING_FACE_POPUP_CLOSE_GRACE_MS);

		return;
	}

	if (result.ok) toast.success(SIGNED_IN);
	else toast.error(result.error ?? FAILED);
}

/**
 * Run once when the app starts: finishes a sign-in this page load brings back, and looks up
 * who a remembered sign-in belongs to. Without either, no request leaves the browser. A build
 * without sharing does neither.
 */
export function startHuggingFace(): void {
	if (!HUGGING_FACE_ENABLED) return;

	void completeSignIn();
	void huggingFace.restoreSession();
}
