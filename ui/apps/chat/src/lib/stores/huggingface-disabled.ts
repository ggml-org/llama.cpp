// What `@ggml-org/llama-huggingface/svelte` resolves to in a build without sharing to
// Hugging Face (a build not made with VITE_PUBLIC_HUGGING_FACE=true, see vite.config.ts):
// an account that is never signed in and components that draw nothing, so the sign-in, the
// upload and the dialogs are not in the bundle. The app reads HUGGING_FACE_ENABLED before it
// shows or starts any of them.

import type { HfAccount as HfAccountState } from '@ggml-org/llama-huggingface/svelte';

type HfAccountApp = Pick<HfAccountState, 'completeSignInFromLocation' | 'restoreSession' | 'user'>;

export class HfAccount implements HfAccountApp {
	user = null;

	async completeSignInFromLocation(): Promise<null> {
		return null;
	}

	async restoreSession(): Promise<void> {}
}

/** A component that draws nothing: a Svelte component is a function of its anchor and props. */
const NO_COMPONENT = (): void => {};

export const HfAccountPanel = NO_COMPONENT;
export const HfLogo = NO_COMPONENT;
export const HfShareDialog = NO_COMPONENT;
