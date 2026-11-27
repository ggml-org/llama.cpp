/**
 * What Llama Chat tells Hugging Face about itself when it signs in, which it does to share a
 * conversation as an agent trace.
 */

/**
 * What the sign-in is asked for: read the account, and create datasets and commit traces to
 * the ones the app created. No scope reaches any other repository.
 */
export const HUGGING_FACE_SCOPES = ['openid', 'profile', 'contribute-repos'] as const;
/** The scope a share cannot do without: creating a dataset and committing a trace. */
export const HUGGING_FACE_REQUIRED_SCOPE = 'contribute-repos';
/** The name the app registers under, shown on the Hugging Face authorization page. */
export const HUGGING_FACE_CLIENT_NAME = 'Llama Chat';
/** The app's icon, shown for it on Hugging Face (@ggml-org/llama-brand, from npm's CDN). */
export const HUGGING_FACE_LOGO_URL =
	'https://cdn.jsdelivr.net/npm/@ggml-org/llama-brand@0.1.0/logo/svg/app/web/maskable-icon-512x512.svg';
/** The dataset a first share suggests, under the account's name. */
export const HUGGING_FACE_DEFAULT_DATASET_NAME = 'llama-chat-traces';
/** How long a finished sign-in popup waits to close itself before saying it can be closed. */
export const HUGGING_FACE_POPUP_CLOSE_GRACE_MS = 400;

/** What the share dialog and the sign-in say for the app: they are about a conversation. */
export const CONVERSATION_SHARE_TEXT = {
	commitMessage: (name: string) => `Add trace: ${name}`,
	CONVERSATION_NOT_FOUND: 'Conversation not found.',
	LINK_COPIED: 'Link copied',
	messageCount: (count: number) => (count === 1 ? '1 message' : `${count} messages`),
	SCOPE_ALL: 'Whole conversation',
	SCOPE_UP_TO_MESSAGE: 'Up to this message',
	SHARED: 'Conversation shared to Hugging Face',
	subject: (name: string) => `"${name}"`,
	SUBJECT_FALLBACK: 'this conversation'
} as const;

export const HUGGING_FACE_SIGN_IN_TEXT = {
	CLOSE_WINDOW: 'You can close this window.',
	FAILED: 'The sign-in failed.',
	SIGNED_IN: 'Signed in to Hugging Face'
} as const;
