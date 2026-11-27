# Llama Hugging Face

`@ggml-org/llama-huggingface` gives an app that runs in the browser, with no server of its own, a Hugging Face account: signing in, and sharing a conversation to a dataset as an [agent trace](https://huggingface.co/docs/hub/agent-traces).

It knows no app. The app says who it is, hands over what to share, and decides where the components show.

| Entry                                | What it holds                                                                                                                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `@ggml-org/llama-huggingface`        | Plain TypeScript, no dependencies: "Sign in with Hugging Face" (OAuth with PKCE and dynamic client registration, `HfOAuthSignIn`) and the Hub calls (`whoAmI`, `listDatasets`, `uploadTrace`). |
| `@ggml-org/llama-huggingface/svelte` | The account as Svelte state (`HfAccount`) and the components that show it, built on `@ggml-org/llama-ui`: `HfAccountPanel`, `HfShareDialog`, `HfLogo`.                                         |

## Use

```ts
import { HfAccount } from '@ggml-org/llama-huggingface/svelte';

export const account = new HfAccount({
	callbackUrl: () => new URL('/', window.location.origin).href,
	clientName: 'My app',
	defaultDatasetName: 'my-app-traces',
	logoUrl: 'https://example.com/my-app.png', // optional: shown for the app on Hugging Face
	requiredScope: 'contribute-repos',
	scopes: ['openid', 'profile', 'contribute-repos'],
	storagePrefix: 'my-app'
});

// on every page load: finishes a sign-in when Hugging Face sent the browser back
const outcome = await account.completeSignInFromLocation();

if (outcome?.closeWindow) window.close();
```

```svelte
<script lang="ts">
	import { HfAccountPanel, HfShareDialog } from '@ggml-org/llama-huggingface/svelte';
</script>

<!-- who is signed in, or the two ways to sign in: a button, or a pasted access token -->
<HfAccountPanel {account} />

<HfShareDialog
	bind:open
	{account}
	buildTrace={async (scope, includeAttachments) => ({
		commitMessage: 'Add trace: my chat',
		content: jsonl, // Session Traces Format, one JSON object per line
		path: 'my-chat.jsonl',
		preview: [{ role: 'user', text: 'What is Rust?' }] // optional: shown faded once it is shared
	})}
	onCopyLink={(url) => navigator.clipboard.writeText(url)}
	scopes={[{ id: 'all', label: 'Whole conversation', detail: '12 messages' }]}
	subject={'"My chat"'}
/>
```

- The share dialog offers the datasets a share can go to as tiles, and a tile to create one. An access token that was pasted reaches every dataset of the account; a sign-in with Hugging Face reaches only the ones the app created, which the browser knows as the ones it shared to (`HfAccount.writableDatasets`). With none, the dialog asks for the name of a new one.
- A sign-in opens the Hugging Face authorization page in a popup, or in the tab when the popup is blocked. The app registers itself for its own origin on first use, so it works from any host and port.
- The sign-in and the pasted token are kept in `localStorage`, under the app's `storagePrefix`.
- With `contribute-repos`, a sign-in can write only to the datasets the app created; a pasted access token with write access is not limited that way.

## Development

The package has no build step and is not published yet: apps in this repo compile its sources, and its components are type-checked by the app that uses them.

```bash
npm install
npm run check
npm test
```
