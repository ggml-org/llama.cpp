<script lang="ts">
	import type { HfAccount } from './account.svelte.js';
	import HfLogo from './HfLogo.svelte';
	import {
		HF_NEW_TOKEN_URL,
		HF_OAUTH_POPUP_BLANK_URL,
		HF_OAUTH_POPUP_FEATURES,
		HF_OAUTH_POPUP_NAME
	} from '../constants.js';
	import { errorMessage } from '../util.js';
	import { ModelOrgAvatar } from '@ggml-org/llama-ui/models';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import { Loader2 } from '@lucide/svelte';

	interface Props {
		account: HfAccount;
		/** Locks the account while something uses it, e.g. a share on its way. */
		disabled?: boolean;
	}

	let { account, disabled = false }: Props = $props();

	let token = $state('');
	let tokenError = $state<string | null>(null);

	let user = $derived(account.user);
	let isSigningInWithHf = $derived(account.isSigningInWithHf);
	let isBusy = $derived(disabled || account.isCheckingToken || isSigningInWithHf);

	function handleSignInWithHf() {
		tokenError = null;

		// opened in the click itself: a popup opened after an await is blocked
		const popup = window.open(
			HF_OAUTH_POPUP_BLANK_URL,
			HF_OAUTH_POPUP_NAME,
			HF_OAUTH_POPUP_FEATURES
		);

		void account.signInWithHuggingFace(popup);
	}

	async function handleSignInWithToken(event: SubmitEvent) {
		event.preventDefault();

		if (!token.trim()) return;

		tokenError = null;

		try {
			await account.signIn(token);

			token = '';
		} catch (error) {
			tokenError = errorMessage(error);
		}
	}
</script>

{#if user}
	<div class="flex flex-col gap-2">
		<div class="flex items-center gap-2 text-body-sm">
			<ModelOrgAvatar org={user.name} size="size-6" />

			<span class="min-w-0 flex-1 truncate">
				Signed in as <span class="font-medium">{user.name}</span>
			</span>

			<Button {disabled} onclick={() => account.signOut()} size="sm" variant="ghost">
				Sign out
			</Button>
		</div>

		{#if account.isReadOnly}
			<p class="text-caption text-destructive">
				This token can only read. <a
					class="underline"
					href={HF_NEW_TOKEN_URL}
					rel="noopener noreferrer"
					target="_blank">Create one with write access</a
				>, then sign out and in again.
			</p>
		{/if}
	</div>
{:else}
	<form class="flex flex-col gap-4" onsubmit={handleSignInWithToken}>
		<div class="flex flex-col gap-2">
			<Button disabled={isBusy} onclick={handleSignInWithHf} type="button" variant="outline">
				{#if isSigningInWithHf}
					<Loader2 class="h-4 w-4 animate-spin" />
				{:else}
					<HfLogo />
				{/if}
				Sign in with Hugging Face
			</Button>

			{#if isSigningInWithHf}
				<p class="text-caption text-muted-foreground">
					Finish signing in on the Hugging Face page.
					<button class="underline" onclick={() => account.cancelSignInWithHf()} type="button">
						Cancel
					</button>
				</p>
			{:else if account.oauthError}
				<p class="text-caption text-destructive">{account.oauthError}</p>
			{/if}
		</div>

		<div class="flex items-center gap-3 text-caption text-muted-foreground">
			<span class="h-px flex-1 bg-border"></span>
			or use an access token
			<span class="h-px flex-1 bg-border"></span>
		</div>

		<div class="flex flex-col gap-2">
			<Label for="hf-token">Hugging Face access token</Label>

			<div class="flex gap-2">
				<Input
					bind:value={token}
					autocomplete="off"
					class="flex-1"
					disabled={isBusy}
					id="hf-token"
					placeholder="hf_..."
					type="password"
				/>

				<Button disabled={!token.trim() || isBusy} type="submit">
					{#if account.isCheckingToken}
						<Loader2 class="h-4 w-4 animate-spin" />
					{/if}
					Sign in
				</Button>
			</div>

			<p class="text-caption text-muted-foreground">
				<a class="underline" href={HF_NEW_TOKEN_URL} rel="noopener noreferrer" target="_blank">
					Create a token with write access
				</a>. It stays in this browser and goes to huggingface.co.
			</p>

			{#if tokenError}
				<p class="text-caption text-destructive">{tokenError}</p>
			{/if}
		</div>
	</form>
{/if}
