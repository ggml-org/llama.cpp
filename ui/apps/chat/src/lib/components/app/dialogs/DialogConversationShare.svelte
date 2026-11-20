<script lang="ts">
	import { ModelOrgAvatar } from '@ggml-org/llama-ui/models';
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as RadioGroup from '@ggml-org/llama-ui/ui/radio-group';
	import { Check, Copy, ExternalLink, Loader2, Share2, TriangleAlert } from '@lucide/svelte';
	import { HF_NEW_TOKEN_URL, HF_REPO_ID_REGEX, HF_TRACES_DOCS_URL } from '#lib/constants/index.ts';
	import { TraceShareScope } from '#lib/enums/index.ts';
	import { conversationsStore, huggingFaceStore, uiStore } from '#lib/stores/index.ts';
	import { copyToClipboard, filterByLeafNodeId } from '#lib/utils/index.ts';
	import { untrack } from 'svelte';
	import { toast } from 'svelte-sonner';

	let options = $derived(uiStore.conversationShare);
	let conversation = $derived(
		conversationsStore.conversations.find((conv) => conv.id === options?.conversationId)
	);

	let messages = $state<DatabaseMessage[]>([]);
	let token = $state('');
	let signInError = $state<string | null>(null);
	let scope = $state<TraceShareScope>(TraceShareScope.ALL);
	let dataset = $state('');
	let isPrivate = $state(false);
	let includeAttachments = $state(false);
	let isSharing = $state(false);
	let shareError = $state<string | null>(null);
	let sharedUrl = $state<string | null>(null);

	// a fresh form every time the dialog opens
	$effect.pre(() => {
		if (!options) return;

		const { conversationId, messageId } = options;

		untrack(() => {
			token = '';
			signInError = null;
			scope = messageId ? TraceShareScope.UP_TO_MESSAGE : TraceShareScope.ALL;
			dataset = huggingFaceStore.suggestedDataset;
			isPrivate = false;
			includeAttachments = false;
			shareError = null;
			sharedUrl = null;
			messages = [];
		});

		void huggingFaceStore.restoreSession().then(() => {
			if (!dataset) dataset = huggingFaceStore.suggestedDataset;
		});

		void conversationsStore.getConversationMessages(conversationId).then((loaded) => {
			if (uiStore.conversationShare?.conversationId === conversationId) messages = loaded;
		});
	});

	let leafId = $derived(
		scope === TraceShareScope.UP_TO_MESSAGE && options?.messageId
			? options.messageId
			: (conversation?.currNode ?? '')
	);
	let tracePath = $derived(messages.length ? filterByLeafNodeId(messages, leafId) : []);
	let wholeCount = $derived(
		messages.length ? filterByLeafNodeId(messages, conversation?.currNode ?? '').length : 0
	);
	let upToCount = $derived(
		messages.length && options?.messageId
			? filterByLeafNodeId(messages, options.messageId).length
			: 0
	);
	let hasAttachments = $derived(tracePath.some((message) => !!message.extra?.length));

	let user = $derived(huggingFaceStore.user);
	let isReadOnlyToken = $derived(user?.tokenRole === 'read');
	let isDatasetValid = $derived(HF_REPO_ID_REGEX.test(dataset.trim()));
	let canShare = $derived(
		!!user && !isReadOnlyToken && isDatasetValid && tracePath.length > 0 && !isSharing
	);

	function close() {
		if (isSharing) return;

		uiStore.conversationShare = null;
	}

	async function handleSignIn(event: SubmitEvent) {
		event.preventDefault();

		if (!token.trim()) return;

		signInError = null;

		try {
			await huggingFaceStore.signIn(token);

			token = '';

			if (!dataset) dataset = huggingFaceStore.suggestedDataset;
		} catch (error) {
			signInError = error instanceof Error ? error.message : String(error);
		}
	}

	function handleSignOut() {
		huggingFaceStore.signOut();
	}

	async function handleShare(event: SubmitEvent) {
		event.preventDefault();

		if (!canShare || !options) return;

		isSharing = true;
		shareError = null;

		try {
			sharedUrl = await huggingFaceStore.shareConversation({
				conversationId: options.conversationId,
				includeAttachments: hasAttachments && includeAttachments,
				isPrivate,
				repoId: dataset.trim(),
				upToMessageId: scope === TraceShareScope.UP_TO_MESSAGE ? options.messageId : undefined
			});
			toast.success('Conversation shared to Hugging Face');
		} catch (error) {
			shareError = error instanceof Error ? error.message : String(error);
		} finally {
			isSharing = false;
		}
	}

	async function handleCopyLink() {
		if (sharedUrl) await copyToClipboard(sharedUrl, 'Link copied');
	}

	function messageCountLabel(count: number): string {
		return count === 1 ? '1 message' : `${count} messages`;
	}
</script>

<AlertDialog.Root onOpenChange={(open) => !open && close()} open={options !== null}>
	<AlertDialog.Content class="sm:max-w-lg">
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<Share2 class="h-5 w-5" />

				Share to Hugging Face
			</AlertDialog.Title>

			<AlertDialog.Description>
				Upload {conversation ? `"${conversation.name}"` : 'this conversation'} to a Hugging Face dataset
				as an
				<a class="underline" href={HF_TRACES_DOCS_URL} rel="noopener noreferrer" target="_blank">
					agent trace
				</a>, which the Hub shows in its trace viewer.
			</AlertDialog.Description>
		</AlertDialog.Header>

		{#if sharedUrl}
			<div class="flex flex-col gap-3">
				<p class="flex items-center gap-2 text-sm">
					<Check class="h-4 w-4 text-primary" />
					Shared to <span class="font-medium">{dataset.trim()}</span>
				</p>

				<div class="flex gap-2">
					<Button class="flex-1" href={sharedUrl} rel="noopener noreferrer" target="_blank">
						<ExternalLink class="h-4 w-4" />
						Open on Hugging Face
					</Button>

					<Button onclick={handleCopyLink} variant="outline">
						<Copy class="h-4 w-4" />
						Copy link
					</Button>
				</div>
			</div>

			<AlertDialog.Footer>
				<AlertDialog.Cancel type="button">Done</AlertDialog.Cancel>
			</AlertDialog.Footer>
		{:else if !user}
			<form class="contents" onsubmit={handleSignIn}>
				<div class="flex flex-col gap-2">
					<Label for="hf-token">Hugging Face access token</Label>

					<Input
						bind:value={token}
						autocomplete="off"
						disabled={huggingFaceStore.isCheckingToken}
						id="hf-token"
						placeholder="hf_..."
						type="password"
					/>

					<p class="text-xs text-muted-foreground">
						<a class="underline" href={HF_NEW_TOKEN_URL} rel="noopener noreferrer" target="_blank">
							Create a token with write access
						</a>. It stays in this browser and is sent only to huggingface.co.
					</p>

					{#if signInError}
						<p class="text-xs text-destructive">{signInError}</p>
					{/if}
				</div>

				<AlertDialog.Footer>
					<AlertDialog.Cancel type="button">Cancel</AlertDialog.Cancel>

					<Button disabled={!token.trim() || huggingFaceStore.isCheckingToken} type="submit">
						{#if huggingFaceStore.isCheckingToken}
							<Loader2 class="h-4 w-4 animate-spin" />
						{/if}
						Sign in
					</Button>
				</AlertDialog.Footer>
			</form>
		{:else}
			<form class="contents" onsubmit={handleShare}>
				<div class="flex flex-col gap-4">
					<div class="flex items-center gap-2 text-sm">
						<ModelOrgAvatar org={user.name} size="size-6" />

						<span class="min-w-0 flex-1 truncate">
							Signed in as <span class="font-medium">{user.name}</span>
						</span>

						<Button disabled={isSharing} onclick={handleSignOut} size="sm" variant="ghost">
							Sign out
						</Button>
					</div>

					{#if isReadOnlyToken}
						<p class="text-xs text-destructive">
							This token can only read. <a
								class="underline"
								href={HF_NEW_TOKEN_URL}
								rel="noopener noreferrer"
								target="_blank">Create one with write access</a
							>, then sign out and in again.
						</p>
					{/if}

					{#if options?.messageId}
						<RadioGroup.Root
							class="gap-2"
							onValueChange={(value) => (scope = value as TraceShareScope)}
							value={scope}
						>
							<div class="flex items-center gap-2">
								<RadioGroup.Item id="share-scope-up-to" value={TraceShareScope.UP_TO_MESSAGE} />

								<Label class="cursor-pointer text-sm font-normal" for="share-scope-up-to">
									Up to this message
									{#if upToCount}<span class="text-muted-foreground"
											>({messageCountLabel(upToCount)})</span
										>{/if}
								</Label>
							</div>

							<div class="flex items-center gap-2">
								<RadioGroup.Item id="share-scope-all" value={TraceShareScope.ALL} />

								<Label class="cursor-pointer text-sm font-normal" for="share-scope-all">
									Whole conversation
									{#if wholeCount}<span class="text-muted-foreground"
											>({messageCountLabel(wholeCount)})</span
										>{/if}
								</Label>
							</div>
						</RadioGroup.Root>
					{/if}

					<div class="flex flex-col gap-2">
						<Label for="hf-dataset">Dataset</Label>

						<Input
							bind:value={dataset}
							autocomplete="off"
							disabled={isSharing}
							id="hf-dataset"
							placeholder="username/llama-chat-traces"
						/>

						<p class="text-xs text-muted-foreground">
							{#if dataset.trim() && !isDatasetValid}
								<span class="text-destructive">Use the form owner/name.</span>
							{:else}
								Created if it does not exist yet. Each share adds one file to it.
							{/if}
						</p>
					</div>

					<div class="flex flex-col gap-3">
						<div class="flex items-center gap-2">
							<Checkbox bind:checked={isPrivate} disabled={isSharing} id="hf-private" />

							<Label class="cursor-pointer text-sm font-normal" for="hf-private">
								Make a new dataset private
							</Label>
						</div>

						{#if hasAttachments}
							<div class="flex items-center gap-2">
								<Checkbox
									bind:checked={includeAttachments}
									disabled={isSharing}
									id="hf-attachments"
								/>

								<Label class="cursor-pointer text-sm font-normal" for="hf-attachments">
									Include attachments
								</Label>
							</div>
						{/if}
					</div>

					<p class="flex gap-2 text-xs text-muted-foreground">
						<TriangleAlert class="h-4 w-4 shrink-0" />
						Traces can hold personal data, file paths, tool output and secrets. Review the conversation
						before you share it to a public dataset.
					</p>

					{#if shareError}
						<p class="text-xs text-destructive">{shareError}</p>
					{/if}
				</div>

				<AlertDialog.Footer>
					<AlertDialog.Cancel disabled={isSharing} type="button">Cancel</AlertDialog.Cancel>

					<Button disabled={!canShare} type="submit">
						{#if isSharing}
							<Loader2 class="h-4 w-4 animate-spin" />
						{/if}
						Share
					</Button>
				</AlertDialog.Footer>
			</form>
		{/if}
	</AlertDialog.Content>
</AlertDialog.Root>
