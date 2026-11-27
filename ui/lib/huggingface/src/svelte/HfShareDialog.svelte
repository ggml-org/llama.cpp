<script lang="ts">
	import type { HfAccount } from './account.svelte.js';
	import HfAccountPanel from './HfAccountPanel.svelte';
	import HfChoiceTile from './HfChoiceTile.svelte';
	import HfTracePreview from './HfTracePreview.svelte';
	import {
		HF_DATASET_TILES_SHOWN,
		HF_NEW_DATASET_VALUE,
		HF_REPO_ID_REGEX,
		HF_REPO_ID_SEPARATOR,
		HF_TRACES_DOCS_URL
	} from '../constants.js';
	import type { HfShareScope, HfTraceFile, HfTracePreviewLine } from '../types.js';
	import { errorMessage, splitRepoId } from '../util.js';
	import * as AlertDialog from '@ggml-org/llama-ui/ui/alert-dialog';
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import { Input } from '@ggml-org/llama-ui/ui/input';
	import { Label } from '@ggml-org/llama-ui/ui/label';
	import * as RadioGroup from '@ggml-org/llama-ui/ui/radio-group';
	import { Skeleton } from '@ggml-org/llama-ui/ui/skeleton';
	import {
		CircleCheck,
		Copy,
		Database,
		ExternalLink,
		Globe,
		Loader2,
		Lock,
		Plus,
		Share,
		TriangleAlert
	} from '@lucide/svelte';
	import { tick, untrack } from 'svelte';

	interface Props {
		account: HfAccount;
		open: boolean;
		/** What is shared, as the description names it, e.g. `"My chat"` or `this conversation`. */
		subject: string;
		/** The parts that can be shared; the choice shows when there is more than one. */
		scopes: HfShareScope[];
		/** The scope picked when the dialog opens; the first one otherwise. */
		initialScope?: string;
		/** Builds the trace of a scope, with or without its attachments. */
		buildTrace: (scope: string, includeAttachments: boolean) => Promise<HfTraceFile>;
		onShared?: (url: string) => void;
		onCopyLink: (url: string) => void;
	}

	let {
		account,
		buildTrace,
		initialScope,
		onCopyLink,
		onShared,
		open = $bindable(),
		scopes,
		subject
	}: Props = $props();

	const DATASET_LABEL_ID = 'hf-dataset-label';

	let scopeId = $state('');
	/** A listed dataset's ID, or HF_NEW_DATASET_VALUE to name a new one. */
	let datasetChoice = $state('');
	let newDataset = $state('');
	let newDatasetInput = $state<HTMLInputElement | null>(null);
	let isShowingAllDatasets = $state(false);
	let isPrivate = $state(false);
	let includeAttachments = $state(false);
	let isSharing = $state(false);
	let shareError = $state<string | null>(null);
	let sharedUrl = $state<string | null>(null);
	/** The file a finished share added, and the messages its preview shows. */
	let sharedFile = $state<{ path: string; preview?: HfTracePreviewLine[] } | null>(null);

	// a fresh form every time the dialog opens
	$effect.pre(() => {
		if (!open) return;

		untrack(() => {
			scopeId = initialScope ?? scopes[0]?.id ?? '';
			datasetChoice = '';
			newDataset = '';
			isShowingAllDatasets = false;
			isPrivate = false;
			includeAttachments = false;
			shareError = null;
			sharedUrl = null;
			sharedFile = null;

			if (!account.isSigningInWithHf) account.oauthError = null;

			// the list is fetched again on every open, so datasets made elsewhere show up
			void account.restoreSession().then(() => account.loadDatasets());
		});
	});

	let scope = $derived(scopes.find((entry) => entry.id === scopeId) ?? scopes[0]);

	let user = $derived(account.user);
	let datasets = $derived(account.datasets);
	// the datasets this account can add to, shown as tiles; the rest of its datasets stay out of it
	let writableDatasets = $derived(account.writableDatasets);
	let tiles = $derived(
		isShowingAllDatasets ? writableDatasets : writableDatasets.slice(0, HF_DATASET_TILES_SHOWN)
	);
	let isListingDatasets = $derived(!account.hasListedDatasets || account.isLoadingDatasets);
	let isNewDataset = $derived(datasetChoice === HF_NEW_DATASET_VALUE);
	let dataset = $derived(isNewDataset ? newDataset.trim() : datasetChoice);
	let pickedDataset = $derived(datasets.find((entry) => entry.id === datasetChoice));

	// a pick that is no longer one the account can add to: another account signed in, or the Hub
	// turned its dataset down, which makes the account forget it
	let isPickStale = $derived(
		!!datasetChoice &&
			datasetChoice !== HF_NEW_DATASET_VALUE &&
			!writableDatasets.some((entry) => entry.id === datasetChoice)
	);

	// once the account's datasets are in, pick the one traces went to last, else the first one it
	// can add to, else offer a new one, and again when the pick goes stale; this also covers a
	// sign-in that finished elsewhere (a popup, another tab)
	$effect(() => {
		if (!open || !user || isListingDatasets || (datasetChoice && !isPickStale)) return;

		untrack(() => {
			datasetChoice = writableDatasets[0]?.id ?? HF_NEW_DATASET_VALUE;
			newDataset = account.newDatasetStart;
		});
	});

	let isDatasetValid = $derived(HF_REPO_ID_REGEX.test(dataset));
	// the owner alone, as the field starts, is not a mistake yet
	let isDatasetNameStarted = $derived(!!dataset && !dataset.endsWith(HF_REPO_ID_SEPARATOR));
	let canShare = $derived(
		!!user && !account.isReadOnly && isDatasetValid && !!scope && !scope.isEmpty && !isSharing
	);

	async function handlePickNew() {
		datasetChoice = HF_NEW_DATASET_VALUE;

		await tick();
		newDatasetInput?.focus();
	}

	function handleOpenChange(isOpen: boolean) {
		// a share on its way keeps the dialog
		if (!isOpen && !isSharing) open = false;
	}

	async function handleShare(event: SubmitEvent) {
		event.preventDefault();

		if (!canShare || !scope) return;

		isSharing = true;
		shareError = null;

		try {
			// the preview is for this dialog: only the file goes to the Hub
			const { preview, ...file } = await buildTrace(
				scope.id,
				!!scope.hasAttachments && includeAttachments
			);

			const url = await account.uploadTrace({
				...file,
				isPrivate: isNewDataset ? isPrivate : !!pickedDataset?.isPrivate,
				repoId: dataset
			});

			sharedFile = { path: file.path, preview };
			sharedUrl = url;
			onShared?.(url);
		} catch (error) {
			shareError = errorMessage(error);
		} finally {
			isSharing = false;
		}
	}
</script>

<AlertDialog.Root onOpenChange={handleOpenChange} {open}>
	<AlertDialog.Content class="sm:max-w-lg">
		<AlertDialog.Header>
			{#if sharedUrl}
				<AlertDialog.Title class="flex items-center gap-2">
					<CircleCheck class="size-5 text-fg-success" />

					Shared to Hugging Face
				</AlertDialog.Title>

				<AlertDialog.Description>
					Added to <span class="font-medium text-foreground">{dataset}</span> as an agent trace, which
					the Hub shows in its trace viewer.
				</AlertDialog.Description>
			{:else}
				<AlertDialog.Title class="flex items-center gap-2">
					<Share class="h-5 w-5" />

					Share to Hugging Face
				</AlertDialog.Title>

				<AlertDialog.Description>
					Upload {subject} to a Hugging Face dataset as an
					<a class="underline" href={HF_TRACES_DOCS_URL} rel="noopener noreferrer" target="_blank">
						agent trace
					</a>, which the Hub shows in its trace viewer.
				</AlertDialog.Description>
			{/if}
		</AlertDialog.Header>

		{#if sharedUrl}
			{@const url = sharedUrl}

			{#if sharedFile}
				<HfTracePreview lines={sharedFile.preview} path={sharedFile.path} />
			{/if}

			<!-- every action in one place, at the bottom right: the main one last -->
			<AlertDialog.Footer class="max-sm:flex-col-reverse">
				<Button onclick={() => (open = false)} variant="ghost">Done</Button>

				<Button onclick={() => onCopyLink(url)} variant="outline">
					<Copy />
					Copy link
				</Button>

				<Button href={url} rel="noopener noreferrer" target="_blank">
					<ExternalLink />
					Open on Hugging Face
				</Button>
			</AlertDialog.Footer>
		{:else if !user}
			<HfAccountPanel {account} />

			<AlertDialog.Footer>
				<AlertDialog.Cancel type="button">Cancel</AlertDialog.Cancel>
			</AlertDialog.Footer>
		{:else}
			<form class="contents" onsubmit={handleShare}>
				<div class="flex flex-col gap-4">
					<HfAccountPanel {account} disabled={isSharing} />

					{#if scopes.length > 1}
						<RadioGroup.Root
							class="gap-2"
							onValueChange={(value) => (scopeId = value)}
							value={scopeId}
						>
							{#each scopes as entry (entry.id)}
								<div class="flex items-center gap-2">
									<RadioGroup.Item id="hf-share-scope-{entry.id}" value={entry.id} />

									<Label
										class="cursor-pointer text-body-sm font-normal"
										for="hf-share-scope-{entry.id}"
									>
										{entry.label}
										{#if entry.detail}<span class="text-muted-foreground">({entry.detail})</span
											>{/if}
									</Label>
								</div>
							{/each}
						</RadioGroup.Root>
					{/if}

					<div class="flex flex-col gap-2">
						<Label id={DATASET_LABEL_ID}>
							{!isListingDatasets && !writableDatasets.length ? 'Create a dataset' : 'Dataset'}
						</Label>

						{#if isListingDatasets}
							<div class="grid grid-cols-2 gap-2">
								<Skeleton class="h-14 rounded-xl" />

								<Skeleton class="h-14 rounded-xl" />
							</div>
						{:else if writableDatasets.length}
							<!-- the datasets it can add to, and a tile to make a new one -->
							<div
								class={[
									'grid grid-cols-2 gap-2',
									isShowingAllDatasets && 'max-h-56 overflow-y-auto'
								]}
								aria-labelledby={DATASET_LABEL_ID}
								role="group"
							>
								{#each tiles as entry (entry.id)}
									<HfChoiceTile
										disabled={isSharing}
										icon={Database}
										isPicked={datasetChoice === entry.id}
										label={splitRepoId(entry.id)[1]}
										onpick={() => (datasetChoice = entry.id)}
										title={entry.id}
									>
										{#if entry.isPrivate}
											<Lock class="size-icon-sm shrink-0" />
											Private
										{:else}
											<Globe class="size-icon-sm shrink-0" />
											Public
										{/if}
									</HfChoiceTile>
								{/each}

								<HfChoiceTile
									disabled={isSharing}
									icon={Plus}
									isPicked={isNewDataset}
									label="New dataset"
									onpick={handlePickNew}
								>
									<span class="truncate">A fresh one</span>
								</HfChoiceTile>
							</div>

							{#if writableDatasets.length > tiles.length}
								<button
									class="self-start text-caption text-muted-foreground underline"
									onclick={() => (isShowingAllDatasets = true)}
									type="button"
								>
									Show {writableDatasets.length - tiles.length} more
								</button>
							{/if}
						{/if}

						{#if isNewDataset}
							<Input
								bind:ref={newDatasetInput}
								bind:value={newDataset}
								aria-label="New dataset name"
								autocomplete="off"
								disabled={isSharing}
								id="hf-dataset"
								placeholder={account.newDatasetStart}
							/>
						{/if}

						<p class="text-caption text-muted-foreground">
							{#if isNewDataset && isDatasetNameStarted && !isDatasetValid}
								<span class="text-destructive">Use the form owner/name.</span>
							{:else if isNewDataset && datasets.some((entry) => entry.id === dataset)}
								{dataset} already exists: the trace goes to it.
							{:else if isNewDataset}
								Created on Hugging Face when you share. Each share adds one file to it.
							{:else if !isListingDatasets}
								Each share adds one file to it.
							{/if}
						</p>
					</div>

					<div class="flex flex-col gap-3">
						{#if isNewDataset}
							<div class="flex items-center gap-2">
								<Checkbox bind:checked={isPrivate} disabled={isSharing} id="hf-private" />

								<Label class="cursor-pointer text-body-sm font-normal" for="hf-private">
									Make the new dataset private
								</Label>
							</div>
						{/if}

						{#if scope?.hasAttachments}
							<div class="flex items-center gap-2">
								<Checkbox
									bind:checked={includeAttachments}
									disabled={isSharing}
									id="hf-attachments"
								/>

								<Label class="cursor-pointer text-body-sm font-normal" for="hf-attachments">
									Include attachments
								</Label>
							</div>
						{/if}
					</div>

					<p class="flex gap-2 text-caption text-muted-foreground">
						<TriangleAlert class="h-4 w-4 shrink-0" />
						Traces can hold personal data, file paths, tool output and secrets. Review the conversation
						before you share it to a public dataset.
					</p>

					{#if shareError}
						<p class="text-caption text-destructive">{shareError}</p>
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
