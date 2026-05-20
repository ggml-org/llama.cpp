<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Table from '$lib/components/ui/table';
	import { BadgesModality, ActionIconCopyToClipboard } from '$lib/components/app';
	import { serverStore } from '$lib/stores/server.svelte';
	import {
		modelsStore,
		modelOptions,
		modelsLoading,
		routerModels
	} from '$lib/stores/models.svelte';
	import { formatFileSize, formatParameters, formatNumber } from '$lib/utils';
	import type { ApiLlamaCppServerProps } from '$lib/types';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		// when set, fetch props from the child process (router mode)
		modelId?: string | null;
	}

	let { open = $bindable(), onOpenChange, modelId = null }: Props = $props();

	let isRouter = $derived(serverStore.isRouterMode);

	// per-model props fetched from the child process
	let routerModelProps = $state<ApiLlamaCppServerProps | null>(null);
	let isLoadingRouterProps = $state(false);

	// in router mode use per-model props, otherwise use global props
	let serverProps = $derived(isRouter && modelId ? routerModelProps : serverStore.props);

	let modelName = $derived(isRouter && modelId ? modelId : modelsStore.singleModelName);
	let models = $derived(modelOptions());
	let isLoadingModels = $derived(modelsLoading());

	// in router mode, find the model option matching modelId
	// in single mode, use the first model as before
	let firstModel = $derived.by(() => {
		if (isRouter && modelId) {
			return models.find((m) => m.model === modelId) ?? null;
		}
		return models[0] ?? null;
	});

	// Get modalities from modelStore using the model ID from the first model
	let modalities = $derived.by(() => {
		if (!firstModel?.id) return [];
		return modelsStore.getModelModalitiesArray(firstModel.id);
	});

	// Look up the raw router entry to access kind/remote/upstream fields,
	// which aren't carried on ModelOption.
	let routerEntry = $derived.by(() => {
		const id = firstModel?.id ?? modelId;
		if (!id) return null;
		return routerModels().find((m) => m.id === id) ?? null;
	});
	let isRemoteEntry = $derived(routerEntry?.kind === 'remote');
	let upstreamMeta = $derived(routerEntry?.upstream ?? null);
	let upstreamUrl = $derived(routerEntry?.remote?.url ?? '');
	let upstreamModelId = $derived(routerEntry?.remote?.model ?? '');
	let upstreamModelsUrl = $derived(
		upstreamUrl ? `${upstreamUrl.replace(/\/$/, '')}/v1/models` : ''
	);
	// Display helpers — render whatever scalar fields the upstream returned.
	function formatUpstreamValue(value: unknown): string {
		if (value === null || value === undefined) return '';
		if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
			return String(value);
		}
		try {
			return JSON.stringify(value);
		} catch {
			return '';
		}
	}
	function humanizeUpstreamKey(key: string): string {
		return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
	}
	// Hide keys that duplicate what the router-level row already shows or that
	// add noise (permissions are huge nested arrays in vLLM).
	const UPSTREAM_KEY_SKIP = new Set(['id', 'object', 'permission']);
	let upstreamRows = $derived.by(() => {
		if (!upstreamMeta) return [] as Array<{ key: string; label: string; value: string }>;
		return Object.entries(upstreamMeta)
			.filter(([k]) => !UPSTREAM_KEY_SKIP.has(k))
			.map(([k, v]) => ({ key: k, label: humanizeUpstreamKey(k), value: formatUpstreamValue(v) }))
			.filter((row) => row.value !== '');
	});

	// Ensure models are fetched when dialog opens
	$effect(() => {
		if (open && models.length === 0) {
			modelsStore.fetch();
		}
	});

	// fetch per-model props from child process when dialog opens in router mode.
	// Remote entries have no child process, so skip — we render upstream metadata instead.
	$effect(() => {
		if (open && isRouter && modelId && !isRemoteEntry) {
			isLoadingRouterProps = true;
			modelsStore
				.fetchModelProps(modelId)
				.then((props) => {
					routerModelProps = props;
				})
				.catch(() => {
					routerModelProps = null;
				})
				.finally(() => {
					isLoadingRouterProps = false;
				});
		}
		if (!open) {
			routerModelProps = null;
		}
	});
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="@container z-9999 !max-h-[80dvh] !max-w-[60rem] max-w-full">
		<style>
			@container (max-width: 56rem) {
				.resizable-text-container {
					max-width: calc(100vw - var(--threshold));
				}
			}
		</style>

		<Dialog.Header>
			<Dialog.Title>Model Information</Dialog.Title>

			<Dialog.Description>Current model details and capabilities</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-6 py-4">
			{#if isLoadingModels || isLoadingRouterProps}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">Loading model information...</div>
				</div>
			{:else if isRemoteEntry && firstModel}
				<!-- Remote (proxy) entries: render upstream metadata captured at router boot, -->
				<!-- with explicit N/A + links when the upstream was unreachable.              -->
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head class="w-[10rem]">Remote Model</Table.Head>
							<Table.Head>
								<div class="inline-flex items-center gap-2">
									<span
										class="resizable-text-container min-w-0 flex-1 truncate"
										style:--threshold="12rem"
									>
										{modelName}
									</span>

									<ActionIconCopyToClipboard
										text={modelName || ''}
										canCopy={!!modelName}
										ariaLabel="Copy model name to clipboard"
									/>
								</div>
							</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						<Table.Row>
							<Table.Cell class="h-10 align-middle font-medium">Upstream URL</Table.Cell>
							<Table.Cell class="h-10 align-middle font-mono text-xs">
								{#if upstreamUrl}
									<a
										href={upstreamModelsUrl}
										target="_blank"
										rel="noreferrer noopener"
										class="text-sky-600 underline-offset-2 hover:underline"
									>
										{upstreamUrl}
									</a>
								{:else}
									<span class="text-muted-foreground">N/A</span>
								{/if}
							</Table.Cell>
						</Table.Row>

						{#if upstreamModelId}
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">Upstream Model ID</Table.Cell>
								<Table.Cell class="h-10 align-middle font-mono text-xs">
									{upstreamModelId}
								</Table.Cell>
							</Table.Row>
						{/if}

						{#if upstreamRows.length > 0}
							{#each upstreamRows as row (row.key)}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">{row.label}</Table.Cell>
									<Table.Cell class="h-10 align-middle font-mono text-xs">
										{row.value}
									</Table.Cell>
								</Table.Row>
							{/each}
						{:else}
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium text-muted-foreground">
									Upstream metadata
								</Table.Cell>
								<Table.Cell class="h-10 align-middle text-xs text-muted-foreground">
									N/A — upstream was unreachable when the router started. Try:
									{#if upstreamUrl}
										<a
											href={upstreamModelsUrl}
											target="_blank"
											rel="noreferrer noopener"
											class="ml-1 text-sky-600 underline-offset-2 hover:underline"
										>
											{upstreamModelsUrl}
										</a>
									{/if}
								</Table.Cell>
							</Table.Row>
						{/if}

						{#if modalities.length > 0}
							<Table.Row>
								<Table.Cell class="align-middle font-medium">Modalities</Table.Cell>
								<Table.Cell>
									<div class="flex flex-wrap gap-1">
										<BadgesModality {modalities} />
									</div>
								</Table.Cell>
							</Table.Row>
						{/if}
					</Table.Body>
				</Table.Root>
			{:else if firstModel}
				{@const modelMeta = firstModel.meta}

				{#if serverProps}
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head class="w-[10rem]">Model</Table.Head>

								<Table.Head>
									<div class="inline-flex items-center gap-2">
										<span
											class="resizable-text-container min-w-0 flex-1 truncate"
											style:--threshold="12rem"
										>
											{modelName}
										</span>

										<ActionIconCopyToClipboard
											text={modelName || ''}
											canCopy={!!modelName}
											ariaLabel="Copy model name to clipboard"
										/>
									</div>
								</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							<!-- Model Path -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">File Path</Table.Cell>

								<Table.Cell
									class="inline-flex h-10 items-center gap-2 align-middle font-mono text-xs"
								>
									<span
										class="resizable-text-container min-w-0 flex-1 truncate"
										style:--threshold="14rem"
									>
										{serverProps.model_path}
									</span>

									<ActionIconCopyToClipboard
										text={serverProps.model_path}
										ariaLabel="Copy model path to clipboard"
									/>
								</Table.Cell>
							</Table.Row>

							<!-- Context Size -->
							{#if serverProps?.default_generation_settings?.n_ctx}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Context Size</Table.Cell>

									<Table.Cell
										>{formatNumber(serverProps.default_generation_settings.n_ctx)} tokens</Table.Cell
									>
								</Table.Row>
							{:else}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium text-red-500"
										>Context Size</Table.Cell
									>

									<Table.Cell class="text-red-500">Not available</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Training Context -->
							{#if modelMeta?.n_ctx_train}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Training Context</Table.Cell>

									<Table.Cell>{formatNumber(modelMeta.n_ctx_train)} tokens</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Model Size -->
							{#if modelMeta?.size}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Model Size</Table.Cell>

									<Table.Cell>{formatFileSize(modelMeta.size)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Parameters -->
							{#if modelMeta?.n_params}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">Parameters</Table.Cell>

									<Table.Cell>{formatParameters(modelMeta.n_params)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Embedding Size -->
							{#if modelMeta?.n_embd}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Embedding Size</Table.Cell>

									<Table.Cell>{formatNumber(modelMeta.n_embd)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Vocabulary Size -->
							{#if modelMeta?.n_vocab}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Vocabulary Size</Table.Cell>

									<Table.Cell>{formatNumber(modelMeta.n_vocab)} tokens</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Vocabulary Type -->
							{#if modelMeta?.vocab_type}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Vocabulary Type</Table.Cell>
									<Table.Cell class="align-middle capitalize">{modelMeta.vocab_type}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Total Slots -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">Parallel Slots</Table.Cell>

								<Table.Cell>{serverProps.total_slots}</Table.Cell>
							</Table.Row>

							<!-- Modalities -->
							{#if modalities.length > 0}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Modalities</Table.Cell>

									<Table.Cell>
										<div class="flex flex-wrap gap-1">
											<BadgesModality {modalities} />
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Build Info -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">Build Info</Table.Cell>

								<Table.Cell class="align-middle font-mono text-xs"
									>{serverProps.build_info}</Table.Cell
								>
							</Table.Row>

							<!-- Chat Template -->
							{#if serverProps.chat_template}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">Chat Template</Table.Cell>

									<Table.Cell class="py-10">
										<div class="rounded-md bg-muted p-4">
											<pre
												class="font-mono text-xs whitespace-pre-wrap">{serverProps.chat_template}</pre>
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}
						</Table.Body>
					</Table.Root>
				{/if}
			{:else if !isLoadingModels}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">No model information available</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
