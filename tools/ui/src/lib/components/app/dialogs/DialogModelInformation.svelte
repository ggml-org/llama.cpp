<script lang="ts">
	import { ActionIconCopyToClipboard, BadgesModality } from '$lib/components/app';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Table from '$lib/components/ui/table';
	import { modelsStore, serverStore } from '$lib/stores';
	import type { ApiLlamaCppServerProps } from '$lib/types';
	import { formatFileSize, formatNumber, formatParameters } from '$lib/utils';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		// when set, fetch props from the child process (router mode)
		modelId?: string | null;
	}

	let { modelId = null, onOpenChange, open = $bindable() }: Props = $props();

	let isRouter = $derived(serverStore.isRouterMode);

	// per-model props fetched from the child process
	let routerModelProps = $state<ApiLlamaCppServerProps | null>(null);
	let isLoadingRouterProps = $state(false);

	// in router mode use per-model props, otherwise use global props
	let serverProps = $derived(isRouter && modelId ? routerModelProps : serverStore.props);

	let modelName = $derived(isRouter && modelId ? modelId : modelsStore.singleModelName);
	let models = $derived(modelsStore.models);
	let isLoadingModels = $derived(modelsStore.loading);

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

		return modelsStore.props.getModelModalitiesArray(firstModel.id);
	});

	// Ensure models are fetched when dialog opens
	$effect(() => {
		if (open && models.length === 0) {
			modelsStore.fetch();
		}
	});

	// fetch per-model props from child process when dialog opens in router mode
	$effect(() => {
		if (open && isRouter && modelId) {
			isLoadingRouterProps = true;
			modelsStore.props
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
	<Dialog.Content
		class="z-9999 max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! md:w-[calc(100vw-4rem)]! md:max-w-[60rem]! md:max-h-[80dvh]!"
	>
		<!-- sticky header holds only the close button; the title scrolls with the body -->
		<Dialog.Header />

		<div class="min-w-0 space-y-6 md:py-4 -mt-4! md:mt-0 pb-4">
			<div class="min-w-0 space-y-2">
				<Dialog.Title>Model Information</Dialog.Title>

				<Dialog.Description>Current model details and capabilities</Dialog.Description>
			</div>

			{#if isLoadingModels || isLoadingRouterProps}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">Loading model information...</div>
				</div>
			{:else if firstModel}
				{@const modelMeta = firstModel.meta}

				{#if serverProps}
					<!-- Desktop: fixed-layout table, long values scroll inside their cell -->
					<Table.Root class="hidden table-fixed md:table">
						<Table.Header>
							<Table.Row>
								<Table.Head class="w-[10rem]">Model</Table.Head>

								<Table.Head>
									<div class="flex min-w-0 items-center gap-2">
										<span class="min-w-0 flex-1 overflow-x-auto whitespace-nowrap">
											{modelName}
										</span>

										<ActionIconCopyToClipboard
											ariaLabel="Copy model name to clipboard"
											canCopy={!!modelName}
											text={modelName || ''}
										/>
									</div>
								</Table.Head>
							</Table.Row>
						</Table.Header>

						<Table.Body>
							<!-- Model Path -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">File Path</Table.Cell>

								<Table.Cell class="h-10 align-middle font-mono text-xs">
									<div class="flex min-w-0 items-center gap-2">
										<span class="min-w-0 flex-1 overflow-x-auto whitespace-nowrap">
											{serverProps.model_path}
										</span>

										<ActionIconCopyToClipboard
											ariaLabel="Copy model path to clipboard"
											text={serverProps.model_path}
										/>
									</div>
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
									<Table.Cell class="py-4" colspan={2}>
										<div class="flex flex-col gap-2">
											<span class="font-medium">Chat Template</span>

											<div class="overflow-x-auto rounded-md bg-muted p-4">
												<pre
													class="font-mono text-xs whitespace-pre">{serverProps.chat_template}</pre>
											</div>
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}
						</Table.Body>
					</Table.Root>

					<!-- Mobile: stacked layout; long values wrap instead of scrolling the page -->
					<div class="flex min-w-0 flex-col gap-4 md:hidden">
						<div class="min-w-0 space-y-1">
							<div class="text-xs font-medium text-muted-foreground">Model</div>

							<div class="flex min-w-0 items-start gap-2">
								<span class="min-w-0 flex-1 break-all font-mono text-xs">{modelName}</span>

								<ActionIconCopyToClipboard
									ariaLabel="Copy model name to clipboard"
									canCopy={!!modelName}
									text={modelName || ''}
								/>
							</div>
						</div>

						<div class="min-w-0 space-y-1">
							<div class="text-xs font-medium text-muted-foreground">File Path</div>

							<div class="flex min-w-0 items-start gap-2">
								<span class="min-w-0 flex-1 break-all font-mono text-xs"
									>{serverProps.model_path}</span
								>

								<ActionIconCopyToClipboard
									ariaLabel="Copy model path to clipboard"
									text={serverProps.model_path}
								/>
							</div>
						</div>

						{#if serverProps?.default_generation_settings?.n_ctx}
							{@render infoRow(
								'Context Size',
								`${formatNumber(serverProps.default_generation_settings.n_ctx)} tokens`
							)}
						{:else}
							{@render infoRow('Context Size', 'Not available', 'text-red-500')}
						{/if}

						{#if modelMeta?.n_ctx_train}
							{@render infoRow('Training Context', `${formatNumber(modelMeta.n_ctx_train)} tokens`)}
						{/if}

						{#if modelMeta?.size}
							{@render infoRow('Model Size', formatFileSize(modelMeta.size))}
						{/if}

						{#if modelMeta?.n_params}
							{@render infoRow('Parameters', formatParameters(modelMeta.n_params))}
						{/if}

						{#if modelMeta?.n_embd}
							{@render infoRow('Embedding Size', formatNumber(modelMeta.n_embd))}
						{/if}

						{#if modelMeta?.n_vocab}
							{@render infoRow('Vocabulary Size', `${formatNumber(modelMeta.n_vocab)} tokens`)}
						{/if}

						{#if modelMeta?.vocab_type}
							{@render infoRow('Vocabulary Type', modelMeta.vocab_type, 'capitalize')}
						{/if}

						{@render infoRow('Parallel Slots', `${serverProps.total_slots}`)}

						{#if modalities.length > 0}
							<div class="min-w-0 space-y-1">
								<div class="text-xs font-medium text-muted-foreground">Modalities</div>

								<div class="flex flex-wrap gap-1">
									<BadgesModality {modalities} />
								</div>
							</div>
						{/if}

						<div class="min-w-0 space-y-1">
							<div class="text-xs font-medium text-muted-foreground">Build Info</div>

							<span class="block break-all font-mono text-xs">{serverProps.build_info}</span>
						</div>

						{#if serverProps.chat_template}
							<div class="min-w-0 space-y-2">
								<div class="text-xs font-medium text-muted-foreground">Chat Template</div>

								<div class="overflow-x-auto rounded-md bg-muted p-4">
									<pre class="font-mono text-xs whitespace-pre">{serverProps.chat_template}</pre>
								</div>
							</div>
						{/if}
					</div>
				{/if}
			{:else if !isLoadingModels}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">No model information available</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>

{#snippet infoRow(label: string, value: string, valueClass: string = '')}
	<div class="flex items-center justify-between gap-3">
		<span class="shrink-0 text-xs font-medium text-muted-foreground {valueClass}">{label}</span>

		<span class="text-sm {valueClass}">{value}</span>
	</div>
{/snippet}
