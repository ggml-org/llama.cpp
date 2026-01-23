<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Table from '$lib/components/ui/table';
	import { BadgeModality, CopyToClipboardIcon } from '$lib/components/app';
	import { serverStore } from '$lib/stores/server.svelte';
	import { modelsStore, modelOptions, modelsLoading } from '$lib/stores/models.svelte';
	import { formatFileSize, formatParameters, formatNumber } from '$lib/utils';
	import { t } from '$lib/i18n';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { open = $bindable(), onOpenChange }: Props = $props();

	let serverProps = $derived(serverStore.props);
	let modelName = $derived(modelsStore.singleModelName);
	let models = $derived(modelOptions());
	let isLoadingModels = $derived(modelsLoading());

	// Get the first model for single-model mode display
	let firstModel = $derived(models[0] ?? null);

	// Get modalities from modelStore using the model ID from the first model
	let modalities = $derived.by(() => {
		if (!firstModel?.id) return [];
		return modelsStore.getModelModalitiesArray(firstModel.id);
	});

	// Ensure models are fetched when dialog opens
	$effect(() => {
		if (open && models.length === 0) {
			modelsStore.fetch();
		}
	});
</script>

<Dialog.Root bind:open {onOpenChange}>
	<Dialog.Content class="@container z-9999 !max-w-[60rem] max-w-full">
		<style>
			@container (max-width: 56rem) {
				.resizable-text-container {
					max-width: calc(100vw - var(--threshold));
				}
			}
		</style>

		<Dialog.Header>
			<Dialog.Title>{t('dialog.model_info.title')}</Dialog.Title>
			<Dialog.Description>{t('dialog.model_info.description')}</Dialog.Description>
		</Dialog.Header>

		<div class="space-y-6 py-4">
			{#if isLoadingModels}
				<div class="flex items-center justify-center py-8">
					<div class="text-sm text-muted-foreground">
						{t('dialog.model_info.loading')}
					</div>
				</div>
			{:else if firstModel}
				{@const modelMeta = firstModel.meta}

				{#if serverProps}
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head class="w-[10rem]">{t('dialog.model_info.header_model')}</Table.Head>

								<Table.Head>
									<div class="inline-flex items-center gap-2">
										<span
											class="resizable-text-container min-w-0 flex-1 truncate"
											style:--threshold="12rem"
										>
											{modelName}
										</span>

										<CopyToClipboardIcon
											text={modelName || ''}
											canCopy={!!modelName}
											ariaLabel={t('dialog.model_info.copy_model_name')}
										/>
									</div>
								</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							<!-- Model Path -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">
									{t('dialog.model_info.file_path')}
								</Table.Cell>

								<Table.Cell
									class="inline-flex h-10 items-center gap-2 align-middle font-mono text-xs"
								>
									<span
										class="resizable-text-container min-w-0 flex-1 truncate"
										style:--threshold="14rem"
									>
										{serverProps.model_path}
									</span>

									<CopyToClipboardIcon
										text={serverProps.model_path}
										ariaLabel={t('dialog.model_info.copy_model_path')}
									/>
								</Table.Cell>
							</Table.Row>

							<!-- Context Size -->
							<Table.Row>
								<Table.Cell class="h-10 align-middle font-medium">
									{t('dialog.model_info.context_size')}
								</Table.Cell>
								<Table.Cell
									>{formatNumber(serverProps.default_generation_settings.n_ctx)}
									{t('dialog.model_info.tokens')}</Table.Cell
								>
							</Table.Row>

							<!-- Training Context -->
							{#if modelMeta?.n_ctx_train}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">
										{t('dialog.model_info.training_context')}
									</Table.Cell>
									<Table.Cell
										>{formatNumber(modelMeta.n_ctx_train)} {t('dialog.model_info.tokens')}</Table.Cell
									>
								</Table.Row>
							{/if}

							<!-- Model Size -->
							{#if modelMeta?.size}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">
										{t('dialog.model_info.model_size')}
									</Table.Cell>
									<Table.Cell>{formatFileSize(modelMeta.size)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Parameters -->
							{#if modelMeta?.n_params}
								<Table.Row>
									<Table.Cell class="h-10 align-middle font-medium">
										{t('dialog.model_info.parameters')}
									</Table.Cell>
									<Table.Cell>{formatParameters(modelMeta.n_params)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Embedding Size -->
							{#if modelMeta?.n_embd}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">
										{t('dialog.model_info.embedding_size')}
									</Table.Cell>
									<Table.Cell>{formatNumber(modelMeta.n_embd)}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Vocabulary Size -->
							{#if modelMeta?.n_vocab}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">
										{t('dialog.model_info.vocab_size')}
									</Table.Cell>
									<Table.Cell
										>{formatNumber(modelMeta.n_vocab)} {t('dialog.model_info.tokens')}</Table.Cell
									>
								</Table.Row>
							{/if}

							<!-- Vocabulary Type -->
							{#if modelMeta?.vocab_type}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">
										{t('dialog.model_info.vocab_type')}
									</Table.Cell>
									<Table.Cell class="align-middle capitalize">{modelMeta.vocab_type}</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Total Slots -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">
									{t('dialog.model_info.parallel_slots')}
								</Table.Cell>
								<Table.Cell>{serverProps.total_slots}</Table.Cell>
							</Table.Row>

							<!-- Modalities -->
							{#if modalities.length > 0}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">
										{t('dialog.model_info.modalities')}
									</Table.Cell>
									<Table.Cell>
										<div class="flex flex-wrap gap-1">
											<BadgeModality {modalities} />
										</div>
									</Table.Cell>
								</Table.Row>
							{/if}

							<!-- Build Info -->
							<Table.Row>
								<Table.Cell class="align-middle font-medium">
									{t('dialog.model_info.build_info')}
								</Table.Cell>
								<Table.Cell class="align-middle font-mono text-xs"
									>{serverProps.build_info}</Table.Cell
								>
							</Table.Row>

							<!-- Chat Template -->
							{#if serverProps.chat_template}
								<Table.Row>
									<Table.Cell class="align-middle font-medium">
										{t('dialog.model_info.chat_template')}
									</Table.Cell>
									<Table.Cell class="py-10">
										<div class="max-h-120 overflow-y-auto rounded-md bg-muted p-4">
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
					<div class="text-sm text-muted-foreground">
						{t('dialog.model_info.no_info')}
					</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
