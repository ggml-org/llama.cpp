<!--
	Llama UI's ModelId for a raw model id: parses it, adds the user's display settings (raw names, quantization,
	tags) and leaves out the auxiliary sidecars.
-->
<script lang="ts">
	import { ModelId as LlamaUiModelId } from '@ggml-org/llama-ui/models';
	import { TruncatedText } from '#lib/components/app/index.ts';
	import type { ModelDraftSidecar } from '#lib/enums/index.ts';
	import { ModelsService } from '#lib/services/models.service.ts';
	import { settingsStore } from '#lib/stores/index.ts';
	import type { ModelModalities, ModelSidecarBadge } from '#lib/types/models.js';
	import { isAuxSidecar } from '#lib/utils/index.ts';

	interface Props {
		modelId: string;
		hideOrgName?: boolean;
		hideName?: boolean;
		hideModalities?: boolean;
		hideCapabilities?: boolean;
		hideParameters?: boolean;
		showRaw?: boolean;
		showRawTooltip?: boolean;
		hideQuantization?: boolean;
		hideTags?: boolean;
		aliases?: string[];
		tags?: string[];
		/** Render the capability/modality/context icons on a second row. */
		iconsOnNewLine?: boolean;
		modalities?: ModelModalities;
		supportsThinking?: boolean;
		supportsToolUse?: boolean;
		/** Native title for the root element; keeps long lists light where a floating tooltip per row is too costly. */
		title?: string;
		/** Context length in tokens; renders a context icon when set. */
		contextLength?: number;
		/** Min/max GGUF file size (main + draft) across quants; renders a range when set. */
		sizeRange?: { min: number; max: number } | null;
		/** Draft sidecars available for the model, badged with their own quant. */
		draftSidecars?: ModelSidecarBadge[];
		/** Draft sidecar kinds the repo offers, one badge per kind and no quant. */
		draftKinds?: ModelDraftSidecar[];
		/** Give the id a row of its own, with its badges and icons together under it. */
		stackId?: boolean;
		/** Allow badges to wrap onto new lines instead of truncating. */
		wrap?: boolean;
		class?: string;
	}

	let {
		class: className = '',
		draftSidecars = [],
		hideParameters = false,
		hideQuantization,
		hideTags,
		modelId,
		showRaw = undefined,
		showRawTooltip = false,
		tags,
		...rest
	}: Props = $props();

	let parsed = $derived(ModelsService.parseModelId(modelId));
	let resolvedShowRaw = $derived(
		showRaw ?? (settingsStore.config.showRawModelNames as boolean) ?? false
	);
	let resolvedHideQuantization = $derived(
		hideQuantization ?? !settingsStore.config.showModelQuantization
	);
	let resolvedHideTags = $derived(hideTags ?? !settingsStore.config.showModelTags);
</script>

{#if resolvedShowRaw}
	<TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} />
{:else}
	<LlamaUiModelId
		{...rest}
		activatedParams={parsed.activatedParams ?? undefined}
		class={className}
		draftSidecars={draftSidecars.filter((badge) => !isAuxSidecar(badge.kind))}
		name={parsed.modelName ?? modelId}
		org={parsed.orgName ?? undefined}
		params={hideParameters ? undefined : (parsed.params ?? undefined)}
		quantization={resolvedHideQuantization ? undefined : (parsed.quantization ?? undefined)}
		sidecar={parsed.sidecar ?? undefined}
		tags={resolvedHideTags ? [] : [...(parsed.tags ?? []), ...(tags ?? [])]}
		tooltip={showRawTooltip ? modelId : undefined}
	/>
{/if}
