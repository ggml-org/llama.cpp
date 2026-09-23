<script lang="ts">
	import { modelQuantLabel, modelSizeLabel } from '../utils';
	import { Play, Square, X } from '@lucide/svelte';
	import { ModelAvatar, ModelCapabilityIcons } from '$lib/components/app';
	import ModelsDiscoverDetailsMetadataItem from '$lib/components/app/models/discover/ModelsDiscoverDetails/ModelsDiscoverDetailsMetadataItem.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { ModelCapability } from '$lib/enums';
	import type { ApiLlamaCppServerProps } from '$lib/types/api';
	import type { ModelOption } from '$lib/types/models';
	import { formatNumber } from '$lib/utils/formatters';

	interface Props {
		isCustomized: boolean;
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
		serverProps: ApiLlamaCppServerProps | null | undefined;
	}

	let {
		isCustomized,
		isLoaded,
		onClose,
		onToggleLoad,
		onUseInNewChat,
		option,
		serverProps
	}: Props = $props();

	// the same chips the discovery details show for a repo, from what we know here
	let params = $derived(option.parsedId?.params ?? null);
	let quant = $derived(modelQuantLabel(option));
	let context = $derived(
		serverProps?.default_generation_settings?.n_ctx ?? option.contextLength ?? null
	);
	let size = $derived(modelSizeLabel(option));
	let supportsToolUse = $derived(option.capabilities.includes(ModelCapability.TOOL_USE));
	let supportsThinking = $derived(option.capabilities.includes(ModelCapability.REASONING));
</script>

<header class="space-y-3 px-4 pt-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelAvatar {option} size="size-12" />

			<div class="min-w-0">
				<div class="flex items-center gap-2">
					<h1 class="truncate text-lg font-semibold">{option.name}</h1>

					{#if isCustomized}
						<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">CUSTOMIZED</Badge>
					{/if}

					<ModelCapabilityIcons
						gapClass="gap-2"
						iconSize="h-4 w-4"
						modalities={option.modalities}
						{supportsThinking}
						{supportsToolUse}
					/>
				</div>

				<p class="truncate text-xs text-muted-foreground">
					{option.model} · {isLoaded ? 'loaded' : 'not loaded'}
				</p>
			</div>
		</div>

		<Button
			aria-label="Close details"
			class="h-7 w-7"
			onclick={onClose}
			size="icon"
			variant="ghost"
		>
			<X class="h-4 w-4" />
		</Button>
	</div>

	{#if params || quant || context || size}
		<div class="flex flex-wrap items-center gap-1.5">
			{#if params}
				<ModelsDiscoverDetailsMetadataItem label="Parameters" value={params} />
			{/if}

			{#if quant}
				<ModelsDiscoverDetailsMetadataItem label="Quantization" value={quant} />
			{/if}

			{#if context}
				<ModelsDiscoverDetailsMetadataItem
					label="Context"
					value={`${formatNumber(context)} tokens`}
				/>
			{/if}

			{#if size}
				<ModelsDiscoverDetailsMetadataItem label="Size" value={size} />
			{/if}
		</div>
	{/if}

	<div class="flex gap-2">
		<Button class="flex-1 gap-1.5" onclick={onUseInNewChat} variant="outline">
			<Play class="h-3.5 w-3.5" />

			Use in New Chat
		</Button>

		<Button class="flex-1 gap-1.5" onclick={onToggleLoad} variant="outline">
			<Square class="h-3.5 w-3.5" />

			{isLoaded ? 'Eject Model' : 'Load Model'}
		</Button>
	</div>
</header>
