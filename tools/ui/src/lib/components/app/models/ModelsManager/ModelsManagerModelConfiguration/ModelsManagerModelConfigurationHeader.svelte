<script lang="ts">
	import { Play, Square, X } from '@lucide/svelte';
	import { ModelAvatar, ModelId } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { ModelCapability } from '$lib/enums';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		isCustomized: boolean;
		isLoaded: boolean;
		onClose: () => void;
		onToggleLoad: () => void;
		onUseInNewChat: () => void;
		option: ModelOption;
	}

	let { isCustomized, isLoaded, onClose, onToggleLoad, onUseInNewChat, option }: Props = $props();

	let supportsToolUse = $derived(option.capabilities.includes(ModelCapability.TOOL_USE));
	let supportsThinking = $derived(option.capabilities.includes(ModelCapability.REASONING));
</script>

<header class="space-y-3 px-4 pt-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-center gap-2">
			<ModelAvatar {option} showBaseModelAvatar size="size-12" />

			<div class="min-w-0">
				<div class="flex items-center gap-2">
					<ModelId
						aliases={option.aliases}
						class="min-w-0"
						modalities={option.modalities}
						modelId={option.model}
						{supportsThinking}
						{supportsToolUse}
						tags={option.tags}
						title={option.model}
					/>

					{#if isCustomized}
						<Badge class="h-5 shrink-0 px-1.5 text-[10px]" variant="secondary">CUSTOMIZED</Badge>
					{/if}
				</div>

				<p class="truncate text-xs text-muted-foreground">
					{isLoaded ? 'Loaded' : 'Not loaded'}
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
