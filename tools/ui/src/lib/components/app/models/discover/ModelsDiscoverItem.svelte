<script lang="ts">
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelInfo } from '$lib/types/huggingface';
	import ModelsDiscoverAvatar from './ModelsDiscoverAvatar.svelte';
	import ModelsDiscoverInfo from './ModelsDiscoverInfo.svelte';

	interface Props {
		model: HfModelInfo;
		active?: boolean;
		/** Show the original (base) model's org avatar instead of the repo's org. */
		showBaseModelAvatar?: boolean;
		onSelect?: (modelId: string) => void;
	}

	let { model, active = false, showBaseModelAvatar = false, onSelect }: Props = $props();

	let org = $derived(model.id.split('/')[0] ?? model.id);

	// Org whose avatar is shown: the base model's org when showBaseModelAvatar
	// (e.g. the Qwen logo for ggml-org/Qwen3.8-27B-GGUF), else the repo's org.
	let avatarOrg = $derived.by(() => {
		if (!showBaseModelAvatar) return org;

		const base = HuggingFaceService.getBaseModels(model)[0];

		return base?.split('/')[0] || org;
	});
</script>

<li>
	<button
		type="button"
		onclick={() => onSelect?.(model.id)}
		aria-current={active ? 'page' : undefined}
		class="flex w-full cursor-pointer items-start gap-2.5 rounded-lg p-2.5 text-left transition-colors {active
			? 'bg-primary/10 hover:bg-primary/15'
			: 'hover:bg-muted/60'}"
	>
		<ModelsDiscoverAvatar org={avatarOrg} quantOrg={showBaseModelAvatar ? org : undefined} />
		<ModelsDiscoverInfo {model} />
	</button>
</li>
