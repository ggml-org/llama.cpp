<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ModelQuantChip, ModelQuantRow } from '$lib/components/models';
	import { minMemoryTierGb } from '$lib/models';

	// one quant of a model, in each state the app can give it; actions come back as callbacks, so the chip downloads
	// nothing itself
	const { Story } = defineMeta({
		component: ModelQuantChip,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelQuantChip'
	});

	const Q4 = 16_400_000_000;
	const Q8 = 28_600_000_000;
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="12rem">
		<StateGalleryItem label="idle" note="downloads on click">
			<ModelQuantChip label="Q4_K_M" sizeBytes={Q4} />
		</StateGalleryItem>

		<StateGalleryItem label="downloading">
			<ModelQuantChip
				label="Q4_K_M"
				progress={{ downloadedBytes: Q4 * 0.42, totalBytes: Q4 }}
				sizeBytes={Q4}
				state="downloading"
			/>
		</StateGalleryItem>

		<StateGalleryItem label="paused">
			<ModelQuantChip label="Q4_K_M" sizeBytes={Q4} state="paused" />
		</StateGalleryItem>

		<StateGalleryItem label="downloaded" note="deletes on click">
			<ModelQuantChip label="Q8_0" sizeBytes={Q8} state="downloaded" />
		</StateGalleryItem>

		<StateGalleryItem label="failed" note="retries on click">
			<ModelQuantChip label="Q4_K_M" sizeBytes={Q4} state="failed" />
		</StateGalleryItem>

		<StateGalleryItem label="with a draft" note="the MTP head rides along">
			<ModelQuantChip
				companion={{ kind: 'mtp', quant: 'Q8_0', sizeBytes: 900_000_000 }}
				label="Q4_K_M"
				sizeBytes={Q4}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="as a link" note="hands the install to another app">
			<ModelQuantChip
				actionLabel="Install Q4_K_M in Llama"
				href="llama://install?repo=ggml-org/Qwen3.8-27B-GGUF&quant=Q4_K_M"
				label="Q4_K_M"
				sizeBytes={Q4}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>

<Story asChild name="Rows">
	<div class="w-[min(36rem,calc(100vw-2rem))] divide-y divide-border p-4">
		<ModelQuantRow bitDepth={8} memGb={minMemoryTierGb(Q8)}>
			<ModelQuantChip label="Q8_0" sizeBytes={Q8} state="downloaded" />
		</ModelQuantRow>

		<ModelQuantRow bitDepth={4} memGb={minMemoryTierGb(Q4)}>
			<ModelQuantChip label="Q4_K_M" sizeBytes={Q4} />
			<ModelQuantChip label="Q4_0" sizeBytes={15_700_000_000} />

			{#snippet command()}
				<pre
					class="overflow-x-auto rounded-panel bg-(--surface-code) px-3 py-2 text-xs text-(--fg-code)">llama serve -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M</pre>
			{/snippet}
		</ModelQuantRow>

		<ModelQuantRow bitDepth={2} memGb={minMemoryTierGb(10_200_000_000)}>
			<ModelQuantChip label="Q2_K" sizeBytes={10_200_000_000} />
		</ModelQuantRow>
	</div>
</Story>
