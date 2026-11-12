<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ModelId, ModelOrgAvatar } from '$lib/components/models';

	// a model's name with its badges and icons, as Llama Chat's lists and llama.app's catalog show it; the models are
	// from llama.app's catalog
	const { Story } = defineMeta({
		component: ModelId,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelId'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="20rem">
		<StateGalleryItem label="name and params">
			<ModelId name="gpt-oss" params="20B" />
		</StateGalleryItem>

		<StateGalleryItem label="with org and quant">
			<ModelId name="gemma-4-E4B-it-GGUF" org="ggml-org" params="E4B" quantization="Q4_0" />
		</StateGalleryItem>

		<StateGalleryItem label="mixture of experts" note="active parameters joined to the total">
			<ModelId activatedParams="A3B" name="Qwen3.6" params="35B" quantization="Q4_K_M" />
		</StateGalleryItem>

		<StateGalleryItem label="capabilities" note="vision, audio, reasoning, tool use">
			<ModelId
				modalities={{ audio: true, video: false, vision: true }}
				name="gemma-4"
				params="12B"
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>

		<StateGalleryItem label="draft model" note="an MTP head rides along">
			<ModelId draftKinds={['mtp']} name="Qwen3.8" params="27B" quantization="Q4_K_M" />
		</StateGalleryItem>

		<StateGalleryItem label="tags">
			<ModelId name="EmbeddingGemma 2" params="740M" tags={['embedding']} />
		</StateGalleryItem>

		<StateGalleryItem label="icons on a new line" note="with context and size, as in Discover">
			<ModelId
				contextLength={262144}
				iconsOnNewLine
				modalities={{ audio: false, video: false, vision: true }}
				name="Qwen3.8-27B-GGUF"
				sizeRange={{ max: 28_600_000_000, min: 16_400_000_000 }}
				supportsThinking
				supportsToolUse
				wrap
			/>
		</StateGalleryItem>

		<StateGalleryItem label="in a row" note="with the maker's avatar">
			<span class="flex items-center gap-2.5">
				<ModelOrgAvatar org="openai" quantOrg="ggml-org" size="size-9" />

				<ModelId
					contextLength={131072}
					iconsOnNewLine
					name="gpt-oss-20b-GGUF"
					sizeRange={{ max: 12_109_566_560, min: 12_109_566_560 }}
					supportsThinking
					supportsToolUse
				/>
			</span>
		</StateGalleryItem>
	</StateGallery>
</Story>
