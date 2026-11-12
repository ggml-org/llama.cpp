<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ModelFamilyCard } from '$lib/components/models';

	// a model family as llama.app's catalog shows it: one card for every size and quant of the family, its capabilities
	// as icons, its sizes in one line. Data from llama.app's catalog and the Hub
	const { Story } = defineMeta({
		component: ModelFamilyCard,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelFamilyCard'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="18rem">
		<StateGalleryItem label="several sizes" note="sees, hears, reasons, uses tools">
			<ModelFamilyCard
				description="Google's most capable open agentic models for your laptops, with vision, audio, and support for 140+ languages."
				href="#gemma-4"
				memGb={16}
				modalities={{ audio: true, video: false, vision: true }}
				name="Gemma 4"
				org="google"
				quantOrg="ggml-org"
				released="Mar 2026"
				sizeRange={{ max: 33_400_000_000, min: 4_590_807_392 }}
				sizes="E2B–31B · 5 sizes"
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>

		<StateGalleryItem label="one size">
			<ModelFamilyCard
				description="Liquid AI's fast local model for reasoning and tool use."
				href="#lfm-2-5"
				memGb={24}
				name="LFM 2.5"
				org="LiquidAI"
				quantOrg="ggml-org"
				released="Oct 2026"
				sizeRange={{ max: 25_493_733_408, min: 14_526_259_232 }}
				sizes="24B-A2B"
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>

		<StateGalleryItem label="non-commercial" note="in place of the release month">
			<ModelFamilyCard
				description="Bespoke Labs' mid-sized decision model. Given a context and typed questions, it returns a probability for each option."
				href="#bespoke-nimble"
				memGb={16}
				name="Bespoke Nimble"
				nonCommercial
				org="bespokelabs"
				quantOrg="ggml-org"
				sizeRange={{ max: 9_527_503_392, min: 6_324_185_632 }}
				sizes="9B"
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
