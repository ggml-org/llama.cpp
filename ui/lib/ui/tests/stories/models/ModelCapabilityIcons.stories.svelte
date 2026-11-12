<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ModelCapabilityIcons } from '$lib/components/models';

	// what a model can do, as muted icons with a tooltip each: modalities (vision, audio, video) after capabilities
	// (tool use, reasoning)
	const { Story } = defineMeta({
		component: ModelCapabilityIcons,
		parameters: {
			layout: 'centered'
		},
		title: 'UI/Blocks/Models/ModelCapabilityIcons'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="10rem">
		<StateGalleryItem label="vision">
			<ModelCapabilityIcons modalities={{ audio: false, video: false, vision: true }} />
		</StateGalleryItem>

		<StateGalleryItem label="reads, sees and hears">
			<ModelCapabilityIcons modalities={{ audio: true, video: false, vision: true }} />
		</StateGalleryItem>

		<StateGalleryItem label="reasoning and tools">
			<ModelCapabilityIcons supportsThinking supportsToolUse />
		</StateGalleryItem>

		<StateGalleryItem label="everything">
			<ModelCapabilityIcons
				modalities={{ audio: true, video: true, vision: true }}
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>

		<StateGalleryItem label="larger" note="as in a model's header">
			<ModelCapabilityIcons
				gapClass="gap-2"
				iconSize="size-icon-md"
				modalities={{ audio: true, video: false, vision: true }}
				supportsThinking
				supportsToolUse
			/>
		</StateGalleryItem>

		<StateGalleryItem label="nothing" note="renders no box at all">
			<ModelCapabilityIcons />
		</StateGalleryItem>
	</StateGallery>
</Story>
