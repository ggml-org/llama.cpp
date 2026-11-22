<script lang="ts" module>
	import { Clock, Gauge } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { BadgeInfo, BadgesModality } from '#lib/components/app/index.ts';
	import { ModelModality } from '#lib/enums/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// the clickable info badges for message statistics and models, and the modality icons
	const { Story } = defineMeta({
		component: BadgeInfo,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Badges/InfoBadges'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3}>
		<StateGalleryItem label="BadgeInfo">
			<div class="flex flex-wrap items-center gap-2">
				<BadgeInfo>
					{#snippet icon()}
						<Gauge class="size-icon-xs" />
					{/snippet}

					42.7 t/s
				</BadgeInfo>

				<BadgeInfo>
					{#snippet icon()}
						<Clock class="size-icon-xs" />
					{/snippet}

					3.2 s
				</BadgeInfo>

				<BadgeInfo>qwen3-8b</BadgeInfo>
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="BadgesModality" note="vision, audio">
			<BadgesModality modalities={[ModelModality.VISION, ModelModality.AUDIO]} />
		</StateGalleryItem>

		<StateGalleryItem label="BadgesModality" note="all">
			<BadgesModality
				modalities={[ModelModality.VISION, ModelModality.AUDIO, ModelModality.VIDEO]}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
