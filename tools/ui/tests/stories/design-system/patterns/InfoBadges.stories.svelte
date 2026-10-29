<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Clock, Gauge } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { BadgeInfo, BadgesModality } from '$lib/components/app';
	import { ModelModality } from '$lib/enums';

	// the clickable info badges for message statistics and models, and the modality icons
	const { Story } = defineMeta({
		component: BadgeInfo,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/InfoBadges'
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
