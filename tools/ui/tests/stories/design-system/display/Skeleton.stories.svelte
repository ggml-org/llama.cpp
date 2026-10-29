<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Skeleton } from '$lib/components/ui/skeleton';

	const { Story } = defineMeta({
		component: Skeleton,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Skeleton'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="16rem">
		<StateGalleryItem label="Text lines">
			<div class="space-y-2">
				<Skeleton class="h-4 w-3/4" />

				<Skeleton class="h-4 w-full" />

				<Skeleton class="h-4 w-1/2" />
			</div>
		</StateGalleryItem>

		<StateGalleryItem label="List item">
			<div class="flex items-center gap-3">
				<Skeleton class="size-10 rounded-full" />

				<div class="flex-1 space-y-2">
					<Skeleton class="h-4 w-2/3" />

					<Skeleton class="h-3 w-1/3" />
				</div>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
