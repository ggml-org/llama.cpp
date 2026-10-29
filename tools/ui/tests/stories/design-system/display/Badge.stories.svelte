<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Badge, type BadgeVariant } from '$lib/components/ui/badge';

	const { Story } = defineMeta({
		component: Badge,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Badge'
	});

	const variants: NonNullable<BadgeVariant>[] = ['default', 'secondary', 'outline', 'destructive'];
</script>

<Story asChild name="Default">
	<StateGallery columns={4} minItemWidth="8rem">
		{#each variants as variant (variant)}
			<StateGalleryItem label={variant}>
				<Badge {variant}>Q4_K_M</Badge>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>
