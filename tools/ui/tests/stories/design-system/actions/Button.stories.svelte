<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { Plus, Trash2 } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import type { ButtonSize, ButtonVariant } from '$lib/components/ui/button/button.svelte';

	const { Story } = defineMeta({
		component: Button,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Actions/Button'
	});

	const variants: NonNullable<ButtonVariant>[] = [
		'default',
		'secondary',
		'tertiary',
		'outline',
		'ghost',
		'destructive',
		'link'
	];
	const sizes: NonNullable<ButtonSize>[] = ['sm', 'default', 'lg'];
</script>

<Story asChild name="Default">
	<StateGallery columns={4}>
		{#each variants as variant (variant)}
			<StateGalleryItem label={variant}>
				<div class="flex flex-wrap items-center gap-2">
					{#each sizes as size (size)}
						<Button {size} {variant}>Button</Button>
					{/each}

					<Button aria-label="Add" size="icon" {variant}>
						<Plus />
					</Button>

					<Button disabled {variant}>Disabled</Button>
				</div>
			</StateGalleryItem>
		{/each}

		<StateGalleryItem label="With icon">
			<div class="flex flex-wrap items-center gap-2">
				<Button><Plus /> New chat</Button>

				<Button variant="destructive"><Trash2 /> Delete</Button>
			</div>
		</StateGalleryItem>
	</StateGallery>
</Story>
