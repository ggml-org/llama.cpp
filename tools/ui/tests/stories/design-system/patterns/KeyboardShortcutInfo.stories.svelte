<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { KeyboardShortcutInfo } from '$lib/components/app';

	// shortcut hints in menus; hidden by default and shown by the row's hover, so forced visible here
	const { Story } = defineMeta({
		component: KeyboardShortcutInfo,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/KeyboardShortcutInfo'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={3} minItemWidth="10rem">
		<StateGalleryItem label="Default">
			<KeyboardShortcutInfo class="opacity-100" keys={['cmd', 'k']} />
		</StateGalleryItem>

		<StateGalleryItem label="With shift">
			<KeyboardShortcutInfo class="opacity-100" keys={['shift', 'cmd', 'o']} />
		</StateGalleryItem>

		<StateGalleryItem label="Destructive">
			<KeyboardShortcutInfo
				class="opacity-100"
				keys={['shift', 'cmd', 'd']}
				variant="destructive"
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
