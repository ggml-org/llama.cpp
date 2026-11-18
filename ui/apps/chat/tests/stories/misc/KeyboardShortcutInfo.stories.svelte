<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { KeyboardShortcutInfo } from '#lib/components/app/index.ts';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// shortcut hints in menus; hidden by default and shown by the row's hover, so forced visible here
	const { Story } = defineMeta({
		component: KeyboardShortcutInfo,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Misc/KeyboardShortcutInfo'
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
