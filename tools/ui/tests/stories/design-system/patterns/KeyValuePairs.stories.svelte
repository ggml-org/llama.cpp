<script lang="ts" module>
	import StateGallery from '../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { KeyValuePairs } from '$lib/components/app';
	import type { KeyValuePair } from '$lib/types';

	// the editable key and value list, as used for MCP server headers
	const { Story } = defineMeta({
		component: KeyValuePairs,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Patterns/KeyValuePairs'
	});
</script>

<script lang="ts">
	let pairs = $state<KeyValuePair[]>([
		{ key: 'Authorization', value: 'Bearer sk-local' },
		{ key: 'X-Workspace', value: 'research' }
	]);
	let emptyPairs = $state<KeyValuePair[]>([]);
</script>

<Story asChild name="Default">
	<StateGallery columns={2} minItemWidth="20rem">
		<StateGalleryItem label="With pairs">
			<KeyValuePairs
				addButtonLabel="Add header"
				onPairsChange={(next) => (pairs = next)}
				{pairs}
				sectionLabel="Headers"
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Empty">
			<KeyValuePairs
				addButtonLabel="Add header"
				emptyMessage="No headers configured."
				onPairsChange={(next) => (emptyPairs = next)}
				pairs={emptyPairs}
				sectionLabel="Headers"
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
