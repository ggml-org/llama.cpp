<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ServerErrorSplash from '$lib/components/app/server/ServerErrorSplash.svelte';
	import ServerLoadingSplash from '$lib/components/app/server/ServerLoadingSplash.svelte';

	// the loading and error splashes are the two screens shown before the app connects
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'App/ServerSplash'
	});
</script>

<Story asChild name="Default">
	<StateGallery columns={2}>
		<StateGalleryItem label="Loading">
			<ServerLoadingSplash class="h-96" />
		</StateGalleryItem>

		<StateGalleryItem label="Error">
			<ServerErrorSplash
				class="h-96"
				error="Failed to fetch: the server at localhost:8080 is not reachable"
				onRetry={() => console.log('retry')}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
