<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ServerErrorSplash from '#lib/components/app/servers/ServerErrorSplash.svelte';
	import ServerLoadingSplash from '#lib/components/app/servers/ServerLoadingSplash.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	// the loading and error splashes are the two screens shown before the app connects
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Servers/ServerSplash'
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
				error="Failed to fetch: the server at localhost:9931 is not reachable"
				onRetry={() => console.log('retry')}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
