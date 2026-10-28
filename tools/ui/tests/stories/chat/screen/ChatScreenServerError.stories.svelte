<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenServerError from '$lib/components/app/chat/ChatScreen/ChatScreenServerError.svelte';
	import { serverStore } from '$lib/stores';

	const { Story } = defineMeta({
		component: ChatScreenServerError,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatScreenServerError'
	});

	function seedServer(state: { error: string | null; loading?: boolean; status?: number | null }) {
		serverStore.error = state.error;
		serverStore.loading = state.loading ?? false;
		serverStore.status = state.status ?? null;
	}
</script>

<Story
	asChild
	beforeEach={() =>
		seedServer({ error: 'Failed to fetch server properties: connection refused', status: 500 })}
	name="Default"
>
	<div class="w-[48rem]">
		<ChatScreenServerError />
	</div>
</Story>

<Story
	asChild
	beforeEach={() =>
		seedServer({
			error: 'Failed to fetch server properties: connection refused',
			loading: true,
			status: 500
		})}
	name="Retrying"
>
	<div class="w-[48rem]">
		<ChatScreenServerError />
	</div>
</Story>

<!-- HTTP 503 while the server is still loading the model -->
<Story
	asChild
	beforeEach={() => seedServer({ error: 'Loading model', status: 503 })}
	name="LoadingModel"
>
	<div class="w-[48rem]">
		<ChatScreenServerError />
	</div>
</Story>
