<script lang="ts" module>
	import {
		conversation,
		conversationMessages,
		MODEL_NAME,
		modelServerProps,
		routerModels,
		seedStores,
		ServerRole
	} from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatFormContextGauge from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/ChatFormContextGauge.svelte';
	import ContextGaugePopup from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugePopup.svelte';
	import { gaugePopupClose } from '$lib/components/app/chat/ChatForm/ChatFormContextGauge/gauge-popup.svelte';
	import { userEvent, within } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatFormContextGauge,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatForm/ChatFormContextGauge'
	});

	const N_CTX = 32768;

	// the popup state is module-wide, so close it around each story
	function seedUsage(promptTokens: number) {
		return () => {
			gaugePopupClose();

			const restore = seedStores({
				activeConversation: conversation,
				activeMessages: conversationMessages(promptTokens),
				modelProps: {
					[MODEL_NAME]: modelServerProps({
						default_generation_settings: { n_ctx: N_CTX }
					} as Partial<ApiLlamaCppServerProps>)
				},
				serverProps: modelServerProps(),
				serverRole: ServerRole.MODEL
			});

			return () => {
				gaugePopupClose();
				restore();
			};
		};
	}

	function seedModelNotLoaded() {
		gaugePopupClose();

		const restore = seedStores({
			activeConversation: conversation,
			activeMessages: conversationMessages(4000),
			models: routerModels,
			routerModels: [],
			selectedModelId: routerModels[0].id,
			serverProps: modelServerProps({ role: ServerRole.ROUTER }),
			serverRole: ServerRole.ROUTER
		});

		return () => {
			gaugePopupClose();
			restore();
		};
	}

	async function openPopup({ canvasElement }: { canvasElement: HTMLElement }) {
		const canvas = within(canvasElement);

		canvas.getByRole('button', { name: 'Context usage' }).focus();
		await userEvent.keyboard('{Enter}');
		await canvas.findByRole('status');
	}
</script>

{#snippet frame()}
	<!-- the popup positions itself against the closest form -->
	<form class="relative mt-96 flex w-80 justify-end rounded-3xl border p-3">
		<ChatFormContextGauge />

		<ContextGaugePopup />
	</form>
{/snippet}

<Story asChild beforeEach={seedUsage(6000)} name="Default" play={openPopup}>
	{@render frame()}
</Story>

<Story asChild beforeEach={seedUsage(6000)} name="DialOnly">
	{@render frame()}
</Story>

<Story asChild beforeEach={seedUsage(27500)} name="Warning" play={openPopup}>
	{@render frame()}
</Story>

<Story asChild beforeEach={seedUsage(31500)} name="Critical" play={openPopup}>
	{@render frame()}
</Story>

<Story asChild beforeEach={seedUsage(36000)} name="OverLimit" play={openPopup}>
	{@render frame()}
</Story>

<Story asChild beforeEach={seedModelNotLoaded} name="ModelNotLoaded" play={openPopup}>
	{@render frame()}
</Story>
