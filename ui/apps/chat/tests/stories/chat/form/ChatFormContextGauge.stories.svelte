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
	import ChatFormContextGauge from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/ChatFormContextGauge.svelte';
	import ContextGaugeDrawer from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/ContextGaugeDrawer.svelte';
	import { gaugePopupClose } from '#lib/components/app/chat/ChatForm/ChatFormContextGauge/gauge-popup.svelte.ts';
	import { deviceStore } from '#lib/stores/index.ts';
	import { userEvent, within } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatFormContextGauge,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Chat/ChatForm/ChatFormContextGauge'
	});

	const N_CTX = 32768;

	// the open state is module-wide, so close it around each story
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

	// the popover renders in a portal, so it is looked up on the whole page
	async function openPopup({ canvasElement }: { canvasElement: HTMLElement }) {
		await userEvent.click(within(canvasElement).getByRole('button', { name: 'Context usage' }));
		await within(document.body).findByRole('dialog');
	}
</script>

{#snippet frame()}
	<!-- room above the form for the popover -->
	<form class="relative mt-96 flex w-80 justify-end rounded-3xl border p-3">
		<ChatFormContextGauge />

		<!-- a phone opens the panel in the drawer the chat form renders -->
		{#if deviceStore.isMobile}
			<ContextGaugeDrawer />
		{/if}
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
