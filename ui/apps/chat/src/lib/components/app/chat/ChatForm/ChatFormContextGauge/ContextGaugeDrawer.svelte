<script lang="ts">
	import ContextGaugePanel from './ContextGaugePanel.svelte';
	import { gaugePopup } from './gauge-popup.svelte';
	import * as Drawer from '@ggml-org/llama-ui/ui/drawer';
</script>

<!-- a phone shows the context panel in a drawer from the screen's bottom; wider screens use the gauge's popover -->
<Drawer.Root bind:open={gaugePopup.open}>
	<Drawer.Content>
		<Drawer.Header>
			<Drawer.Title>Context</Drawer.Title>

			<Drawer.Description class="sr-only">
				Context window usage of the model the chat runs
			</Drawer.Description>
		</Drawer.Header>

		<ContextGaugePanel inDrawer />
	</Drawer.Content>
</Drawer.Root>
