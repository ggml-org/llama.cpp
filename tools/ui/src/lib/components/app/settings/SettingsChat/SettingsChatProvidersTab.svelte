<script lang="ts">
	import { SettingsBackends } from '$lib/components/app/backends';
	import * as Tabs from '$lib/components/ui/tabs';
	import type { BackendProtocol } from '$lib/types';

	const PROTOCOL_TABS: Array<{ label: string; value: BackendProtocol }> = [
		{ label: 'Llama-compat', value: 'llama.cpp' },
		{ label: 'OpenAI-compat', value: 'openai' }
	];

	let protocol = $state<BackendProtocol>('llama.cpp');
</script>

<div class="space-y-3">
	<p class="text-sm text-muted-foreground">Endpoints this UI can talk to.</p>

	<Tabs.Root
		class="gap-3"
		onValueChange={(value) => (protocol = value as BackendProtocol)}
		value={protocol}
	>
		<Tabs.List>
			{#each PROTOCOL_TABS as protocolTab (protocolTab.value)}
				<Tabs.Trigger value={protocolTab.value}>{protocolTab.label}</Tabs.Trigger>
			{/each}
		</Tabs.List>

		{#each PROTOCOL_TABS as protocolTab (protocolTab.value)}
			<Tabs.Content value={protocolTab.value}>
				<SettingsBackends protocol={protocolTab.value} />
			</Tabs.Content>
		{/each}
	</Tabs.Root>
</div>
