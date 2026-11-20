<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Card from '@ggml-org/llama-ui/ui/card';

	let { needRefresh: needRefreshProp, updateServiceWorker } = $props();
	let needRefresh = $derived(needRefreshProp ?? false);
</script>

{#if needRefresh}
	<Card.Root class="overflow-hidden gap-1 py-5">
		<Card.Header class="px-5">
			<Card.Title class="text-label">Update available</Card.Title>
		</Card.Header>

		<Card.Content class="gap-6 grid px-5">
			<p class="text-caption text-muted-foreground">
				A new version is available. Reload to update.
			</p>

			<Button
				class="justify-self-end-safe"
				onclick={() => {
					void updateServiceWorker();
					needRefresh = false;
				}}
				size="sm"
			>
				Reload
			</Button>
		</Card.Content>
	</Card.Root>
{/if}
