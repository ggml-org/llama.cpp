<script lang="ts">
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { AlertTriangle } from '@lucide/svelte';
	import { contextError, clearContextError } from '$lib/stores/chat.svelte';
</script>

<!-- Context Length Error Alert Dialog -->
<AlertDialog.Root
	open={contextError() !== null}
	onOpenChange={(open) => !open && clearContextError()}
>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<AlertTriangle class="text-destructive h-5 w-5" />
				Message Too Long
			</AlertDialog.Title>
			<AlertDialog.Description>
				Your message exceeds the model's context window and cannot be processed.
			</AlertDialog.Description>
		</AlertDialog.Header>

		{#if contextError()}
			<div class="space-y-3 text-sm">
				<div class="bg-muted rounded-lg p-3">
					<div class="mb-2 font-medium">Token Usage:</div>
					<div class="text-muted-foreground space-y-1">
						<div>
							Estimated tokens: <span class="font-mono"
								>{contextError()?.estimatedTokens.toLocaleString()}</span
							>
						</div>
						<div>
							Maximum allowed: <span class="font-mono"
								>{contextError()?.maxAllowed.toLocaleString()}</span
							>
						</div>
						<div>
							Context window: <span class="font-mono"
								>{contextError()?.maxContext.toLocaleString()}</span
							>
						</div>
					</div>
				</div>

				<div>
					<div class="mb-2 font-medium">Suggestions:</div>
					<ul class="text-muted-foreground list-inside list-disc space-y-1">
						<li>Shorten your message</li>
						<li>Remove some file attachments</li>
						<li>Start a new conversation</li>
					</ul>
				</div>
			</div>
		{/if}

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => clearContextError()}>Got it</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
