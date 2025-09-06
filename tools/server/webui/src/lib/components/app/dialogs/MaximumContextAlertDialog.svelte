<script lang="ts">
	import { AlertTriangle } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { maxContextError, clearMaxContextError } from '$lib/stores/chat.svelte';
</script>

<AlertDialog.Root
	open={maxContextError() !== null}
	onOpenChange={(open) => !open && clearMaxContextError()}
>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title class="flex items-center gap-2">
				<AlertTriangle class="h-5 w-5 text-destructive" />

				Message Too Long
			</AlertDialog.Title>

			<AlertDialog.Description>
				Your message exceeds the model's context window and cannot be processed.
			</AlertDialog.Description>
		</AlertDialog.Header>

		{#if maxContextError()}
			<div class="space-y-3 text-sm">
				<div class="rounded-lg bg-muted p-3">
					<div class="mb-2 font-medium">Token Usage:</div>

					<div class="space-y-1 text-muted-foreground">
						<div>
							Estimated tokens:

							<span class="font-mono">
								{maxContextError()?.estimatedTokens.toLocaleString()}
							</span>
						</div>

						<div>
							Context window:

							<span class="font-mono">
								{maxContextError()?.maxContext.toLocaleString()}
							</span>
						</div>
					</div>
				</div>

				<div>
					<div class="mb-2 font-medium">Suggestions:</div>

					<ul class="list-inside list-disc space-y-1 text-muted-foreground">
						<li>Shorten your message</li>

						<li>Remove some file attachments</li>

						<li>Start a new conversation</li>
					</ul>
				</div>
			</div>
		{/if}

		<AlertDialog.Footer>
			<AlertDialog.Action onclick={() => clearMaxContextError()}>Got it</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
