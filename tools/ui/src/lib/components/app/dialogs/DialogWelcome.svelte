<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Label } from '$lib/components/ui/label';
	import * as RadioGroup from '$lib/components/ui/radio-group';
	import { UI_PRESET_ORDER, UI_PRESETS } from '$lib/constants';
	import { UiPreset } from '$lib/enums';
	import { settingsStore } from '$lib/stores';
	import { isUiPreset } from '$lib/utils';

	interface Props {
		open: boolean;
	}

	let { open }: Props = $props();

	let selected = $state<string>(UiPreset.MINIMAL);

	function handleSubmit(event: SubmitEvent) {
		event.preventDefault();

		if (isUiPreset(selected)) {
			settingsStore.applyUiPreset(selected);
		}
	}
</script>

<!-- a pick is required: the dialog closes only once a preset is stored -->
<Dialog.Root {open}>
	<Dialog.Content
		class="max-w-xl! p-4"
		escapeKeydownBehavior="ignore"
		interactOutsideBehavior="ignore"
	>
		<Dialog.Header showCloseButton={false}>
			<Dialog.Title class="select-none">Welcome to llama.cpp</Dialog.Title>

			<Dialog.Description>
				Pick the features to start with. You can change this later in Settings &gt; General.
			</Dialog.Description>
		</Dialog.Header>

		<form class="contents" onsubmit={handleSubmit}>
			<RadioGroup.Root bind:value={selected} class="gap-3">
				{#each UI_PRESET_ORDER as preset (preset)}
					{@const info = UI_PRESETS[preset]}
					{@const IconComponent = info.icon}

					<Label
						class={[
							'flex cursor-pointer items-start gap-3 rounded-lg border p-4 font-normal transition-colors hover:bg-muted/50',
							selected === preset ? 'border-primary/40 bg-muted/30' : 'border-border/50'
						]}
						for="ui-preset-{preset}"
					>
						<RadioGroup.Item class="mt-0.5" id="ui-preset-{preset}" value={preset} />

						<div class="min-w-0 flex-1 space-y-1">
							<div class="flex items-center gap-2 font-medium">
								<IconComponent class="h-4 w-4" />

								{info.label}
							</div>

							<p class="text-xs text-muted-foreground">{info.description}</p>

							<ul class="list-disc pl-4 text-xs text-muted-foreground">
								{#each info.features as feature (feature)}
									<li>{feature}</li>
								{/each}
							</ul>
						</div>
					</Label>
				{/each}
			</RadioGroup.Root>

			<Dialog.Footer>
				<Button size="sm" type="submit">Continue</Button>
			</Dialog.Footer>
		</form>
	</Dialog.Content>
</Dialog.Root>
