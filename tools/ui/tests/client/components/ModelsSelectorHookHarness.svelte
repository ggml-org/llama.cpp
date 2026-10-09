<script lang="ts">
	import { useModelsSelector } from '$lib/hooks/use-models-selector.svelte';

	interface Props {
		currentModel?: () => string | null;
	}

	let { currentModel = () => null }: Props = $props();

	// the caller seeds modelsStore.models before mounting, so the hook's own
	// fetch() short-circuits on an already populated list; the hook keeps the getter
	// and calls it later, so reading the prop here is intended
	// svelte-ignore state_referenced_locally
	const selector = useModelsSelector({ currentModel, onOpenChange: () => {} });

	/** Every model id the selector lists, in the order the sections render them. */
	export function listedIds(): string[] {
		return [
			...selector.favoriteItems.map((item) => item.option.id),
			...selector.loadedItems.map((item) => item.option.id),
			...selector.groupedFilteredOptions.available.flatMap((group) =>
				group.items.map((item) => item.option.id)
			)
		];
	}

	export function shownModelId(): string | undefined {
		return selector.getDisplayOption()?.id;
	}

	export function isEmpty(): boolean {
		return selector.isEmpty;
	}
</script>

<span data-testid="selector-hook"></span>
