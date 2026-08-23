<script lang="ts">
	import { Filter, Search } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Input } from '$lib/components/ui/input';
	import * as Popover from '$lib/components/ui/popover';
	import { debounce } from '$lib/utils/debounce';

	interface Props {
		providers: readonly string[];
		excludedProviders: ReadonlySet<string>;
		includeProject: boolean;
		onQueryChange: (query: string) => void;
		onProvidersChange: (excludedProviders: ReadonlySet<string>) => void;
		onIncludeProjectChange: (includeProject: boolean) => void;
	}

	let {
		excludedProviders,
		includeProject,
		onIncludeProjectChange,
		onProvidersChange,
		onQueryChange,
		providers
	}: Props = $props();

	const SEARCH_DEBOUNCE_MS = 150;
	const dispatchQuery = debounce((query: string) => onQueryChange(query), SEARCH_DEBOUNCE_MS);

	function handleInput(event: Event & { currentTarget: HTMLInputElement }) {
		dispatchQuery(event.currentTarget.value);
	}

	function toggleProvider(provider: string, included: boolean) {
		const next = new Set(excludedProviders);

		if (included) next.delete(provider);
		else next.add(provider);

		onProvidersChange(next);
	}

	function resetFilters() {
		onProvidersChange(new Set());
		onIncludeProjectChange(true);
	}

	const isFilterActive = $derived(excludedProviders.size > 0 || !includeProject);
</script>

<div class="flex items-center gap-2">
	<div class="relative flex-1">
		<Search
			class="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
			aria-hidden="true"
		/>
		<Input
			type="search"
			placeholder="Search skills..."
			aria-label="Search skills"
			class="pl-8"
			oninput={handleInput}
		/>
	</div>

	<Popover.Root>
		<Popover.Trigger>
			{#snippet child({ props })}
				<Button
					{...props}
					variant="outline"
					size="icon"
					class="relative shrink-0"
					aria-label="Filter skills"
				>
					<Filter class="size-4" />
					{#if isFilterActive}
						<span
							data-testid="skill-filter-active-dot"
							class="absolute -right-0.5 -top-0.5 size-2 rounded-full bg-primary"
							aria-hidden="true"
						></span>
					{/if}
				</Button>
			{/snippet}
		</Popover.Trigger>

		<Popover.Content class="w-64 p-3" align="end">
			<div class="flex flex-col gap-3">
				{#if providers.length > 0}
					<div class="flex flex-col gap-2">
						<span class="text-xs font-medium text-muted-foreground">Providers</span>
						{#each providers as provider (provider)}
							<label class="flex items-center gap-2 text-sm">
								<Checkbox
									checked={!excludedProviders.has(provider)}
									onCheckedChange={(checked) => toggleProvider(provider, checked === true)}
								/>
								{provider}
							</label>
						{/each}
					</div>
				{/if}

				<div class="flex flex-col gap-2 border-t pt-2">
					<label class="flex items-center gap-2 text-sm">
						<Checkbox
							checked={includeProject}
							onCheckedChange={(checked) => onIncludeProjectChange(checked === true)}
						/>
						Include project skills
					</label>
				</div>

				<Button
					variant="ghost"
					size="sm"
					class="w-fit"
					disabled={!isFilterActive}
					onclick={resetFilters}
				>
					Reset
				</Button>
			</div>
		</Popover.Content>
	</Popover.Root>
</div>
