<script lang="ts">
	import { skillProviderLabel, toggleSkillProviderExclusion } from './skill-presentation';
	import { ListFilter } from '@lucide/svelte';
	import { SearchInput } from '$lib/components/app/forms';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import * as Popover from '$lib/components/ui/popover';
	import { debounce } from '$lib/utils/debounce';

	interface Props {
		providers: readonly string[];
		excludedProviders: ReadonlySet<string>;
		includeProject: boolean;
		value?: string;
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
		providers,
		value = ''
	}: Props = $props();

	const SEARCH_DEBOUNCE_MS = 150;
	const dispatchQuery = debounce((query: string) => onQueryChange(query), SEARCH_DEBOUNCE_MS);

	function toggleProvider(provider: string, included: boolean) {
		onProvidersChange(toggleSkillProviderExclusion(excludedProviders, provider, included));
	}

	function resetFilters() {
		onProvidersChange(new Set());
		onIncludeProjectChange(true);
	}

	const isFilterActive = $derived(excludedProviders.size > 0 || !includeProject);
</script>

<div class="flex items-center gap-2">
	<label for="skills-search" class="sr-only">Search skills</label>
	<SearchInput
		class="flex-1"
		id="skills-search"
		{value}
		placeholder="Search skills..."
		onInput={dispatchQuery}
	/>

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
					<ListFilter class="size-4" />
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
								{skillProviderLabel(provider)}
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
