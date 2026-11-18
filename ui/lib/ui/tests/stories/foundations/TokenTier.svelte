<!-- One tier of colour tokens as swatches, grouped as in its CSS: each with its name and its value (showReference: the
     light and dark values it points at, by primitive name). -->
<script lang="ts">
	import { reference, type TokenGroup } from './tokens';

	let {
		groups,
		intro,
		showReference
	}: { groups: TokenGroup[]; intro: string; showReference: boolean } = $props();
</script>

<div class="w-[min(72rem,calc(100vw-2rem))] space-y-8 p-4">
	<p class="max-w-2xl text-body-sm text-fg-muted">{intro}</p>

	{#each groups as group (group.title)}
		<section>
			<h2 class="mb-3 text-label text-fg-muted first-letter:uppercase">{group.title}</h2>

			<div class="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-4">
				{#each group.tokens as token (token.name)}
					<figure class="m-0 flex flex-col gap-2">
						<div
							style:background-color="var(--{token.name})"
							class="h-14 rounded-control border border-line-default"
						></div>

						<figcaption class="flex flex-col gap-0.5 font-mono text-[11px] leading-4">
							<span class="text-fg-default">--{token.name}</span>

							{#if showReference}
								<span class="truncate text-fg-muted" title={token.light}
									>light: {reference(token.light)}</span
								>

								{#if token.dark}
									<span class="truncate text-fg-muted" title={token.dark}
										>dark: {reference(token.dark)}</span
									>
								{/if}
							{:else}
								<span class="truncate text-fg-muted" title={token.light}>{token.light}</span>
							{/if}
						</figcaption>
					</figure>
				{/each}
			</div>
		</section>
	{/each}
</div>
