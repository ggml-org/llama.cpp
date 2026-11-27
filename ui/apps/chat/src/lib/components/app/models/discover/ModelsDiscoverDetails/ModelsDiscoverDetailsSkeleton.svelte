<script lang="ts">
	import { Skeleton } from '@ggml-org/llama-ui/ui/skeleton';

	interface Props {
		/** Leaner pane for a narrow host such as Search: no Hugging Face link, no description. */
		compact?: boolean;
	}

	let { compact = false }: Props = $props();
</script>

<!-- Static skeleton of ModelsDiscoverDetails: header, metadata chips, download options and readme. -->
<div class="@container space-y-6 p-6 pr-0" data-slot="model-details-skeleton">
	<!-- Header: avatar, name with capability icons, base model line, HF link -->
	<div class="space-y-3">
		<div class="flex flex-wrap items-start justify-between gap-3">
			<div class="flex min-w-0 items-center gap-2">
				<Skeleton class="size-12 rounded-md" />

				<div class="min-w-0 space-y-1.5">
					<Skeleton class="h-6 w-64 max-w-full" />

					<Skeleton class="h-3.5 w-44 max-w-full" />
				</div>
			</div>

			{#if !compact}
				<Skeleton class="h-7.5 w-44 shrink-0 rounded-md" />
			{/if}
		</div>

		<!-- downloads / likes / last updated -->
		<div class="flex h-5 items-center gap-4">
			<Skeleton class="h-3.5 w-16" />

			<Skeleton class="h-3.5 w-12" />

			<Skeleton class="h-3.5 w-24" />
		</div>

		<!-- catalog family description, else the HF card description: it wraps to a
		     second line in a narrow host -->
		{#if !compact}
			<div class="space-y-2 py-0.75">
				<Skeleton class="h-3.5 w-full" />

				<Skeleton class="hidden h-3.5 w-2/5 @max-2xl:block" />
			</div>
		{/if}

		<!-- metadata chips, as wide as the usual four so they wrap where those do -->
		<div class="space-y-3">
			<div class="flex flex-wrap items-center gap-1.5">
				<Skeleton class="h-6.5 w-45 rounded-md" />

				<Skeleton class="h-6.5 w-36 rounded-md" />

				<Skeleton class="h-6.5 w-47 rounded-md" />

				<Skeleton class="h-6.5 w-30 rounded-md" />
			</div>
		</div>
	</div>

	<!-- Download options: one row per bit depth, its quant chips on the right -->
	<div class="rounded-3xl bg-muted/60 elevated-control">
		<div class="flex w-full flex-col divide-y divide-border/50 px-4 pb-1 dark:divide-border/35">
			{#each [0, 1, 2] as _, index (index)}
				<div
					class="grid grid-cols-[5rem_1fr] items-center gap-3 py-3 @max-lg:grid-cols-1 @max-lg:gap-1.5"
				>
					<div class="space-y-1">
						<Skeleton class="h-5 w-12" />

						<Skeleton class="h-3 w-28" />
					</div>

					<div class="flex flex-wrap items-center justify-end gap-1.5 @max-lg:justify-start">
						<Skeleton class="h-6.5 w-30 rounded-md" />

						{#if index === 0}
							<Skeleton class="h-6.5 w-30 rounded-md" />
						{/if}
					</div>
				</div>
			{/each}
		</div>
	</div>

	<!-- Readme -->
	<div class="p-3 rounded-3xl bg-muted/60 elevated-control">
		<div class="space-y-2.5">
			<!-- the card's title, then its first paragraph -->
			<Skeleton class="mt-1 mb-4 h-7 w-1/3" />

			<Skeleton class="h-3.5 w-full" />

			<Skeleton class="h-3.5 w-full" />

			<Skeleton class="h-3.5 w-5/6" />

			<Skeleton class="h-3.5 w-2/3" />
		</div>
	</div>
</div>
