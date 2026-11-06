<!--
	The chapter navigation: the page's section heads as a list fixed at the right, centred. Inactive chapters are small
	and dimmed; the one being read is in the full foreground colour and a little larger. Clicking a chapter scrolls to
	it. Shown only where there is room beside the text.
-->
<script lang="ts">
	let { root }: { root: HTMLElement | undefined } = $props();

	let chapters: { el: HTMLElement; id: string; n: string; title: string }[] = $state([]);
	let active = $state(0);

	// runs once the page it navigates is mounted
	$effect(() => {
		if (!root) return;

		const list = [...root.querySelectorAll<HTMLElement>(':scope > h2')].map((el, i) => {
			const n = String(i + 1).padStart(2, '0');

			el.id ||= `chapter-${n}`;

			return { el, id: el.id, n, title: el.textContent?.trim() ?? '' };
		});

		chapters = list;

		// the active chapter is the last one whose head has passed 40% of the way down the screen
		let frame = 0;

		const update = () => {
			frame = 0;
			const line = window.innerHeight * 0.4;
			const index = list.findLastIndex(({ el }) => el.getBoundingClientRect().top <= line);

			active = Math.max(0, index);
		};
		const onScroll = () => (frame ||= requestAnimationFrame(update));

		update();
		window.addEventListener('scroll', onScroll, { passive: true });
		window.addEventListener('resize', onScroll);

		return () => {
			cancelAnimationFrame(frame);
			window.removeEventListener('scroll', onScroll);
			window.removeEventListener('resize', onScroll);
		};
	});

	function go(event: MouseEvent, el: HTMLElement) {
		event.preventDefault();
		const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

		window.scrollTo({
			behavior: reduce ? 'auto' : 'smooth',
			top: el.getBoundingClientRect().top + window.scrollY - 32
		});
	}
</script>

{#if chapters.length}
	<nav aria-label="Chapters" class="chapters">
		<ol>
			{#each chapters as chapter, i (chapter.id)}
				<li>
					<a
						aria-current={i === active ? 'location' : undefined}
						class={{ active: i === active }}
						href="#{chapter.id}"
						onclick={(event) => go(event, chapter.el)}
					>
						<span class="n">{chapter.n}</span>

						<span class="title">{chapter.title}</span>
					</a>
				</li>
			{/each}
		</ol>
	</nav>
{/if}

<style>
	.chapters {
		--mono: 'Llama Mono', ui-monospace, monospace;
		position: fixed;
		top: 50%;
		right: clamp(16px, 2.5vw, 40px);
		z-index: 10;
		display: none;
		width: 13rem;
		transform: translateY(-50%);
	}

	/* only where the column leaves room for it */
	@media (width >= 1280px) {
		.chapters {
			display: block;
		}
	}

	ol {
		display: grid;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	a {
		display: grid;
		grid-template-columns: 1.75rem 1fr;
		align-items: baseline;
		padding-block: 4px;
		color: var(--fg-muted);
		font-size: 12px;
		line-height: 1.3;
		text-decoration: none;
		opacity: 0.6;
		transition:
			color 200ms ease-out,
			font-size 200ms ease-out,
			opacity 200ms ease-out;
	}

	a:hover {
		opacity: 1;
	}

	a.active {
		color: var(--fg-default);
		font-size: 14px;
		font-weight: 500;
		opacity: 1;
	}

	.n {
		font: 500 10px var(--mono);
		letter-spacing: 0.04em;
	}

	a.active .n {
		color: var(--art-cinder);
	}

	a:focus-visible {
		border-radius: 4px;
		outline: 2px solid var(--line-focus);
		outline-offset: 2px;
	}

	@media (prefers-reduced-motion: reduce) {
		a {
			transition: none;
		}
	}
</style>
