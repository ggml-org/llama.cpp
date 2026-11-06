<!--
	Communications from the real templates in Brand › Communications, scaled to fit: a model drop post on light, and
	the two link previews, llama.app's on the day sky and llama.cpp's on dark; their copy is the templates' own
-->
<script lang="ts">
	import LinkPreview from '../../brand/communications/posts/LinkPreview.svelte';
	import ModelDropPost from '../../brand/communications/posts/ModelDropPost.svelte';
	import { linkPreview, modelDrop } from '../../brand/communications/posts/post';
</script>

<div class="posts">
	<div class="post tall rounded-card elevated-surface">
		<ModelDropPost copy={modelDrop} format="portrait" tone="light" />
	</div>

	<div class="stack">
		<div class="post rounded-card elevated-surface">
			<LinkPreview copy={linkPreview} kind="og" tone="day" />
		</div>

		<div class="post rounded-card elevated-surface">
			<LinkPreview copy={linkPreview} kind="repo" tone="dark" />
		</div>
	</div>
</div>

<style>
	.posts {
		display: grid;
		grid-template-columns: minmax(0, 4fr) minmax(0, 5fr);
		gap: 16px;
		align-items: start;
	}

	.stack {
		display: grid;
		gap: 16px;
	}

	.post {
		overflow: hidden;
	}

	@media (width < 720px) {
		.posts {
			grid-template-columns: 1fr;
		}
	}
</style>
