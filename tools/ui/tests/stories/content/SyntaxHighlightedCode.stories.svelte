<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SyntaxHighlightedCode from '$lib/components/app/content/SyntaxHighlightedCode.svelte';

	const { Story } = defineMeta({
		component: SyntaxHighlightedCode,
		parameters: {
			layout: 'centered'
		},
		title: 'Content/SyntaxHighlightedCode'
	});

	const code = `fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn main() {
    let a = String::from("borrow");
    println!("{}", longest(&a, "checker"));
}`;
</script>

<Story args={{ class: 'w-[36rem]', code, language: 'rust' }} name="Default" />
