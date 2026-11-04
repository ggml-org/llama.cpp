// .svx pages (mdsvex) import as Svelte components
declare module '*.svx' {
	import type { Component } from 'svelte';
	const page: Component;
	export default page;
}
