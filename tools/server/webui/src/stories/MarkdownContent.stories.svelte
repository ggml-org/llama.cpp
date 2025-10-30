<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MarkdownContent } from '$lib/components/app';
	import { AI_TUTORIAL_MD } from './fixtures/ai-tutorial.js';
	import { API_DOCS_MD } from './fixtures/api-docs.js';
	import { BLOG_POST_MD } from './fixtures/blog-post.js';
	import { DATA_ANALYSIS_MD } from './fixtures/data-analysis.js';
	import { README_MD } from './fixtures/readme.js';
	import { MATH_FORMULAS_MD } from './fixtures/math-formulas.js';
	import { EMPTY_MD } from './fixtures/empty.js';

	const { Story } = defineMeta({
		title: 'Components/MarkdownContent',
		component: MarkdownContent,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story name="Empty" args={{ content: EMPTY_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }} />

<Story
	name="AI Tutorial"
	args={{ content: AI_TUTORIAL_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="API Documentation"
	args={{ content: API_DOCS_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="Technical Blog"
	args={{ content: BLOG_POST_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="Data Analysis"
	args={{ content: DATA_ANALYSIS_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="README file"
	args={{ content: README_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="Math Formulas"
	args={{ content: MATH_FORMULAS_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

<Story
	name="URL Links"
	args={{
		content: `# URL Links Test

Here are some example URLs that should open in new tabs:

- [Hugging Face Homepage](https://huggingface.co)
- [GitHub Repository](https://github.com/ggml-org/llama.cpp)
- [OpenAI Website](https://openai.com)
- [Google Search](https://www.google.com)

You can also test inline links like https://example.com or https://docs.python.org.

All links should have \`target="_blank"\` and \`rel="noopener noreferrer"\` attributes for security.`,
		class: 'max-w-[56rem] w-[calc(100vw-2rem)]'
	}}
	play={async ({ canvasElement }) => {
		const { expect } = await import('storybook/internal/test');
		
		// Wait for component to render
		await new Promise(resolve => setTimeout(resolve, 100));
		
		// Find all links in the rendered content
		const links = canvasElement.querySelectorAll('a[href]');
		
		// Test that we have the expected number of links
		expect(links.length).toBeGreaterThan(0);
		
		// Test each link for proper attributes
		links.forEach((link) => {
			const href = link.getAttribute('href');
			
			// Test that external links have proper security attributes
			if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
				expect(link.getAttribute('target')).toBe('_blank');
				expect(link.getAttribute('rel')).toBe('noopener noreferrer');
			}
		});
		
		// Test specific links exist
		const hugginFaceLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://huggingface.co'
		);
		expect(hugginFaceLink).toBeTruthy();
		expect(hugginFaceLink?.textContent).toBe('Hugging Face Homepage');
		
		const githubLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://github.com/ggml-org/llama.cpp'
		);
		expect(githubLink).toBeTruthy();
		expect(githubLink?.textContent).toBe('GitHub Repository');
		
		const openaiLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://openai.com'
		);
		expect(openaiLink).toBeTruthy();
		expect(openaiLink?.textContent).toBe('OpenAI Website');
		
		const googleLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://www.google.com'
		);
		expect(googleLink).toBeTruthy();
		expect(googleLink?.textContent).toBe('Google Search');
		
		// Test inline links (auto-linked URLs)
		const exampleLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://example.com'
		);
		expect(exampleLink).toBeTruthy();
		
		const pythonDocsLink = Array.from(links).find(link => 
			link.getAttribute('href') === 'https://docs.python.org'
		);
		expect(pythonDocsLink).toBeTruthy();
		
		console.log(`✅ URL Links test passed - Found ${links.length} links with proper attributes`);
	}}
/>
