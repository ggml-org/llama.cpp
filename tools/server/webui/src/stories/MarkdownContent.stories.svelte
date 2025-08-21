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

<Story
    name="Empty"
    args={{ content: EMPTY_MD, class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
/>

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
