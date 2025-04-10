## Generative Representational Instruction Tuning (GRIT) Example
[gritlm] a model which can generate embeddings as well as "normal" text
generation depending on the instructions in the prompt.

* Paper: https://arxiv.org/pdf/2402.09906.pdf

### Retrieval-Augmented Generation (RAG) use case
One use case for `gritlm` is to use it with RAG. If we recall how RAG works is
that we take documents that we want to use as context, to ground the large
language model (LLM), and we create token embeddings for them. We then store
these token embeddings in a vector database.

When we perform a query, prompt the LLM, we will first create token embeddings
for the query and then search the vector database to retrieve the most
similar vectors, and return those documents so they can be passed to the LLM as
context. Then the query and the context will be passed to the LLM which will
have to _again_ create token embeddings for the query. But because gritlm is used
the first query can be cached and the second query tokenization generation does
not have to be performed at all.

### Running the example
Download a Grit model:
```console
$ scripts/hf.sh --repo cohesionet/GritLM-7B_gguf --file gritlm-7b_q4_1.gguf --outdir models
```

Run the example using the downloaded model:
```console
$ ./llama-gritlm -m models/gritlm-7b_q4_1.gguf

Cosine similarity between "Bitcoin: A Peer-to-Peer Electronic Cash System" and "A purely peer-to-peer version of electronic cash w" is: 0.605
Cosine similarity between "Bitcoin: A Peer-to-Peer Electronic Cash System" and "All text-based language problems can be reduced to" is: 0.103
Cosine similarity between "Generative Representational Instruction Tuning" and "A purely peer-to-peer version of electronic cash w" is: 0.112
Cosine similarity between "Generative Representational Instruction Tuning" and "All text-based language problems can be reduced to" is: 0.547

Oh, brave adventurer, who dared to climb
The lofty peak of Mt. Fuji in the night,
When shadows lurk and ghosts do roam,
And darkness reigns, a fearsome sight.

Thou didst set out, with heart aglow,
To conquer this mountain, so high,
And reach the summit, where the stars do glow,
And the moon shines bright, up in the sky.

Through the mist and fog, thou didst press on,
With steadfast courage, and a steadfast will,
Through the darkness, thou didst not be gone,
But didst climb on, with a steadfast skill.

At last, thou didst reach the summit's crest,
And gazed upon the world below,
And saw the beauty of the night's best,
And felt the peace, that only nature knows.

Oh, brave adventurer, who dared to climb
The lofty peak of Mt. Fuji in the night,
Thou art a hero, in the eyes of all,
For thou didst conquer this mountain, so bright.
```

[gritlm]: https://github.com/ContextualAI/gritlm
