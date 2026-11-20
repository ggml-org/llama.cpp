// Mermaid blocks of several diagram types, and one that can't be drawn. The flowchart's
// click directives must leave it inert.
const fence = '```';

export const MERMAID_DIAGRAMS_MD = String.raw`
How a request reaches the model:

${fence}mermaid
flowchart LR
    U[Browser] -->|POST /v1/chat/completions| S(llama-server)
    S --> Q{Slot free?}
    Q -->|yes| D[Decode]
    Q -->|no| W[Queue]
    W --> D
    D -->|SSE tokens| U
    click S href "https://github.com/ggml-org/llama.cpp" "llama.cpp on GitHub" _blank
    click D call alert(1)
${fence}

The same exchange over time:

${fence}mermaid
sequenceDiagram
    participant C as Llama Chat
    participant S as llama-server
    C->>S: POST /v1/chat/completions
    S-->>C: data: {"choices":[...]}
    S-->>C: data: [DONE]
    Note over C,S: the stream ends with DONE
${fence}

A slot's life:

${fence}mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Processing: prompt
    Processing --> Generating: prompt evaluated
    Generating --> Idle: stop or limit
${fence}

Where the time goes:

${fence}mermaid
pie title Time per request
    "Prompt processing" : 35
    "Token generation" : 60
    "Sampling" : 5
${fence}

A diagram with a mistake in it:

${fence}mermaid
flowchart LR
    A -->
${fence}
`;
