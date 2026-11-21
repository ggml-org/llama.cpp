// Diagrams drawn with Pele, as renderMermaidDiagrams() draws them, for the preview stories
import { render } from 'pele';

export const FLOWCHART_SVG = render(`flowchart LR
    U[Browser] -->|POST /v1/chat/completions| S(llama-server)
    S --> Q{Slot free?}
    Q -->|yes| D[Decode]
    Q -->|no| W[Queue]
    W --> D
    D -->|SSE tokens| U`).svg;

// wider than any screen, so fitting it shrinks it
export const PIPELINE_SVG = render(`flowchart LR
    A[Download GGUF] --> B[Read metadata] --> C[Load tensors] --> D[Build graph]
    D --> E[Tokenize prompt] --> F[Prompt processing] --> G[Sample token]
    G --> H[Detokenize] --> I[Stream to client] --> J{Stop?}
    J -->|no| G
    J -->|yes| K[Free slot]`).svg;

// small enough that fitting it grows it, up to the fit cap
export const SMALL_SVG = render(`flowchart TD
    A[Prompt] --> B[Answer]`).svg;

export const SEQUENCE_SVG = render(`sequenceDiagram
    participant C as Llama Chat
    participant S as llama-server
    C->>S: POST /v1/chat/completions
    S-->>C: data: {"choices":[...]}
    S-->>C: data: [DONE]
    Note over C,S: the stream ends with DONE`).svg;
