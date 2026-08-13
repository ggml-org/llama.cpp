# Skills flow

> User-facing summary of Agent Skills in llama-ui. The server owns skill
> discovery, parsing, trust, and file access; this UI only presents and
> activates what the server reports.

## Server requirement

Agent Skills are available only when the `llama-server` the UI is connected
to was started with the Skills option enabled (`--skills`). Without it the
server reports no Skills routes:

- the Skills page shows a "Skills are not enabled" state (no retry), and
- the `/skills` command degrades to a "Skills are unavailable on this
  server" notice.

A server that does expose Skills serves a read-only catalog of the skills
it discovered. Skill content and resource files live on the server host;
the UI never reads the host filesystem and never displays host paths or
provider roots.

## Read-only catalog

The sidebar **Skills** entry opens the read-only catalog page for the
conversation's selected working directory. The page shows each skill's
name, description, scope (project/global), provider, instruction facts
(size in bytes/lines and estimated token count), the catalog snapshot
timestamp, a resource count, and any safe server diagnostics.

- The resource count is exact when the server says the listing is
  complete; a truncated listing renders as a lower bound (for example
  "Resources: 3+").
- Loading, empty (server returned no skills), unavailable (Skills not
  enabled), and error (retryable) states are distinct.
- The catalog refreshes when the conversation's working directory
  changes; an in-flight request for the previous directory never replaces
  the new directory's view.

## Budget behavior

**Max skill budget** (settings, default `2000`, a non-negative integer)
limits how much of the catalog is packed into each agentic run's prompt.
The budget applies to the serialized catalog envelope, so the model sees
the skills that fit within the budget in server order.

- A zero budget or an empty catalog injects no skill content and exposes
  no skill tools for that run.
- A budget that covers the whole catalog exposes only the skill reader
  tool; a partial envelope also exposes a skill listing tool that returns
  structured snapshot entries (never raw XML).
- The Settings screen shows "Budget is 0" on the catalog page when the
  budget is zero — distinct from a server that simply has no skills.

## Exact versus estimated counts

Skill instruction sizes are labeled in the catalog. When the server can
tokenize exactly, counts are exact; when it cannot (for example no model
is loaded), the server reports a deterministic byte-based estimate and the
UI labels the value as estimated. The UI never issues a model-loading
request just to count skill text.

## `/skills` command

Typing `/skills` in the chat input and picking the command opens the
catalog page. Typing `/skills <name>` and picking the command asks the
server to read that skill by name; a successful read is treated as
explicit consent and activates the skill for the conversation:

- Activation persists per conversation and survives reload and
  conversation export/import. A later `/skills <name>` for an already
  activated skill reports it is already activated and changes nothing.
- A failed, unavailable, or missing read persists nothing and shows a
  notice ("Skill not found" / "Skills are unavailable on this server").
- The command can be selected before Skills availability is known. If the server reports Skills unavailable, it shows the unavailable notice.

## Model consent

When a model-driven run first reads a skill whose identity the
conversation has not approved, the tool call pauses and the UI asks for
explicit approval, showing only safe identity facts: the skill name,
scope, provider, and the requested resource path when reading a resource.
Allow proceeds and activates the skill through the same durable path as
`/skills <name>`; deny returns a structured result with no skill content
and records no activation. Concurrent reads of the same skill in one
conversation share a single approval decision. A resource read is allowed
only after the exact skill identity is already activated, and each
resource read still executes its own server request.

## Result rendering

Skill and skill-resource tool results render with typed labels ("Skill -
name" / "Skill resource - name" plus provider, scope, and path); the server
XML content stays ordinary text content. Results without valid skill
metadata keep the generic tool rendering.

```mermaid
sequenceDiagram
    participant UI as Skills page / ChatForm
    participant skillsStore as skillsStore
    participant skillsSvc as SkillsService
    participant LS as LocalStorage (settings)
    participant server as llama-server (--skills)

    Note over UI,server: CATALOG (read-only, working-directory scoped)

    UI->>skillsStore: refresh(cwd)
    skillsStore->>skillsSvc: list(cwd)
    Note right of skillsSvc: Sends the selected working directory;<br/>no directory header when none is selected
    skillsSvc->>server: GET /skills
    server-->>skillsSvc: safe catalog records + diagnostics (no paths)
    skillsSvc-->>skillsStore: SkillCatalogResponse
    skillsStore->>skillsStore: store latest CWD slot (monotonic generation)
    skillsStore-->>UI: safe fields (name, description, scope, provider,<br/>instruction facts, estimate label, resource count, diagnostics)

    Note over UI,server: EXPLICIT /skills &lt;name&gt; ACTIVATION

    UI->>ChatForm: pick "/skills &lt;name&gt;"
    ChatForm->>skillsSvc: read(name, cwd)
    skillsSvc->>server: POST /skills/read
    alt successful base read
        server-->>ChatForm: opaque identity + content
        ChatForm->>ChatForm: persist synthetic assistant tool call + paired tool result
        Note right of ChatForm: Typed skill metadata (opaque id, name, scope, provider)<br/>survives reload and export/import
        ChatForm-->>UI: "Skill activated" (or "already activated")
    else missing / failed / unavailable
        server-->>ChatForm: error
        ChatForm-->>UI: notice, nothing persisted
    end

    Note over UI,server: MODEL-DRIVEN CONSENT

    UI->>server: model reads a skill via the run snapshot
    alt identity not yet approved
        server-->>UI: pause; consent card (safe identity facts only)
        UI->>UI: Allow / Deny
        alt Allow
            UI->>UI: durable activation (same path as /skills &lt;name&gt;)
        else Deny
            UI->>UI: structured result, no content, no activation
        end
    else identity already approved
        UI->>UI: proceed without prompting
    end
```
