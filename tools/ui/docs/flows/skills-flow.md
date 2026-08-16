```mermaid
sequenceDiagram
    participant UI as Skills page / ChatForm
    participant Nav as Sidebar navigation
    participant skillsStore as skillsStore
    participant activationStore as skillActivationStore
    participant agenticStore as agenticStore
    participant toolsStore as toolsStore
    participant SkillsSvc as SkillsService
    participant CmdSvc as SkillCommandService
    participant PackingSvc as SkillsPackingService
    participant Tokenize as POST /tokenize
    participant ChatSvc as ChatService
    participant DbSvc as DatabaseService
    participant server as llama-server (--skills)

    Note over skillsStore: State:<br/>CWD-keyed catalog slots<br/>monotonic request generations<br/>availability: unknown/loading/available/disabled/error<br/>shared catalog requests per CWD
    Note over activationStore: Durable base activations<br/>conversation + opaque server identity<br/>resource approvals are session-scoped

    %% =========================================================================
    Note over UI,server: STARTUP AND SKILLS AVAILABILITY
    %% =========================================================================

    Nav->>skillsStore: probeAvailability(cwd)
    activate skillsStore
    skillsStore->>skillsStore: ensureCatalog(cwd)
    Note right of skillsStore: Concurrent probe and route callers<br/>share one request, each caller<br/>keeps an independent abort signal
    skillsStore->>SkillsSvc: list(cwd)
    SkillsSvc->>server: GET /skills<br/>X-Skill-Cwd when a CWD is selected
    alt Skills disabled
        server-->>SkillsSvc: 404 route not registered
        SkillsSvc-->>skillsStore: ApiError(404)
        skillsStore->>skillsStore: availability = disabled
        skillsStore-->>Nav: hide Skills entry
    else Skills enabled and catalog request succeeds
        server-->>SkillsSvc: SkillCatalogResponse
        SkillsSvc-->>skillsStore: catalog
        skillsStore->>skillsStore: store ready slot for CWD
        skillsStore->>skillsStore: availability = available
        skillsStore-->>Nav: show Skills entry
    else Skills enabled but request fails
        server-->>SkillsSvc: error envelope
        SkillsSvc-->>skillsStore: ApiError
        skillsStore->>skillsStore: store error slot
        skillsStore->>skillsStore: availability = error
        skillsStore-->>Nav: show Skills entry with retry
    end
    deactivate skillsStore

    %% =========================================================================
    Note over UI,server: CATALOG LOAD AND CWD CHANGES
    %% =========================================================================

    UI->>UI: derive cwd from active conversation or pending CWD
    UI->>skillsStore: ensureCatalog(cwd)
    activate skillsStore
    skillsStore->>SkillsSvc: list(cwd)
    SkillsSvc->>server: GET /skills
    activate server
    server->>server: resolve effective CWD
    Note right of server: Absent header uses server process CWD,<br/>invalid explicit CWD returns 400
    server->>server: discover trusted project/global roots
    server->>server: canonicalize and contain candidates
    server->>server: parse bounded UTF-8 SKILL.md files
    server->>server: validate names and descriptions
    server->>server: retain first entry per name in precedence order
    server->>server: measure instruction bytes, lines, tokens
    server->>server: enumerate bounded resource metadata
    server-->>SkillsSvc: safe entries + diagnostics + opaque catalog XML
    deactivate server
    SkillsSvc-->>skillsStore: SkillCatalogResponse
    skillsStore->>skillsStore: accept only latest generation for cwd
    skillsStore-->>UI: safe catalog fields
    Note right of UI: name, description, scope, provider,<br/>instruction facts, estimate label,<br/>timestamp, resource count, diagnostics
    Note right of UI: Diagnostics render as separate rows with explicit<br/>Skill, Scope, and Provider labels when returned,<br/>duplicate diagnostic codes remain distinct
    deactivate skillsStore

    alt CWD changes while a request is in flight
        UI->>skillsStore: invalidate(previousCwd)
        skillsStore->>skillsStore: bump generation and clear old slot
        UI->>skillsStore: ensureCatalog(newCwd)
        skillsStore->>SkillsSvc: list(newCwd)
        SkillsSvc->>server: GET /skills with new CWD
        server-->>SkillsSvc: new CWD catalog
        SkillsSvc-->>skillsStore: new catalog
        skillsStore-->>UI: render new CWD only
        Note over skillsStore: Late old response may resolve for its caller<br/>but cannot replace the new screen slot
    end

    alt Catalog state
        UI-->>UI: loading skeleton
    else Empty catalog
        UI-->>UI: empty state
    else Unavailable catalog
        UI-->>UI: Skills unavailable state, no retry
    else Retryable error
        UI-->>UI: error state and retry action
    else Ready catalog
        UI-->>UI: catalog list and diagnostics
    end

    %% =========================================================================
    Note over UI,server: READ-ONLY CATALOG PREVIEW
    %% =========================================================================

    UI->>UI: select catalog entry
    UI->>SkillsSvc: read({name}, cwd)
        Note right of SkillsSvc: Sends X-Skill-Cwd when `cwd` is selected,<br/>absent header uses the server process CWD
    SkillsSvc->>server: POST /skills/read<br/>{name}
    server->>server: re-resolve name for effective CWD
    alt Base skill read
        server-->>SkillsSvc: kind=skill, metadata, resources,<br/>source, body_markdown, opaque content_xml
        SkillsSvc-->>UI: SkillBaseReadResult
        UI->>UI: render Markdown or complete Raw source
        UI->>UI: group safe resource paths for display
        Note right of UI: Preview is browsing only,<br/>no consent and no activation persistence
    else Missing, invalid, unavailable, or server error
        server-->>SkillsSvc: shared error envelope
        SkillsSvc-->>UI: preview error
    else Unexpected resource result
        server-->>SkillsSvc: kind=resource
        SkillsSvc-->>UI: reject as preview error
    end
    Note over UI: Selecting another skill, changing CWD, retrying,<br/>or unmounting aborts the prior read, stale results do not render

    %% =========================================================================
    Note over UI,PackingSvc: CATALOG BUDGET STATUS
    %% =========================================================================

    UI->>PackingSvc: buildSkillRunSnapshot(cwd, catalog)
    UI->>PackingSvc: pack(snapshot, {budget, ...packOptions})
        Note right of UI: budget = maxSkillBudget from settings,<br/>packOptions = resolveSkillPackOptions(model,<br/>router mode, loaded check)
        Note right of PackingSvc: Same tokenizer/estimate policy as the<br/>agentic run path below
    PackingSvc-->>UI: SkillPackedCatalog | pack error
    UI->>UI: render dismissible SkillBudgetStatus banner
        Note right of UI: Budget status only; no consent and<br/>no activation persistence

    %% =========================================================================
    Note over UI,server: EXPLICIT /skills AND /skills name argument
    %% =========================================================================

    UI->>UI: type /skills in ChatForm
    UI->>UI: command picker uses prefix/substring matches<br/>from ready catalog names
        Note right of UI: The picker reads the ready catalog slot,<br/>it does not fetch or resolve names
    alt Select /skills without an argument
        UI->>UI: goto /skills
        UI->>skillsStore: route refresh for current CWD
    else Select /skills name argument
        UI->>CmdSvc: dispatchSkillActivation(name, cwd)
            Note right of CmdSvc: SkillCommandService entry point.<br/>Resolves the base read, then routes<br/>through the shared durable activation path.<br/>Success wakes the agentic loop as an<br/>assistant turn (never a system-tagged<br/>user message).
        CmdSvc->>SkillsSvc: read({name}, cwd)
            Note right of SkillsSvc: Sends the active conversation CWD<br/>through X-Skill-Cwd when selected
        SkillsSvc->>server: POST /skills/read<br/>{name}
        alt Base read succeeds
            server-->>SkillsSvc: SkillBaseReadResult with opaque identity
            SkillsSvc-->>CmdSvc: resolved base result
            CmdSvc->>activationStore: loadConversation(conversationId)
            activationStore->>DbSvc: read conversation messages
            DbSvc-->>activationStore: persisted typed skill metadata
            activationStore->>activationStore: reconstruct base identities
            alt Identity already activated
                activationStore-->>CmdSvc: created = false
                CmdSvc-->>UI: created = false
                UI-->>UI: info toast: already activated
            else New identity
                CmdSvc->>activationStore: recordActivation(base result, no tool call id)
                activationStore->>DbSvc: persist synthetic assistant tool call<br/>and paired tool result
                DbSvc-->>activationStore: persisted messages
                activationStore->>CmdSvc: mirror active messages and remember identity
                CmdSvc-->>UI: created = true
                UI-->>UI: success toast: activated
            end
        else Missing or failed read
            server-->>SkillsSvc: 404 or error envelope
            SkillsSvc-->>CmdSvc: failed read
            CmdSvc-->>UI: failed command
            UI-->>UI: not-found or unavailable notice
            Note over activationStore: No activation or message is persisted
        end
    end

    %% =========================================================================
    Note over UI,Tokenize: AGENTIC RUN SNAPSHOT AND BUDGET
    %% =========================================================================

    UI->>agenticStore: runAgenticFlow(conversationId, messages, options)
    activate agenticStore
    agenticStore->>agenticStore: read maxSkillBudget
    alt Budget is zero
        agenticStore-->>agenticStore: register no Skills prompt or adapters
    else Budget is positive
        agenticStore->>activationStore: loadConversation(conversationId)
        activationStore->>DbSvc: read persisted conversation metadata
        DbSvc-->>activationStore: typed base activation records
        agenticStore->>skillsStore: createRunSnapshot(cwd)
        skillsStore->>SkillsSvc: list(cwd)
        SkillsSvc->>server: GET /skills
        server-->>SkillsSvc: current catalog response
        SkillsSvc-->>skillsStore: run-owned catalog response
        skillsStore->>PackingSvc: buildSkillRunSnapshot(cwd, catalog)
        PackingSvc->>PackingSvc: deep-copy and freeze entries
        PackingSvc->>PackingSvc: serialize complete catalog envelope, total=N included=N
        PackingSvc-->>agenticStore: immutable SkillRunSnapshot
        agenticStore->>PackingSvc: pack(snapshot, budget, mode)
        alt Direct tokenizer available
            PackingSvc->>Tokenize: POST /tokenize selected model<br/>add_special=false
            Tokenize-->>PackingSvc: exact token count
        else Estimated mode or tokenizer unavailable
            PackingSvc->>PackingSvc: estimate ceil(UTF-8 bytes / 4)
            Note right of PackingSvc: Label estimate, do not select, wake,<br/>load, or retry another model
        end
        PackingSvc->>PackingSvc: measure complete envelope first
        alt Complete envelope fits budget
            PackingSvc-->>agenticStore: complete envelope, read_skill only
        else Only a prefix fits
            PackingSvc->>PackingSvc: keep leading entries in server order
            PackingSvc-->>agenticStore: partial envelope, list_skill + read_skill
        end
        agenticStore->>toolsStore: read enabled Skills tool names
        Note right of toolsStore: Settings-only `skill:<tool>` keys<br/>filter the budget-authorized adapters
        agenticStore->>agenticStore: compare adapter names with existing tools
        Note right of agenticStore: Existing custom, builtin, and MCP names win,<br/>colliding Skills adapters are omitted with diagnostics
        Note right of agenticStore: User-disabled Skills tools are omitted<br/>after budget and collision checks
        agenticStore->>agenticStore: decorate first system message request-locally
    end
    deactivate agenticStore

    %% =========================================================================
    Note over agenticStore,server: MODEL-DRIVEN SKILL TOOLS
    %% =========================================================================

    agenticStore->>ChatSvc: sendMessage(messages, tools, options)
    ChatSvc->>server: POST /v1/chat/completions
    server-->>ChatSvc: streamed content, reasoning, or tool calls
    ChatSvc-->>agenticStore: normalized assistant turn

    alt list_skill tool call
        agenticStore->>agenticStore: return JSON snapshot entries
        Note right of agenticStore: name, description, scope, provider,<br/>never raw XML or identity
        Note right of agenticStore: list_skill returns only safe snapshot facts,<br/>read_skill errors are structured and contain no XML
    else read_skill tool call
        agenticStore->>agenticStore: parse name and optional relative path
        alt Invalid arguments or name outside frozen snapshot
            agenticStore-->>ChatSvc: structured Skills error result
        else Valid snapshot request
            agenticStore->>SkillsSvc: read({name, path?}, snapshot.cwd)
            Note right of SkillsSvc: Sends X-Skill-Cwd from the frozen<br/>snapshot CWD when selected
            SkillsSvc->>server: POST /skills/read<br/>{name, optional path}
            server->>server: resolve current identity and validate containment
            alt Base read
                server-->>SkillsSvc: kind=skill + opaque identity + content_xml
            else Resource read
                server-->>SkillsSvc: kind=resource + opaque identity + content_xml
            else Read failure
                server-->>SkillsSvc: error envelope
                SkillsSvc-->>agenticStore: structured error result
            end
            alt Successful result with identity already activated
                agenticStore->>activationStore: recordActivation(result)
            else Successful result with identity not activated
                agenticStore->>agenticStore: pause tool call for shared consent decision
                agenticStore-->>UI: consent card with name, scope, provider, optional path
                alt User allows
                    UI-->>agenticStore: allow
                    agenticStore->>activationStore: recordActivation(result)
                else User denies or aborts
                    UI-->>agenticStore: deny
                    agenticStore-->>ChatSvc: structured denied result, no content
                    Note over activationStore: Denial persists no activation
                end
            end
            Note over agenticStore: Concurrent reads of one identity share<br/>one pending consent decision
        end
    end

    %% =========================================================================
    Note over activationStore,DbSvc: DURABLE ACTIVATION AND TOOL RESULTS
    %% =========================================================================

    alt Approved new base activation
        activationStore->>DbSvc: persist paired tool result with typed metadata
        Note right of activationStore: Model path anchors result to its assistant<br/>tool call, slash path creates synthetic pair
        DbSvc-->>activationStore: message persisted
        activationStore-->>agenticStore: activationRecorded + message id
        agenticStore->>agenticStore: avoid duplicate tool-result creation
    else Approved resource read or repeated base read
        activationStore->>activationStore: remember session result or dedupe
        activationStore-->>agenticStore: typed metadata extra
        agenticStore->>DbSvc: persist normal tool result with typed extra
    end

    agenticStore->>ChatSvc: append tool result to next model request
    ChatSvc->>server: continue /v1/chat/completions stream
    loop Until no tool calls, abort, or turn limit
        server-->>ChatSvc: next assistant turn
        ChatSvc-->>agenticStore: content or normalized tool calls
        agenticStore->>agenticStore: execute Skills calls and append results
    end
    agenticStore-->>UI: final response and timings

    %% =========================================================================
    Note over UI,DbSvc: RELOAD, EXPORT, AND RESULT PRESENTATION
    %% =========================================================================

    UI->>activationStore: loadConversation(conversationId)
    activationStore->>DbSvc: read persisted messages after reload/import
    DbSvc-->>activationStore: typed SKILL extras
    activationStore->>activationStore: restore only valid base identities
    Note over activationStore: Opaque identity is the key,<br/>host roots, paths, XML, and content are not stored

    UI->>UI: render persisted tool result
    alt Valid typed Skills metadata
        UI->>UI: use Skill or Skill resource title
        Note right of UI: Typed metadata supplies the skill name,<br/>scope, provider, and safe resource path
        UI->>UI: show provider, scope, and safe relative resource path
        UI->>UI: render server XML as ordinary text
    else Missing or malformed metadata
        UI->>UI: use generic tool renderer
    end

    %% =========================================================================
    Note over UI,server: ERRORS AND ABORTS
    %% =========================================================================

    alt User aborts catalog, preview, or agentic request
        UI->>SkillsSvc: AbortSignal.abort()
        SkillsSvc-->>UI: AbortError
        Note over skillsStore,agenticStore: Abort stale work, keep newer CWD,<br/>run snapshot, and conversation state intact
    else Server/API error
        server-->>SkillsSvc: 400, 404, 500, or 503 error envelope
        SkillsSvc-->>UI: typed failure
        UI-->>UI: show unavailable, not-found, retry, or tool error state
        Note over activationStore: Failed, missing, denied, or unavailable reads<br/>never grant durable activation
    end
```
