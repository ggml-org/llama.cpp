```mermaid
sequenceDiagram
    participant UI as 🧩 ChatSettings
    participant settingsStore as 🗄️ settingsStore
    participant serverStore as 🗄️ serverStore
    participant ParamSvc as ⚙️ ParameterSyncService
    participant LS as 💾 LocalStorage

    Note over settingsStore: State:<br/>config: SettingsConfigType<br/>theme: string ("auto" | "light" | "dark")<br/>isInitialized: boolean<br/>userOverrides: Set&lt;string&gt;

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 🚀 INITIALIZATION
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over settingsStore: Auto-initialized in constructor (browser only)
    settingsStore->>settingsStore: initialize()
    activate settingsStore

    settingsStore->>settingsStore: loadConfig()
    settingsStore->>LS: get("llama-config")
    LS-->>settingsStore: StoredConfig | null

    alt config exists
        settingsStore->>settingsStore: Merge with SETTING_CONFIG_DEFAULT
        Note right of settingsStore: Fill missing keys with defaults
    else no config
        settingsStore->>settingsStore: config = SETTING_CONFIG_DEFAULT
    end

    settingsStore->>LS: get("llama-userOverrides")
    LS-->>settingsStore: string[] | null
    settingsStore->>settingsStore: userOverrides = new Set(data)

    settingsStore->>settingsStore: loadTheme()
    settingsStore->>LS: get("llama-theme")
    LS-->>settingsStore: theme | "auto"

    settingsStore->>settingsStore: isInitialized = true
    deactivate settingsStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 🔄 SYNC WITH SERVER DEFAULTS
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over UI: Triggered from +layout.svelte when serverStore.props loaded
    UI->>settingsStore: syncWithServerDefaults()
    activate settingsStore

    settingsStore->>serverStore: defaultParams
    serverStore-->>settingsStore: {temperature, top_p, top_k, ...}

    loop each SYNCABLE_PARAMETER
        alt key NOT in userOverrides
            settingsStore->>settingsStore: config[key] = serverDefault[key]
            Note right of settingsStore: Non-overridden params adopt server default
        else key in userOverrides
            Note right of settingsStore: Keep user value, skip server default
        end
    end

    alt serverStore.props has webuiSettings
        settingsStore->>settingsStore: Apply webuiSettings from server
        Note right of settingsStore: Server-provided UI settings<br/>(e.g. showRawOutputSwitch)
    end

    settingsStore->>settingsStore: saveConfig()
    deactivate settingsStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: ⚙️ UPDATE CONFIG
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>settingsStore: updateConfig(key, value)
    activate settingsStore
    settingsStore->>settingsStore: config[key] = value

    alt value matches server default for key
        settingsStore->>settingsStore: userOverrides.delete(key)
        Note right of settingsStore: Matches server default, remove override
    else value differs from server default
        settingsStore->>settingsStore: userOverrides.add(key)
        Note right of settingsStore: Mark as user-modified (won't be overwritten)
    end

    settingsStore->>settingsStore: saveConfig()
    settingsStore->>LS: set(CONFIG_LOCALSTORAGE_KEY, config)
    settingsStore->>LS: set(USER_OVERRIDES_LOCALSTORAGE_KEY, [...userOverrides])
    deactivate settingsStore

    UI->>settingsStore: updateMultipleConfig({key1: val1, key2: val2})
    activate settingsStore
    Note right of settingsStore: Batch update, single save
    settingsStore->>settingsStore: For each key: config[key] = value
    settingsStore->>settingsStore: For each key: userOverrides.add(key)
    settingsStore->>settingsStore: saveConfig()
    deactivate settingsStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 🔄 RESET
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>settingsStore: resetConfig()
    activate settingsStore
    settingsStore->>settingsStore: config = {...SETTING_CONFIG_DEFAULT}
    settingsStore->>settingsStore: userOverrides.clear()
    Note right of settingsStore: All params reset to defaults<br/>Next syncWithServerDefaults will adopt server values
    settingsStore->>settingsStore: saveConfig()
    deactivate settingsStore

    UI->>settingsStore: resetParameterToServerDefault(key)
    activate settingsStore
    settingsStore->>settingsStore: userOverrides.delete(key)
    settingsStore->>serverStore: defaultParams[key]
    settingsStore->>settingsStore: config[key] = serverDefault
    settingsStore->>settingsStore: saveConfig()
    deactivate settingsStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 🎨 THEME
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>settingsStore: updateTheme(newTheme)
    activate settingsStore
    settingsStore->>settingsStore: theme = newTheme
    settingsStore->>settingsStore: saveTheme()
    settingsStore->>LS: set("llama-theme", theme)
    deactivate settingsStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 📊 PARAMETER INFO
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>settingsStore: getParameterInfo(key)
    settingsStore->>ParamSvc: getParameterInfo(key, config, serverDefaults, userOverrides)
    ParamSvc-->>settingsStore: ParameterInfo
    Note right of ParamSvc: {<br/>  currentValue,<br/>  serverDefault,<br/>  isUserOverride: boolean,<br/>  canSync: boolean,<br/>  isDifferentFromServer: boolean<br/>}

    UI->>settingsStore: getParameterDiff()
    settingsStore->>ParamSvc: createParameterDiff(config, serverDefaults, userOverrides)
    ParamSvc-->>settingsStore: ParameterDiff[]
    Note right of ParamSvc: Array of parameters where user != server

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,LS: 📋 CONFIG CATEGORIES
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over settingsStore: Syncable with server (from /props):
    rect rgb(240, 255, 240)
        Note over settingsStore: temperature, top_p, top_k, min_p<br/>repeat_penalty, presence_penalty, frequency_penalty<br/>dynatemp_range, dynatemp_exponent<br/>typ_p, xtc_probability, xtc_threshold<br/>dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n
    end

    Note over settingsStore: UI-only (not synced):
    rect rgb(255, 240, 240)
        Note over settingsStore: systemMessage, custom (JSON)<br/>showStatistics, enableContinueGeneration<br/>autoMicOnEmpty, disableAutoScroll<br/>apiKey, pdfAsImage, disableReasoningParsing, showRawOutputSwitch
    end
```
