---
trigger: manual
---

# Coding rules

## Svelte & SvelteKit

### Services vs Stores Separation Pattern

#### `lib/services/` - Pure Business Logic

-   **Purpose**: Stateless business logic and external communication
-   **Contains**:
    -   API calls to external services (ApiService)
    -   Pure business logic functions (ChatService, etc.)
-   **Rules**:
    -   NO Svelte runes ($state, $derived, $effect)
    -   NO reactive state management
    -   Pure functions and classes only
    -   Can import types but not stores
    -   Focus on "how" - implementation details

#### `lib/stores/` - Reactive State Management

-   **Purpose**: Svelte-specific reactive state with runes
-   **Contains**:
    -   Reactive state classes with $state, $derived, $effect
    -   Database operations (DatabaseStore)
    -   UI-focused state management
    -   Store orchestration logic
-   **Rules**:
    -   USE Svelte runes for reactivity
    -   Import and use services for business logic
    -   NO direct database operations
    -   NO direct API calls (use services)
    -   Focus on "what" - reactive state for UI

#### Enforcement

-   Services should be testable without Svelte
-   Stores should leverage Svelte's reactivity system
-   Clear separation: services handle data, stores handle state
-   Services can be reused across multiple stores

#### Misc

-   Always use `let` for $derived state variables
