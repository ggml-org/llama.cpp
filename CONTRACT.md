# Agentic built-in tools contract

This document defines the shared server and frontend interfaces consumed by the Wave 2 question implementation.

## Interactive request rail

`AgenticInteractiveRequest` is a discriminated union:

```ts
type AgenticInteractiveRequest =
	| {
			kind: 'permission';
			toolName: string;
			serverLabel: string;
	  }
	| {
			kind: 'question';
			toolName: 'question';
			requestID: string;
			questions: AgenticQuestionPrompt[];
	  };
```

The shared resolver accepts either an ordinary permission decision or question answers:

```ts
type AgenticQuestionAnswers = string[][];
type AgenticInteractiveResolution = ToolPermissionDecision | AgenticQuestionAnswers;

agenticResolvePermission(
	conversationId: string,
	resolution: AgenticInteractiveResolution
): void;
```

Each outer array entry corresponds to the question at the same index. Each inner array contains the selected labels and/or custom answer for that question. Slice Q submits answers through `agenticResolvePermission(conversationId, answers)`. Dismissal uses `ToolPermissionDecision.DENY`.

Question calls do not show the ordinary allow/deny card. The flow is:

1. The model calls `question` with `questions`.
2. `POST /tools` returns `status: "awaiting_user"` with a request ID and the questions.
3. The agentic store publishes a `kind: "question"` interactive request and pauses on the shared resolver.
4. The question action card resolves with `string[][]` answers or `ToolPermissionDecision.DENY`.
5. The agentic store calls `POST /tools` again with the original arguments, `request_id`, and either `answers` or `rejected: true`.
6. The completed response becomes the normal tool-result message and the agentic loop continues.

No question-specific server endpoint is used.

## question

Enum value:

```ts
BuiltInTool.QUESTION = 'question'
```

Initial result:

```json
{
  "status": "awaiting_user",
  "kind": "question",
  "request_id": "question-<unique-id>",
  "payload": {
    "request_id": "question-<unique-id>",
    "questions": [
      {
        "question": "Complete question",
        "header": "Short label",
        "options": [
          {"label": "Choice", "description": "Choice explanation"}
        ],
        "multiple": false,
        "custom": true
      }
    ]
  }
}
```

Answered result:

```json
{
  "status": "completed",
  "plain_text_response": "User has answered your questions: \"<question>\"=\"<answers>\". You can now continue with the user's answers in mind."
}
```

Dismissed result:

```json
{
  "status": "completed",
  "plain_text_response": "The user dismissed this question.",
  "is_error": true
}
```

Validation failures return `{"error":"<message>"}`.

Slice Q replaces:

- `tools/ui/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockQuestion.svelte`
- `tools/ui/src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/question.ts`
- `tools/ui/src/lib/components/app/chat/ChatMessages/ChatMessageActions/ChatMessageActionCard/ChatMessageActionCardQuestionRequest.svelte`
