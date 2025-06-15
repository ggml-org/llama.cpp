import { ToolCallRequest } from '../../utils/types';

export const ToolCallArgsDisplay = ({
  toolCall,
  baseClassName = 'collapse bg-base-200 collapse-arrow mb-4',
}: {
  toolCall: ToolCallRequest;
  baseClassName?: string;
}) => {
  let parsedArgs = toolCall.function.arguments;
  try {
    parsedArgs = JSON.stringify(JSON.parse(parsedArgs), null, 2);
  } catch (e) {
    // Might still be generating
  }
  return (
    <details className={baseClassName} open={false}>
      <summary className="collapse-title">
        <b>Tool call:</b> {toolCall.function.name}
      </summary>
      <div className="collapse-content">
        <div className="mb-1">Arguments:</div>
        <pre className="whitespace-pre-wrap bg-base-300 p-2 rounded">
          {parsedArgs}
        </pre>
      </div>
    </details>
  );
};
