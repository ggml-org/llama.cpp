import { ToolCallRequest } from '../../utils/types';

export const ToolCallArgsDisplay = ({
  toolCall,
  baseClassName = 'collapse bg-base-200 collapse-arrow mb-4',
}: {
  toolCall: ToolCallRequest;
  baseClassName?: string;
}) => {
  return (
    <details className={baseClassName} open={false}>
      <summary className="collapse-title">
        <b>Tool call:</b> {toolCall.function.name}
      </summary>
      <div className="collapse-content">
        <div className="font-bold mb-1">Arguments:</div>
        <pre className="whitespace-pre-wrap bg-base-300 p-2 rounded">
          {toolCall.function.arguments}
        </pre>
      </div>
    </details>
  );
};
