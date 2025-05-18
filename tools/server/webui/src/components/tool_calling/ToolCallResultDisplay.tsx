export const ToolCallResultDisplay = ({
  content,
  baseClassName = 'collapse bg-base-200 collapse-arrow mb-4',
}: {
  content: string;
  baseClassName?: string;
}) => {
  return (
    <details className={baseClassName} open={true}>
      <summary className="collapse-title">
        <b>Tool call result</b>
      </summary>
      <div className="collapse-content">
        <pre className="whitespace-pre-wrap bg-base-300 p-2 rounded">
          {content}
        </pre>
      </div>
    </details>
  );
};
