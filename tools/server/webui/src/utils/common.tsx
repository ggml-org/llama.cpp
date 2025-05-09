export const XCloseButton: React.ElementType<
  React.ClassAttributes<HTMLButtonElement> &
    React.HTMLAttributes<HTMLButtonElement>
> = ({ className, ...props }) => (
  <button className={`btn btn-square btn-sm ${className ?? ''}`} {...props}>
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M6 18L18 6M6 6l12 12"
      />
    </svg>
  </button>
);

export const OpenInNewTab = ({
  href,
  children,
}: {
  href: string;
  children: string;
}) => (
  <a
    className="underline"
    href={href}
    target="_blank"
    rel="noopener noreferrer"
  >
    {children}
  </a>
);

export function BtnWithTooltips({
  className,
  onClick,
  onMouseLeave,
  children,
  tooltipsContent,
  disabled,
}: {
  className?: string;
  onClick: () => void;
  onMouseLeave?: () => void;
  children: React.ReactNode;
  tooltipsContent: string;
  disabled?: boolean;
}) {
  return (
    <div className="tooltip tooltip-bottom" data-tip={tooltipsContent}>
      <button
        className={`${className ?? ''} flex items-center justify-center`}
        onClick={onClick}
        disabled={disabled}
        onMouseLeave={onMouseLeave}
      >
        {children}
      </button>
    </div>
  );
}
