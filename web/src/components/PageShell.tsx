





import type { ReactElement, ReactNode } from 'react'

export interface PageShellProps {
  title: string
  subtitle?: string

  actions?: ReactNode
  children?: ReactNode

  maxWidth?: number
}

export function PageShell({
  title,
  subtitle,
  actions,
  children,
  maxWidth = 960,
}: PageShellProps): ReactElement {
  return (
    <div className="page-shell">
      <header className="page-shell__header">
        <div className="page-shell__header-inner" style={maxWidth ? { maxWidth } : undefined}>
          <div className="page-shell__heading">
            <div className="page-shell__eyebrow" aria-hidden="true">
              <span className="page-shell__marker" />
              工作空间
            </div>
            <h1 className="page-shell__title">{title}</h1>
            {subtitle ? <p className="page-shell__subtitle">{subtitle}</p> : null}
          </div>
          {actions ? <div className="page-shell__actions">{actions}</div> : null}
        </div>
      </header>
      <div className="page-shell__body" style={maxWidth ? { maxWidth } : undefined}>
        {children}
      </div>
    </div>
  )
}
