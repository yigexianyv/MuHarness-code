import { useCallback, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactElement } from 'react'
import { Icon } from './Icon'

export type DeskPage = 'overview' | 'chat' | 'inbox' | 'ledger' | 'automations' | 'memory' | 'settings'

export interface DeskNavigationProps {
  current: DeskPage
  onNavigate: (next: DeskPage) => void
  onNewWork: () => void
  connected: boolean
  inboxCount: number
  runningCount: number
}

const SECONDARY_PAGES = [
  { key: 'automations', label: '自动化', detail: '安排重复工作', icon: 'automations' },
  { key: 'memory', label: '记忆', detail: '管理长期积累', icon: 'memory' },
  { key: 'settings', label: '设置', detail: '模型、扩展与连接', icon: 'settings' },
] as const

export default function DeskNavigation({
  current,
  onNavigate,
  onNewWork,
  connected,
  inboxCount,
  runningCount,
}: DeskNavigationProps): ReactElement {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuId = useId()
  const menuRoot = useRef<HTMLDivElement>(null)
  const menuTrigger = useRef<HTMLButtonElement>(null)
  const menuItems = useRef<Array<HTMLButtonElement | null>>([])
  const initialMenuFocus = useRef(0)
  const secondaryActive = SECONDARY_PAGES.some((page) => page.key === current)

  const closeMenu = useCallback((restoreFocus: boolean) => {
    setMenuOpen(false)
    if (restoreFocus) menuTrigger.current?.focus()
  }, [])

  useEffect(() => {
    if (!menuOpen) return
    menuItems.current[initialMenuFocus.current]?.focus()

    const handleOutsideClick = (event: PointerEvent): void => {
      if (event.target instanceof Node && !menuRoot.current?.contains(event.target)) {
        closeMenu(false)
      }
    }
    const handleEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeMenu(true)
      }
    }
    document.addEventListener('pointerdown', handleOutsideClick)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('pointerdown', handleOutsideClick)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [menuOpen, closeMenu])

  const openMenu = (focusIndex = 0): void => {
    initialMenuFocus.current = focusIndex
    if (menuOpen) menuItems.current[focusIndex]?.focus()
    else setMenuOpen(true)
  }

  const moveMenuFocus = (event: KeyboardEvent<HTMLDivElement>): void => {
    const index = menuItems.current.findIndex((item) => item === event.target)
    let nextIndex: number
    switch (event.key) {
      case 'ArrowDown': nextIndex = (index + 1) % SECONDARY_PAGES.length; break
      case 'ArrowUp': nextIndex = (index - 1 + SECONDARY_PAGES.length) % SECONDARY_PAGES.length; break
      case 'Home': nextIndex = 0; break
      case 'End': nextIndex = SECONDARY_PAGES.length - 1; break
      case 'Tab': closeMenu(false); return
      default: return
    }
    event.preventDefault()
    menuItems.current[nextIndex]?.focus()
  }

  return (
    <header className="desk-nav">
      <button
        type="button"
        className="desk-nav__brand"
        onClick={() => onNavigate('overview')}
        aria-label="MuHarness 工作桌面，返回总览"
      >
        <span className="desk-nav__symbol" aria-hidden="true">μ<span /></span>
        <span className="desk-nav__brand-copy"><strong>MuHarness</strong><span>工作桌面</span></span>
      </button>

      <nav className="desk-nav__main" aria-label="主导航">
        {([
          { key: 'overview', label: '总览', count: 0 },
          { key: 'chat', label: '工作区', count: runningCount },
          { key: 'inbox', label: '决策', count: inboxCount },
          { key: 'ledger', label: '成果', count: 0 },
        ] as const).map((page) => (
          <button
            key={page.key}
            type="button"
            className={`desk-nav__tab${current === page.key ? ' is-current' : ''}`}
            onClick={() => onNavigate(page.key)}
            aria-current={current === page.key ? 'page' : undefined}
            aria-label={page.count > 0 ? `${page.label}，${page.count} ${page.key === 'inbox' ? '项待决策' : '项执行中'}` : page.label}
          >
            {page.label}
            {page.count > 0 && <span className={`desk-nav__count${page.key === 'inbox' ? ' desk-nav__count--attention' : ''}`} aria-hidden="true">{page.count}</span>}
          </button>
        ))}
        <div className="desk-nav__more" ref={menuRoot}>
          <button
            type="button"
            ref={menuTrigger}
            className={`desk-nav__tab desk-nav__more-trigger${secondaryActive ? ' is-current' : ''}`}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-controls={menuId}
            aria-current={secondaryActive ? 'page' : undefined}
            onClick={() => menuOpen ? closeMenu(false) : openMenu()}
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault()
                openMenu(event.key === 'ArrowUp' ? SECONDARY_PAGES.length - 1 : 0)
              }
            }}
          >
            更多<Icon name="chevronDown" size={14} />
          </button>
          {menuOpen && (
            <div id={menuId} role="menu" aria-label="更多工作入口" className="desk-nav__menu" onKeyDown={moveMenuFocus}>
              {SECONDARY_PAGES.map((page, index) => (
                <button
                  key={page.key}
                  ref={(item) => { menuItems.current[index] = item }}
                  type="button"
                  role="menuitem"
                  tabIndex={-1}
                  className={`desk-nav__menu-item${current === page.key ? ' is-current' : ''}`}
                  aria-current={current === page.key ? 'page' : undefined}
                  onClick={() => {
                    closeMenu(true)
                    onNavigate(page.key)
                  }}
                >
                  <Icon name={page.icon} size={17} />
                  <span><strong>{page.label}</strong><small>{page.detail}</small></span>
                </button>
              ))}
            </div>
          )}
        </div>
      </nav>

      <div className="desk-nav__actions">
        <span className={`desk-nav__connection${connected ? ' is-connected' : ''}`} role="status" aria-live="polite">
          <span className="desk-nav__connection-dot" aria-hidden="true" />
          <span>{connected ? '本地服务已连接' : '本地服务未连接'}</span>
        </span>
        <button type="button" className="desk-nav__new-work" onClick={onNewWork}>
          <Icon name="plus" size={17} /><span>新建工作</span>
        </button>
      </div>
    </header>
  )
}
