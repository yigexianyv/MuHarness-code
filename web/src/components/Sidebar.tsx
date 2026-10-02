import type { ReactElement } from 'react'
import type { PageKey } from '../App'
import { Icon } from './Icon'
import type { IconName } from './Icon'
import { StatusDot } from './ui'

export interface SidebarItem {
  key: PageKey
  label: string
  icon: IconName
  /** 悬停提示：说明这个入口汇总了什么 */
  hint: string
}

export interface SidebarGroup {
  id: 'active' | 'attention' | 'verified'
  label: string
  items: SidebarItem[]
}

/**
 * 导航按“人要做什么”分三组，而不是按数据类型平铺：
 * 进行中（正在推进的工作）→ 等你处理（所有需要人介入的事）→ 已验证（沉淀下来的结果）。
 */
export const SIDEBAR_GROUPS: SidebarGroup[] = [
  {
    id: 'active',
    label: '进行中',
    items: [
      { key: 'chat', label: '工作台', icon: 'chat', hint: '对话与正在推进的长任务' },
      { key: 'automations', label: '自动化', icon: 'automations', hint: '定时与触发式任务' },
    ],
  },
  {
    id: 'attention',
    label: '等你处理',
    items: [
      { key: 'inbox', label: '待处理', icon: 'inbox', hint: '审批、长任务提问、可恢复的中断' },
    ],
  },
  {
    id: 'verified',
    label: '已验证',
    items: [
      { key: 'ledger', label: '账本', icon: 'ledger', hint: '执行记录、审计结论与交付物' },
      { key: 'memory', label: '记忆', icon: 'memory', hint: '跨会话保留的长期记忆' },
    ],
  },
]

export const SIDEBAR_ITEMS: SidebarItem[] = SIDEBAR_GROUPS.flatMap((group) => group.items)

export interface SidebarProps {
  current: PageKey
  onNavigate: (page: PageKey) => void
  connected: boolean

  badges?: Partial<Record<PageKey, number>>

  dots?: Partial<Record<PageKey, boolean>>
}

export default function Sidebar({
  current,
  onNavigate,
  connected,
  badges,
  dots,
}: SidebarProps): ReactElement {
  const badgeFor = (key: PageKey): number | null => {
    const value = badges?.[key]
    return typeof value === 'number' && value > 0 ? value : null
  }

  const renderItem = (item: SidebarItem): ReactElement => {
    const badge = badgeFor(item.key)
    return (
      <button
        key={item.key}
        type="button"
        className={`nav-item ${current === item.key ? 'active' : ''}`}
        onClick={() => onNavigate(item.key)}
        aria-current={current === item.key ? 'page' : undefined}
        aria-label={item.label}
        title={`${item.label}：${item.hint}`}
      >
        <span className="nav-item__icon"><Icon name={item.icon} size={19} /></span>
        <span className="nav-item__copy">
          <span className="nav-item__label">{item.label}</span>
          <span className="nav-item__hint" aria-hidden="true">{item.hint}</span>
        </span>
        {badge !== null ? (
          <span className="nav-badge" aria-label={`${badge}`}>
            {badge > 99 ? '99+' : badge}
          </span>
        ) : dots?.[item.key] ? (
          <span className="nav-badge nav-badge--dot" aria-hidden="true" />
        ) : null}
      </button>
    )
  }

  return (
    <nav className="sidebar" aria-label="主导航">
      <div className="sidebar-brand" aria-label="MuHarness">
        <span className="brand-symbol" aria-hidden="true">μ</span>
        <span className="sidebar-brand__copy">
          <span className="brand-wordmark">MuHarness</span>
          <span className="brand-tagline">AGENT WORKSPACE</span>
        </span>
      </div>
      <div className="sidebar-nav">
        {SIDEBAR_GROUPS.map((group, index) => {
          const waiting = group.items.some((item) => badgeFor(item.key) !== null)
          return (
            <div
              key={group.id}
              className={`sidebar-group sidebar-group--${group.id}${waiting ? ' sidebar-group--waiting' : ''}`}
              role="group"
              aria-label={group.label}
            >
              <div className="sidebar-group__heading" aria-hidden="true">
                <span className="sidebar-group__label">{group.label}</span>
                <span className="sidebar-group__index">{String(index + 1).padStart(2, '0')}</span>
              </div>
              {group.items.map(renderItem)}
            </div>
          )
        })}
      </div>
      <div className="sidebar-footer">
        <button
          type="button"
          className={`nav-item ${current === 'settings' ? 'active' : ''}`}
          onClick={() => onNavigate('settings')}
          aria-current={current === 'settings' ? 'page' : undefined}
          title="设置"
          aria-label="设置"
        >
          <span className="nav-item__icon"><Icon name="settings" size={19} /></span>
          <span className="nav-item__label">设置</span>
        </button>
        <div
          className="host-status"
          title={connected ? 'Host 已连接' : 'Host 已离线'}
          aria-label={connected ? 'Host 就绪' : 'Host 离线'}
          role="status"
          aria-live="polite"
        >
          <StatusDot tone={connected ? 'ready' : 'offline'} />
          <span className="host-status__copy">
            <span className="host-status__title">{connected ? '本地服务已连接' : '等待本地服务'}</span>
            <span className="host-status__detail">{connected ? '任务状态实时同步' : '连接后同步任务状态'}</span>
          </span>
        </div>
      </div>
    </nav>
  )
}
