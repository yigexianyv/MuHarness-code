import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import DeskNavigation, { type DeskNavigationProps } from './DeskNavigation'

const render = (props: Partial<DeskNavigationProps> = {}): string => renderToStaticMarkup(
  <DeskNavigation
    current="overview"
    onNavigate={() => {}}
    onNewWork={() => {}}
    connected
    inboxCount={0}
    runningCount={0}
    {...props}
  />,
)

describe('DeskNavigation', () => {
  it('工作桌面提供四个主要入口和独立新建动作，次级入口按需展开', () => {
    const html = render()
    expect(html).toContain('aria-label="主导航"')
    for (const label of ['总览', '工作区', '决策', '成果', '新建工作']) expect(html).toContain(label)
    expect(html).toContain('MuHarness 工作桌面，返回总览')
    expect(html).toContain('aria-haspopup="menu"')
    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain('role="menuitem"')
    expect(html).not.toContain('sidebar')
  })

  it('主入口用 aria-current 表明当前页', () => {
    const html = render({ current: 'inbox' })
    expect(html).toMatch(/aria-current="page" aria-label="决策"/)
    expect(html.match(/aria-current="page"/g)).toHaveLength(1)
  })

  it('次级页面仍显示当前所属入口，关闭菜单时不丢失当前位置', () => {
    const html = render({ current: 'memory' })
    expect(html).toContain('desk-nav__more-trigger is-current')
    expect(html).toMatch(/aria-expanded="false"[^>]*aria-current="page"/)
    expect(html.match(/aria-current="page"/g)).toHaveLength(1)
  })

  it('展示来自实际状态的执行和决策数量，数量同时进入无障碍名称', () => {
    const html = render({ inboxCount: 105, runningCount: 3 })
    expect(html).toContain('aria-label="工作区，3 项执行中"')
    expect(html).toContain('aria-label="决策，105 项待决策"')
    expect(html).toContain('>105</span>')
    expect(html).toContain('>3</span>')
    expect(render()).not.toContain('desk-nav__count')
  })

  it('服务状态随真实连接变化并向辅助技术播报', () => {
    expect(render()).toContain('本地服务已连接')
    expect(render()).toContain('desk-nav__connection is-connected')
    const offline = render({ connected: false })
    expect(offline).toContain('本地服务未连接')
    expect(offline).toContain('role="status" aria-live="polite"')
    expect(offline).not.toContain('desk-nav__connection is-connected')
  })
})
