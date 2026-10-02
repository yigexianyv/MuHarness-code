

import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'

import Sidebar from './Sidebar'

describe('Sidebar (App Shell)', () => {
  it('渲染全部导航项', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected />,
    )
    for (const label of ['工作台', '自动化', '待处理', '账本', '记忆', '设置']) {
      expect(html).toContain(label)
    }
    expect(html).toContain('MuHarness')
  })

  it('当前页有 active 状态', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="ledger" onNavigate={() => {}} connected />,
    )
    expect(html).toContain('nav-item active')
  })

  it('connected 显示 Host 就绪', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected />,
    )
    expect(html).toContain('Host 就绪')
    expect(html).toContain('status-dot--ready')
    expect(html).not.toContain('Host 离线')
  })

  it('断开连接显示 Host 离线', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected={false} />,
    )
    expect(html).toContain('Host 离线')
    expect(html).toContain('status-dot--offline')
  })

  it('图标导航提供 title 与无障碍名称，不在窄栏展示版本噪声', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected />,
    )
    expect(html).toContain('title="工作台：对话与正在推进的长任务"')
    expect(html).toContain('aria-label="工作台"')
    expect(html).not.toContain('v0.1.0')
  })

  it('提供 badges 时在对应导航项渲染徽标', () => {
    const html = renderToStaticMarkup(
      <Sidebar
        current="chat"
        onNavigate={() => {}}
        connected
        badges={{ chat: 2, inbox: 3 }}
      />,
    )
    expect(html).toContain('nav-badge')
    expect(html).toContain('>2</span>')
    expect(html).toContain('>3</span>')
  })

  it('badges 为 0 或缺失时不渲染徽标', () => {
    const html = renderToStaticMarkup(
      <Sidebar
        current="chat"
        onNavigate={() => {}}
        connected
        badges={{ chat: 0, inbox: undefined }}
      />,
    )
    expect(html).not.toContain('nav-badge')
  })

  it('dots 为 true 时在对应导航项渲染状态点', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected dots={{ chat: true }} />,
    )
    expect(html).toContain('nav-badge--dot')
  })

  it('无 dots 时不渲染状态点', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected />,
    )
    expect(html).not.toContain('nav-badge--dot')
  })

  it('导航按 进行中 / 等你处理 / 已验证 分组，有待处理时高亮该组', () => {
    const html = renderToStaticMarkup(
      <Sidebar current="chat" onNavigate={() => {}} connected badges={{ inbox: 2 }} />,
    )
    const order = ['进行中', '等你处理', '已验证'].map((label) => html.indexOf(`aria-label="${label}"`))
    expect(order.every((index) => index >= 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    expect(html).toContain('sidebar-group--attention sidebar-group--waiting')
    expect(html).not.toContain('sidebar-group--active sidebar-group--waiting')
  })
})
