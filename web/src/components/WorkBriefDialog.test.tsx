import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import WorkBriefDialog from './WorkBriefDialog'

const props = { onClose: vi.fn(), onPrepared: vi.fn() }

describe('新建工作向导', () => {
  it('关闭时不显示表单', () => {
    expect(renderToStaticMarkup(<WorkBriefDialog {...props} open={false} />)).toBe('')
  })

  it('服务端安全渲染，提供目标、范围、约束、验收及模式，不自动运行', () => {
    const html = renderToStaticMarkup(<WorkBriefDialog {...props} open />)
    expect(html).toContain('role="dialog"')
    expect(html).toContain('aria-modal="true"')
    expect(html).toContain('aria-labelledby=')
    expect(html).toContain('工作目标')
    expect(html).toContain('参考资料与工作范围')
    expect(html).toContain('必须遵守的约束')
    expect(html).toContain('完成验收标准')
    expect(html).toContain('执行模式')
    expect(html).toContain('规划模式')
    expect(html).toContain('只创建会话并填入草稿')
    expect(html).toContain('创建并进入工作区')
    expect(html).toContain('实时指令预览')
    expect(html).toContain('复制指令')
    expect(props.onPrepared).not.toHaveBeenCalled()
  })
})
