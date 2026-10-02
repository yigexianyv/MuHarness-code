import { describe, expect, it, vi } from 'vitest'

import {
  buildWorkBrief,
  clearWorkBriefDraft,
  emptyWorkBrief,
  loadWorkBriefDraft,
  saveWorkBriefDraft,
  WORK_BRIEF_DRAFT_KEY,
} from './workBrief'

const storage = (raw: string | null = null) => ({
  getItem: vi.fn(() => raw),
  setItem: vi.fn(),
  removeItem: vi.fn(),
})

describe('工作说明指令', () => {
  it('只使用用户填写的内容，保留内部换行，清理各段边缘空白', () => {
    const content = buildWorkBrief({
      goal: '  分析接口延迟\n定位最慢的步骤  ',
      references: '  backend/server.py  ',
      constraints: '\n不要修改公开 API\n',
      acceptance: ' 输出证据与修改方案 ',
      mode: 'plan',
    })
    expect(content).toBe('## 工作目标\n分析接口延迟\n定位最慢的步骤\n\n## 参考资料与工作范围\nbackend/server.py\n\n## 必须遵守的约束\n不要修改公开 API\n\n## 完成验收标准\n输出证据与修改方案')
  })

  it('不为缺省项生成约束、验收标准或占位内容', () => {
    expect(buildWorkBrief({ ...emptyWorkBrief(), goal: '  整理报告  ', constraints: ' \n ' }))
      .toBe('## 工作目标\n整理报告')
    expect(buildWorkBrief(emptyWorkBrief())).toBe('')
  })

  it('规划/执行模式作为独立运行设置，不向用户的指令插入额外命令', () => {
    const draft = { ...emptyWorkBrief(), goal: '研究问题' }
    expect(buildWorkBrief({ ...draft, mode: 'plan' })).toBe(buildWorkBrief(draft))
  })
})

describe('本地工作说明草稿', () => {
  it('安全恢复合法字段与规划模式，忽略其他应用状态', () => {
    const saved = storage(JSON.stringify({ goal: '我的目标', references: '链接', constraints: '边界', acceptance: '标准', mode: 'plan', token: 'not-a-real-token' }))
    expect(loadWorkBriefDraft(saved)).toEqual({ goal: '我的目标', references: '链接', constraints: '边界', acceptance: '标准', mode: 'plan' })
    expect(saved.getItem).toHaveBeenCalledWith(WORK_BRIEF_DRAFT_KEY)
  })

  it('只保存五项用户输入，不持久化传入对象中的其他字段', () => {
    const saved = storage()
    const draft = { ...emptyWorkBrief(), goal: '报告', applicationSecret: 'not-a-real-secret' }
    expect(saveWorkBriefDraft(draft, saved)).toBe(true)
    expect(saved.setItem).toHaveBeenCalledWith(WORK_BRIEF_DRAFT_KEY, JSON.stringify({ goal: '报告', references: '', constraints: '', acceptance: '', mode: 'normal' }))
  })

  it('异常草稿与不受支持的模式恢复成安全默认值', () => {
    for (const raw of ['{', 'null', '[]', '"text"']) expect(loadWorkBriefDraft(storage(raw))).toEqual(emptyWorkBrief())
    expect(loadWorkBriefDraft(storage(JSON.stringify({ goal: 12, references: null, mode: 'execute' })))).toEqual(emptyWorkBrief())
  })

  it('存储不可用、读取失败或写入失败不阻止界面使用', () => {
    const failed = {
      getItem: vi.fn(() => { throw new Error('Blocked') }),
      setItem: vi.fn(() => { throw new Error('Full') }),
      removeItem: vi.fn(() => { throw new Error('Blocked') }),
    }
    expect(loadWorkBriefDraft(failed)).toEqual(emptyWorkBrief())
    expect(saveWorkBriefDraft(emptyWorkBrief(), failed)).toBe(false)
    expect(clearWorkBriefDraft(failed)).toBe(false)
    expect(loadWorkBriefDraft()).toEqual(emptyWorkBrief())
    expect(saveWorkBriefDraft(emptyWorkBrief())).toBe(false)
  })

  it('成功创建或显式清空后移除对应草稿', () => {
    const saved = storage()
    expect(clearWorkBriefDraft(saved)).toBe(true)
    expect(saved.removeItem).toHaveBeenCalledWith(WORK_BRIEF_DRAFT_KEY)
  })
})
