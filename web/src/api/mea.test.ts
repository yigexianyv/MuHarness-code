import { beforeEach, describe, expect, it, vi } from 'vitest'

const { callMock } = vi.hoisted(() => ({ callMock: vi.fn() }))

vi.mock('../rpc', () => ({
  rpcClient: { call: callMock },
}))

import {
  answerMea,
  cancelMea,
  getMea,
  listMeas,
  listMeaTools,
  noteMea,
  pauseMea,
  resumeMea,
  startMea,
} from './mea'

describe('long task (mea) web api', () => {
  beforeEach(() => {
    callMock.mockReset()
  })

  it('mea.start 只在需要时传可选参数', async () => {
    callMock.mockResolvedValue({ mea: { id: 'mea-1' }, task: { id: 'task-1' } })
    await startMea('conv-1', 'task-1')
    expect(callMock).toHaveBeenLastCalledWith('mea.start', {
      conversation_id: 'conv-1',
      task_id: 'task-1',
    })

    await startMea('conv-1', 'task-1', { roundBudget: 40, extraTools: ['web_search'], autoApproveSandbox: false })
    expect(callMock).toHaveBeenLastCalledWith('mea.start', {
      conversation_id: 'conv-1',
      task_id: 'task-1',
      round_budget: 40,
      extra_tools: ['web_search'],
      auto_approve_sandbox: false,
    })
  })

  it('查询类方法', async () => {
    callMock.mockResolvedValueOnce({ mea: { id: 'mea-1' }, rounds: [] })
    await getMea('mea-1')
    expect(callMock).toHaveBeenLastCalledWith('mea.get', { mea_id: 'mea-1' })

    callMock.mockResolvedValueOnce({ meas: [{ id: 'mea-1' }], count: 1 })
    expect(await listMeas('conv-1', { limit: 5 })).toEqual([{ id: 'mea-1' }])
    expect(callMock).toHaveBeenLastCalledWith('mea.list', { conversation_id: 'conv-1', limit: 5 })

    callMock.mockResolvedValueOnce({ tools: ['http_request'] })
    expect(await listMeaTools()).toEqual(['http_request'])
    expect(callMock).toHaveBeenLastCalledWith('mea.tools', {})
  })

  it('回答和补充指令', async () => {
    callMock.mockResolvedValue({ accepted: true, reason: null, revision: 2, amendment_id: 'A1' })
    await answerMea('mea-1', '测试库')
    expect(callMock).toHaveBeenLastCalledWith('mea.answer', { mea_id: 'mea-1', text: '测试库' })

    await noteMea('mea-1', '改用测试库')
    expect(callMock).toHaveBeenLastCalledWith('mea.note', {
      mea_id: 'mea-1',
      text: '改用测试库',
      kind: 'persistent',
      immediate: false,
    })

    await noteMea('mea-1', '先看日志', { kind: 'once', immediate: true })
    expect(callMock).toHaveBeenLastCalledWith('mea.note', {
      mea_id: 'mea-1',
      text: '先看日志',
      kind: 'once',
      immediate: true,
    })
  })

  it('暂停、继续和取消', async () => {
    callMock.mockResolvedValue({ mea: { id: 'mea-1', status: 'paused' } })
    expect((await pauseMea('mea-1')).status).toBe('paused')
    expect(callMock).toHaveBeenLastCalledWith('mea.pause', { mea_id: 'mea-1' })

    await resumeMea('mea-1', 10)
    expect(callMock).toHaveBeenLastCalledWith('mea.resume', { mea_id: 'mea-1', extra_rounds: 10 })

    await cancelMea('mea-1')
    expect(callMock).toHaveBeenLastCalledWith('mea.cancel', { mea_id: 'mea-1' })
  })
})
