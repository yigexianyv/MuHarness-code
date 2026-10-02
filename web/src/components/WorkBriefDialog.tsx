import { useEffect, useId, useRef, useState, type FormEvent, type ReactElement } from 'react'
import { createPortal } from 'react-dom'

import { createConversation } from '../api/conversations'
import {
  buildWorkBrief,
  clearWorkBriefDraft,
  emptyWorkBrief,
  loadWorkBriefDraft,
  saveWorkBriefDraft,
  type WorkBriefDraft,
} from '../agent/workBrief'

export interface WorkBriefDialogProps {
  open: boolean
  onClose: () => void
  onPrepared: (conversationId: string, content: string, mode: 'normal' | 'plan') => void
}

interface PreparedWork {
  conversationId: string
  content: string
  mode: 'normal' | 'plan'
}

export function WorkBriefDialog({ open, onClose, onPrepared }: WorkBriefDialogProps): ReactElement | null {
  const id = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const goalRef = useRef<HTMLTextAreaElement>(null)
  const closeRef = useRef(onClose)
  const preparedCallbackRef = useRef(onPrepared)
  closeRef.current = onClose
  preparedCallbackRef.current = onPrepared
  const submittingRef = useRef(false)
  const preparedRef = useRef<PreparedWork | null>(null)
  const [draft, setDraft] = useState<WorkBriefDraft>(emptyWorkBrief)
  const [draftLoaded, setDraftLoaded] = useState(false)
  const [saved, setSaved] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [copyStatus, setCopyStatus] = useState('')
  const [previewExpanded, setPreviewExpanded] = useState(true)
  const [created, setCreated] = useState(false)
  const content = buildWorkBrief(draft)

  useEffect(() => {
    if (!open || draftLoaded) return
    setDraft(loadWorkBriefDraft())
    setDraftLoaded(true)
  }, [open, draftLoaded])

  useEffect(() => {
    if (open && draftLoaded && !created) setSaved(saveWorkBriefDraft(draft))
  }, [draft, draftLoaded, open, created])

  useEffect(() => {
    if (!open) return
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const focusTimer = window.setTimeout(() => goalRef.current?.focus(), 0)
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (!submittingRef.current) closeRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const focusable = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), textarea:not(:disabled), input:not(:disabled), select:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])',
      ) ?? []).filter((element) => element.getClientRects().length > 0)
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (!first || !last) {
        event.preventDefault()
        dialogRef.current?.focus()
      } else if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      window.clearTimeout(focusTimer)
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [open])

  function update<Field extends keyof WorkBriefDraft>(field: Field, value: WorkBriefDraft[Field]): void {
    setDraft((current) => ({ ...current, [field]: value }))
    setError('')
    setCopyStatus('')
  }

  function requestClose(): void {
    if (!submittingRef.current) closeRef.current()
  }

  async function copyPreview(): Promise<void> {
    try {
      if (typeof navigator === 'undefined' || !navigator.clipboard?.writeText) throw new Error('Clipboard unavailable')
      await navigator.clipboard.writeText(content)
      setCopyStatus('指令已复制。')
    } catch {
      setPreviewExpanded(true)
      setCopyStatus('复制不可用，请在预览中选中内容手动复制。')
    }
  }

  function resetDraft(): void {
    setDraft(emptyWorkBrief())
    clearWorkBriefDraft()
    setError('')
    setCopyStatus('')
    goalRef.current?.focus()
  }

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    if (submittingRef.current) return
    if (!draft.goal.trim() && !preparedRef.current) {
      setError('请先写清楚这次工作的目标。')
      goalRef.current?.focus()
      return
    }
    submittingRef.current = true
    setBusy(true)
    setError('')
    let prepared = preparedRef.current
    if (!prepared) {
      try {
        const conversation = await createConversation()
        prepared = { conversationId: conversation.id, content, mode: draft.mode }
        preparedRef.current = prepared
        setCreated(true)
      } catch (reason) {
        setError(reason instanceof Error ? `创建失败：${reason.message}` : '创建失败，请稍后重试。')
        submittingRef.current = false
        setBusy(false)
        return
      }
    }
    try {
      preparedCallbackRef.current(prepared.conversationId, prepared.content, prepared.mode)
      clearWorkBriefDraft()
      preparedRef.current = null
      setCreated(false)
      setDraft(emptyWorkBrief())
      setSaved(false)
      closeRef.current()
    } catch {
      // A created conversation must remain recoverable if its view cannot be opened.
      setError('工作已创建，但工作区未能打开。点击“进入已创建工作区”重试。')
    } finally {
      submittingRef.current = false
      setBusy(false)
    }
  }

  if (!open) return null
  const titleId = `${id}-title`
  const descriptionId = `${id}-description`
  const dialog = (
    <div className="work-brief-overlay">
      <div ref={dialogRef} className="work-brief" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={descriptionId} tabIndex={-1}>
        <header className="work-brief__header">
          <div>
            <span className="work-brief__eyebrow">开始一项新工作</span>
            <h2 id={titleId}>先把目标和边界说清楚</h2>
            <p id={descriptionId}>填写工作说明，进入工作区后确认发送，Agent 才会开始运行。</p>
          </div>
          <button className="work-brief__close" type="button" disabled={busy} aria-label="关闭新建工作向导" onClick={requestClose}>×</button>
        </header>
        <form onSubmit={submit} className="work-brief__form">
          <div className="work-brief__body">
            <div className="work-brief__fields">
              <label className="work-brief__field" htmlFor={`${id}-goal`}>
                <span className="work-brief__field-heading">工作目标 <em>必填</em></span>
                <textarea ref={goalRef} id={`${id}-goal`} value={draft.goal} onChange={(event) => update('goal', event.target.value)} required disabled={busy || created} rows={3} placeholder="你希望解决什么问题？最终要得到什么？" />
              </label>
              <label className="work-brief__field" htmlFor={`${id}-references`}>
                <span className="work-brief__field-heading">参考资料与工作范围 <small>选填</small></span>
                <textarea id={`${id}-references`} value={draft.references} onChange={(event) => update('references', event.target.value)} disabled={busy || created} rows={3} placeholder="相关链接、文件路径、需要分析的范围，或已有背景。" />
                <span className="work-brief__field-help">这里填写说明，不会上传文件，也不会立即读取资源。</span>
              </label>
              <label className="work-brief__field" htmlFor={`${id}-constraints`}>
                <span className="work-brief__field-heading">必须遵守的约束 <small>选填</small></span>
                <textarea id={`${id}-constraints`} value={draft.constraints} onChange={(event) => update('constraints', event.target.value)} disabled={busy || created} rows={2} placeholder="例如：保留现有接口；只修改指定目录；先分析再实施。" />
              </label>
              <label className="work-brief__field" htmlFor={`${id}-acceptance`}>
                <span className="work-brief__field-heading">完成验收标准 <small>选填</small></span>
                <textarea id={`${id}-acceptance`} value={draft.acceptance} onChange={(event) => update('acceptance', event.target.value)} disabled={busy || created} rows={2} placeholder="怎样才算做完？需要哪些结果、验证或交付物？" />
              </label>
              <fieldset className="work-brief__modes" disabled={busy || created}>
                <legend>进入工作区后的模式</legend>
                <div className="work-brief__mode-options">
                  <button type="button" className={`work-brief__mode${draft.mode === 'normal' ? ' is-selected' : ''}`} aria-pressed={draft.mode === 'normal'} onClick={() => update('mode', 'normal')}><strong>执行模式</strong><span>确认发送后，按目标开展工作。</span></button>
                  <button type="button" className={`work-brief__mode${draft.mode === 'plan' ? ' is-selected' : ''}`} aria-pressed={draft.mode === 'plan'} onClick={() => update('mode', 'plan')}><strong>规划模式</strong><span>确认发送后，先梳理方案与步骤。</span></button>
                </div>
              </fieldset>
            </div>
            <aside className="work-brief__preview" aria-label="完整工作指令预览">
              <div className="work-brief__preview-header">
                <div><span className="work-brief__preview-label">实时指令预览</span><h3>这段内容将填入工作区</h3></div>
                <button type="button" className="work-brief__preview-toggle" aria-expanded={previewExpanded} aria-controls={`${id}-preview`} onClick={() => setPreviewExpanded((value) => !value)}>{previewExpanded ? '收起' : '展开'}</button>
              </div>
              {previewExpanded ? <div id={`${id}-preview`} className="work-brief__preview-body">{content ? <pre className="work-brief__preview-content" tabIndex={0}>{content}</pre> : <p className="work-brief__preview-placeholder">左侧填写内容后，这里会显示完整指令。未填写的选填项不会加入指令。</p>}</div> : null}
              <div className="work-brief__preview-footer"><span>{content.length} 字符 · {draft.mode === 'plan' ? '规划模式' : '执行模式'}</span><button type="button" className="btn btn-sm work-brief__copy" disabled={!content} onClick={() => void copyPreview()}>复制指令</button></div>
              <p className="work-brief__copy-status" role="status">{copyStatus}</p>
              <div className="work-brief__preview-note"><strong>创建只是准备工作</strong><p>这一步只创建会话并填入草稿，不会发送指令、执行工具或创建后台任务。你可以在工作区继续编辑，再决定何时发送。</p></div>
            </aside>
          </div>
          <footer className="work-brief__footer">
            <div className="work-brief__draft-status"><span>{busy ? '正在创建，请等待工作区准备完成。' : saved ? '输入草稿已保存在当前浏览器。' : '浏览器暂未保存草稿，请保留当前窗口。'}</span><button type="button" disabled={busy || created} onClick={resetDraft}>清空草稿</button></div>
            {error ? <p className="work-brief__error" role="alert">{error}</p> : null}
            <div className="work-brief__actions"><button type="button" className="btn" disabled={busy} onClick={requestClose}>关闭</button><button type="submit" className="btn btn-primary work-brief__submit" disabled={busy || (!draft.goal.trim() && !created)}>{busy ? '正在创建…' : created ? '进入已创建工作区' : '创建并进入工作区'}</button></div>
          </footer>
        </form>
      </div>
    </div>
  )
  return typeof document === 'undefined' ? dialog : createPortal(dialog, document.body)
}

export default WorkBriefDialog
