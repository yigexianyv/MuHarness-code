export interface WorkBriefDraft {
  goal: string
  references: string
  constraints: string
  acceptance: string
  mode: 'normal' | 'plan'
}

export const WORK_BRIEF_DRAFT_KEY = 'muharness.work-brief.v1'

export function emptyWorkBrief(): WorkBriefDraft {
  return { goal: '', references: '', constraints: '', acceptance: '', mode: 'normal' }
}

export function buildWorkBrief(draft: WorkBriefDraft): string {
  const sections: Array<[string, string]> = [
    ['工作目标', draft.goal],
    ['参考资料与工作范围', draft.references],
    ['必须遵守的约束', draft.constraints],
    ['完成验收标准', draft.acceptance],
  ]
  return sections
    .filter(([, value]) => value.trim().length > 0)
    .map(([title, value]) => `## ${title}\n${value.trim()}`)
    .join('\n\n')
}

type DraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

function browserStorage(): DraftStorage | undefined {
  try {
    return typeof window === 'undefined' ? undefined : window.localStorage
  } catch {
    return undefined
  }
}

export function loadWorkBriefDraft(storage = browserStorage()): WorkBriefDraft {
  const fallback = emptyWorkBrief()
  try {
    const raw = storage?.getItem(WORK_BRIEF_DRAFT_KEY)
    if (!raw) return fallback
    const saved: unknown = JSON.parse(raw)
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return fallback
    const values = saved as Record<string, unknown>
    return {
      goal: typeof values.goal === 'string' ? values.goal : '',
      references: typeof values.references === 'string' ? values.references : '',
      constraints: typeof values.constraints === 'string' ? values.constraints : '',
      acceptance: typeof values.acceptance === 'string' ? values.acceptance : '',
      mode: values.mode === 'plan' ? 'plan' : 'normal',
    }
  } catch {
    return fallback
  }
}

export function saveWorkBriefDraft(draft: WorkBriefDraft, storage = browserStorage()): boolean {
  if (!storage) return false
  try {
    // Store only these user-owned fields; never serialize surrounding application state.
    const { goal, references, constraints, acceptance, mode } = draft
    storage.setItem(WORK_BRIEF_DRAFT_KEY, JSON.stringify({ goal, references, constraints, acceptance, mode }))
    return true
  } catch {
    return false
  }
}

export function clearWorkBriefDraft(storage = browserStorage()): boolean {
  if (!storage) return false
  try {
    storage.removeItem(WORK_BRIEF_DRAFT_KEY)
    return true
  } catch {
    return false
  }
}
