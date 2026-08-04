import type { Verdict } from '../types'

const presentation: Record<Verdict, {label: string; icon: string; classes: string}> = {
  SUPPORTED: {label: 'مدعوم', icon: '✓', classes: 'bg-emerald-100 text-emerald-800'},
  REFUTED: {label: 'مفنّد', icon: '✕', classes: 'bg-rose-100 text-rose-800'},
  NOT_ENOUGH_INFORMATION: {label: 'غير كافٍ من الأدلة', icon: '?', classes: 'bg-amber-100 text-amber-900'},
}

export function VerdictBadge({verdict, large = false}: {verdict: Verdict; large?: boolean}) {
  const item = presentation[verdict]
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-4 py-2 font-bold ${item.classes} ${large ? 'text-xl' : 'text-sm'}`}>
      <span aria-hidden="true">{item.icon}</span>{item.label}
    </span>
  )
}
