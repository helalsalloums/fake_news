import type { ClassProbabilities } from '../types'

const labels: Record<keyof ClassProbabilities, string> = {
  SUPPORTED: 'مدعوم',
  REFUTED: 'مفنّد',
  NOT_ENOUGH_INFORMATION: 'الأدلة غير كافية',
}

export function Confidence({value, label = 'درجة الثقة'}: {value: number; label?: string}) {
  const percentage = Math.round(value * 100)
  return (
    <div aria-label={`${label}: ${percentage}%`}>
      <div className="mb-1 flex justify-between text-sm text-slate-600">
        <span>{label}</span><b>{percentage}%</b>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full rounded-full bg-teal-600" style={{width: `${percentage}%`}} />
      </div>
    </div>
  )
}

export function ProbabilityDetails({probabilities}: {probabilities: ClassProbabilities}) {
  return (
    <details className="mt-3 text-sm">
      <summary className="cursor-pointer text-teal-800">تفاصيل ثقة النموذج</summary>
      <dl className="mt-2 grid gap-2 sm:grid-cols-3">
        {(Object.keys(labels) as Array<keyof ClassProbabilities>).map((key) => (
          <div key={key} className="rounded-lg bg-slate-100 p-2">
            <dt className="text-slate-500">{labels[key]}</dt>
            <dd className="font-bold">{Math.round(probabilities[key] * 100)}%</dd>
          </div>
        ))}
      </dl>
    </details>
  )
}
