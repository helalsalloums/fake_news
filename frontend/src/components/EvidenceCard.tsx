import type { Evidence } from '../types'

export function EvidenceCard({evidence}: {evidence: Evidence}) {
  const border = evidence.stance === 'SUPPORTING' ? 'border-emerald-300' : evidence.stance === 'CONTRADICTING' ? 'border-rose-300' : 'border-slate-200'
  return (
    <article className={`rounded-xl border-r-4 ${border} bg-white p-4 shadow-sm`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h4 className="font-bold">{evidence.title || evidence.source}</h4>
        <span className="text-xs text-slate-500">جودة الدليل {Math.round(evidence.scores.overall * 100)}%</span>
      </div>
      <p className="my-3 leading-7 text-slate-700">{evidence.snippet}</p>
      <div className="flex flex-wrap gap-3 text-xs text-slate-500">
        <span>{evidence.source}</span>
        {evidence.published_at && <time>{new Date(evidence.published_at).toLocaleDateString('ar')}</time>}
        <a className="font-semibold text-teal-700 underline" href={evidence.url} target="_blank" rel="noreferrer">عرض المصدر</a>
      </div>
    </article>
  )
}
