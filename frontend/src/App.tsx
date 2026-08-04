import {FormEvent, useState} from 'react'
import {checkFact} from './api'
import {Confidence, ProbabilityDetails} from './components/Confidence'
import {EvidenceCard} from './components/EvidenceCard'
import {VerdictBadge} from './components/VerdictBadge'
import type {FactCheckResult} from './types'

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<FactCheckResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (text.trim().length < 3) return
    setLoading(true); setError(''); setResult(null)
    try { setResult(await checkFact(text.trim())) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'حدث خطأ غير متوقع') }
    finally { setLoading(false) }
  }

  return (
    <main dir="rtl" className="mx-auto min-h-screen max-w-5xl px-4 py-10 sm:px-8">
      <header className="mb-8">
        <p className="mb-2 text-sm font-bold text-teal-700">تحقق قائم على الأدلة</p>
        <h1 className="text-3xl font-black text-slate-900 sm:text-5xl">مدقّق الأخبار العربية</h1>
        <p className="mt-4 max-w-3xl leading-8 text-slate-600">يفحص النظام الادعاءات استناداً إلى المصادر المتاحة، ولا يدّعي معرفة الحقيقة المطلقة.</p>
      </header>

      <form onSubmit={submit} className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200 sm:p-7">
        <label htmlFor="claim" className="mb-3 block font-bold">ألصق الخبر أو الادعاء هنا</label>
        <textarea id="claim" value={text} onChange={(event) => setText(event.target.value)} maxLength={50000} rows={7}
          placeholder="أعلنت وزارة الصحة أن عدد الإصابات بلغ 500 حالة..."
          className="w-full resize-y rounded-xl border border-slate-300 p-4 leading-8 outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-100" />
        <div className="mt-4 flex items-center justify-between gap-4">
          <span className="text-xs text-slate-400">{text.length.toLocaleString('ar')} / ٥٠٬٠٠٠</span>
          <button disabled={loading || text.trim().length < 3} className="rounded-xl bg-teal-700 px-6 py-3 font-bold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50">
            {loading ? 'جارٍ التحقق…' : 'تحقق من الخبر'}
          </button>
        </div>
      </form>

      {error && <div role="alert" className="mt-6 rounded-xl border border-rose-200 bg-rose-50 p-4 text-rose-800">{error}</div>}

      {result && <section className="mt-8 space-y-6" aria-live="polite">
        <div className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
          <p className="mb-3 text-sm font-bold text-slate-500">النتيجة</p>
          <VerdictBadge verdict={result.verdict} large />
          <p className="my-5 leading-8">{result.explanation}</p>
          <Confidence value={result.confidence} />
          {result.conflict_detected && <p className="mt-4 rounded-lg bg-amber-50 p-3 font-bold text-amber-900">تنبيه: عثر النظام على أدلة موثوقة متعارضة.</p>}
        </div>

        <div>
          <h2 className="mb-4 text-2xl font-black">الادعاءات المستخرجة</h2>
          <div className="space-y-5">
            {result.claims.map((claim) => <article key={claim.id} className="rounded-2xl bg-white p-6 shadow-sm ring-1 ring-slate-200">
              <div className="flex flex-wrap items-start justify-between gap-3"><h3 className="max-w-2xl text-lg font-bold leading-8">{claim.claim}</h3><VerdictBadge verdict={claim.verdict} /></div>
              <p className="my-4 leading-7 text-slate-700">{claim.explanation}</p>
              <div className="grid gap-4 sm:grid-cols-2"><Confidence value={claim.confidence} label="ثقة القرار النهائي"/><Confidence value={claim.model_confidence} label="ثقة النموذج"/></div>
              <ProbabilityDetails probabilities={claim.class_probabilities} />
              {!!claim.evidence.length && <div className="mt-6 space-y-4">
                <h4 className="font-black text-emerald-800">أدلة تؤيد الادعاء</h4>
                {claim.evidence.filter((item) => item.stance === 'SUPPORTING').map((item) => <EvidenceCard key={item.id} evidence={item} />)}
                <h4 className="pt-2 font-black text-rose-800">أدلة تعارض الادعاء</h4>
                {claim.evidence.filter((item) => item.stance === 'CONTRADICTING').map((item) => <EvidenceCard key={item.id} evidence={item} />)}
                {claim.evidence.every((item) => item.stance === 'NEUTRAL') && <p className="text-slate-500">لم يعثر النظام على دليل مباشر كافٍ.</p>}
              </div>}
            </article>)}
          </div>
        </div>
      </section>}
    </main>
  )
}
