import type { FactCheckResult } from './types'

export async function checkFact(text: string): Promise<FactCheckResult> {
  const response = await fetch('/api/v1/fact-check', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, language: 'ar'}),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? 'تعذر إكمال عملية التحقق')
  }
  return response.json() as Promise<FactCheckResult>
}
