export type Verdict = 'SUPPORTED' | 'REFUTED' | 'NOT_ENOUGH_INFORMATION'
export type EvidenceStance = 'SUPPORTING' | 'CONTRADICTING' | 'NEUTRAL'

export interface ClassProbabilities {
  SUPPORTED: number
  REFUTED: number
  NOT_ENOUGH_INFORMATION: number
}

export interface Evidence {
  id: string
  title: string
  url: string
  snippet: string
  source: string
  published_at: string | null
  retrieved_at: string
  stance: EvidenceStance
  scores: {
    relevance: number
    source: number
    recency: number
    directness: number
    agreement: number
    overall: number
  }
  verification: {
    model_verdict: Verdict
    model_confidence: number
    class_probabilities: ClassProbabilities
    model_version: string
    rule_findings: string[]
  }
}

export interface ClaimResult {
  id: string
  claim: string
  claim_type: string
  verdict: Verdict
  confidence: number
  confidence_basis: string
  model_verdict: Verdict
  model_confidence: number
  class_probabilities: ClassProbabilities
  evidence_quality: number
  conflict_detected: boolean
  evidence: Evidence[]
  explanation: string
}

export interface FactCheckResult {
  id: string
  language: string
  verdict: Verdict
  confidence: number
  confidence_basis: string
  claims: ClaimResult[]
  evidence: Evidence[]
  explanation: string
  conflict_detected: boolean
  created_at: string
}
