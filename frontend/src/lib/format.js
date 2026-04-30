export const pct = (n) => `${(n ?? 0).toFixed(1)}%`;
export const fmtSec = (s) => `${(s ?? 0).toFixed(2)}s`;
export const riskColor = (label) => ({
  Low:    "text-risk-low border-risk-low/40 bg-risk-low/10",
  Medium: "text-risk-med border-risk-med/40 bg-risk-med/10",
  High:   "text-risk-high border-risk-high/40 bg-risk-high/10",
}[label] ?? "text-muted border-border bg-panel2");
