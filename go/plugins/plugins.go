// Package plugins declares the Context Runtime seam interfaces.
package plugins

import "github.com/redevops-io/context-runtime/crtypes"

// Plugin is the common discovery surface for plugins that advertise metadata.
type Plugin interface {
	Info() PluginInfo
}

// PluginKind identifies the kind of a plugin for discovery.
type PluginKind string

const (
	PluginKindModel       PluginKind = "model"
	PluginKindReasoner    PluginKind = "reasoner"
	PluginKindStore       PluginKind = "store"
	PluginKindRetriever   PluginKind = "retriever"
	PluginKindScheduler   PluginKind = "scheduler"
	PluginKindKnowledge   PluginKind = "knowledge"
	PluginKindCompression PluginKind = "compression"
	PluginKindVerifier    PluginKind = "verifier"
	PluginKindRouter      PluginKind = "router"
	PluginKindPolicy      PluginKind = "policy"
	PluginKindPlanner     PluginKind = "planner"
)

// PluginInfo is the metadata used to discover and select plugins.
type PluginInfo struct {
	Name         string     `json:"name"`
	Kind         PluginKind `json:"kind"`
	Version      string     `json:"version"`
	Capabilities []string   `json:"capabilities"`
}

// IntentAnalyzer converts a request goal into a normalized planning intent.
type IntentAnalyzer interface {
	Analyze(goal crtypes.Goal) crtypes.Intent
}

// CandidateGenerator proposes and prunes candidate plans for an intent.
type CandidateGenerator interface {
	Generate(intent crtypes.Intent, goal crtypes.Goal) []crtypes.Candidate
	Prune(candidates []crtypes.Candidate, goal crtypes.Goal) []crtypes.Candidate
}

// ScoredCandidate is the Go form of a (Candidate, PlanScore) pair.
type ScoredCandidate struct {
	Candidate crtypes.Candidate `json:"candidate"`
	Score     crtypes.PlanScore `json:"score"`
}

// CostOptimizer scores candidates and selects the final plan.
type CostOptimizer interface {
	Score(candidate crtypes.Candidate, goal crtypes.Goal) crtypes.PlanScore
	Select(scored []ScoredCandidate, goal crtypes.Goal) crtypes.Plan
}

// FieldStatistics is calibration data for one estimated cost-model field.
type FieldStatistics struct {
	Field             string  `json:"field"`
	MeanAbsoluteError float64 `json:"mean_absolute_error"`
	Calibration       float64 `json:"calibration"`
	CILow             float64 `json:"ci_low"`
	CIHigh            float64 `json:"ci_high"`
	SampleCount       int     `json:"sample_count"`
	LastUpdated       *string `json:"last_updated"`
}

// CostModelStatistics reports estimator calibration and confidence data.
type CostModelStatistics struct {
	EstimatorVersion string            `json:"estimator_version"`
	Fields           []FieldStatistics `json:"fields"`
	Bucket           *string           `json:"bucket"`
}

// CostEstimator predicts plan scores and learns from completed traces.
type CostEstimator interface {
	Estimate(candidate crtypes.Candidate, goal crtypes.Goal) crtypes.PlanScore
	Statistics(bucket *string) CostModelStatistics
	Observe(plan crtypes.Plan, trace crtypes.Trace)
}

// ModelPlugin is transport to a single model.
type ModelPlugin interface {
	Complete(req crtypes.ModelRequest) crtypes.ModelResult
	Capabilities(model string) crtypes.ModelCapabilities
	CountTokens(text string, model string) int
	Info() PluginInfo
}

// ReasoningStrategy names a reasoner orchestration strategy.
type ReasoningStrategy string

const (
	ReasoningStrategySingleShot        ReasoningStrategy = "single_shot"
	ReasoningStrategyPlanWorkerCritic  ReasoningStrategy = "plan_worker_critic"
	ReasoningStrategyDebate            ReasoningStrategy = "debate"
	ReasoningStrategyToolLoop          ReasoningStrategy = "tool_loop"
)

// BuiltContext is a plan assembled into executable context.
type BuiltContext struct {
	Plan          crtypes.Plan           `json:"plan"`
	Graph         crtypes.ExecutionGraph `json:"graph"`
	Hits          []crtypes.Hit          `json:"hits"`
	AssembledText string                 `json:"assembled_text"`
	TokenBudget   map[string]int         `json:"token_budget"`
}

// ReasonRequest is the input to a reasoner strategy.
type ReasonRequest struct {
	Context     BuiltContext        `json:"context"`
	Strategy    ReasoningStrategy   `json:"strategy"`
	Capability  string              `json:"capability"`
	Constraints crtypes.Constraints `json:"constraints"`
}

// ReasonerPlugin is a reasoning strategy over one or more model calls.
type ReasonerPlugin interface {
	Reason(req ReasonRequest) crtypes.ModelResult
	Info() PluginInfo
}

// Retrieval identifies a retrieval implementation or source family.
type Retrieval string

const (
	RetrievalVector Retrieval = "vector"
	RetrievalBM25   Retrieval = "bm25"
	RetrievalHybrid Retrieval = "hybrid"
	RetrievalGraph  Retrieval = "graph"
	RetrievalSQL    Retrieval = "sql"
	RetrievalAPI    Retrieval = "api"
	RetrievalLogs   Retrieval = "logs"
	RetrievalFile   Retrieval = "file"
	RetrievalCode   Retrieval = "code"
)

// RetrieverPlugin searches an indexed retrieval backend.
type RetrieverPlugin interface {
	Search(query string, k int, method Retrieval) []crtypes.Hit
	Info() PluginInfo
}

// StorePlugin indexes source material for retrieval.
type StorePlugin interface {
	Index(path string) map[string]any
	Info() PluginInfo
}

// Schedule groups execution graph nodes into ordered parallel waves.
type Schedule struct {
	Waves          [][]string      `json:"waves"`
	MaxConcurrency int             `json:"max_concurrency"`
	Retry          map[string]int  `json:"retry"`
}

// SchedulerPlugin turns an execution graph into a physical execution schedule.
type SchedulerPlugin interface {
	Schedule(graph crtypes.ExecutionGraph, constraints crtypes.Constraints) Schedule
	Info() PluginInfo
}

// Compressed is compressed text plus provenance.
type Compressed struct {
	Text         string   `json:"text"`
	Tokens       int      `json:"tokens"`
	DerivedFrom  []string `json:"derived_from"`
	Omitted      []string `json:"omitted"`
	RefreshAfter *string  `json:"refresh_after"`
}

// CompressionPlugin reduces text to a token target while preserving provenance.
type CompressionPlugin interface {
	Compress(text string, targetTokens int) Compressed
}

// Verdict is the verifier's pass/fail judgment for a model result.
type Verdict struct {
	Passed     bool     `json:"passed"`
	Confidence float64  `json:"confidence"`
	Findings   []string `json:"findings"`
}

// VerifierPlugin checks a model result against the chosen plan and context.
type VerifierPlugin interface {
	Verify(result crtypes.ModelResult, plan crtypes.Plan, ctx BuiltContext) Verdict
}

// ToolSpec describes an externally callable tool.
type ToolSpec struct {
	Name             string         `json:"name"`
	Description      string         `json:"description"`
	Parameters       map[string]any `json:"parameters"`
	SideEffecting    bool           `json:"side_effecting"`
	ApprovalRequired bool           `json:"approval_required"`
}

// ToolResult is the structured output from an externally callable tool.
type ToolResult struct {
	OK    bool          `json:"ok"`
	Data  any           `json:"data"`
	Hits  []crtypes.Hit `json:"hits"`
	Text  string        `json:"text"`
	Error *string       `json:"error"`
}

// ToolPlugin reaches an external system such as a SIEM, BI core, or firewall.
type ToolPlugin interface {
	Spec() ToolSpec
	Run(args map[string]any) ToolResult
}

// TraceExporter exports finalized traces out of the process.
type TraceExporter interface {
	Export(trace crtypes.Trace)
}
