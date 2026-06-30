package crtypes

// SpecVersion is the Context Runtime schema version carried by persisted boundary types.
const SpecVersion = "0.1"

// Sensitivity is the request data sensitivity classification.
type Sensitivity string

const (
	SensitivityPublic     Sensitivity = "public"
	SensitivityInternal   Sensitivity = "internal"
	SensitivityRestricted Sensitivity = "restricted"
)

// SourceKind identifies the kind of source referenced by a goal.
type SourceKind string

const (
	SourceKindDocs    SourceKind = "docs"
	SourceKindCode    SourceKind = "code"
	SourceKindLogs    SourceKind = "logs"
	SourceKindMetrics SourceKind = "metrics"
	SourceKindAPI     SourceKind = "api"
	SourceKindGraph   SourceKind = "graph"
	SourceKindMemory  SourceKind = "memory"
)

// IntentBucket is the normalized intent category assigned to a goal.
type IntentBucket string

const (
	IntentBucketExactLookup   IntentBucket = "exact_lookup"
	IntentBucketConceptual    IntentBucket = "conceptual"
	IntentBucketIncident      IntentBucket = "incident"
	IntentBucketCodeReasoning IntentBucket = "code_reasoning"
	IntentBucketSynthesis     IntentBucket = "synthesis"
	IntentBucketHighRisk      IntentBucket = "high_risk"
	IntentBucketSensitive     IntentBucket = "sensitive"
	IntentBucketMultiHop      IntentBucket = "multi_hop"
	IntentBucketUnknown       IntentBucket = "unknown"
)

// Risk is the categorical risk level attached to an intent.
type Risk string

const (
	RiskLow    Risk = "low"
	RiskMedium Risk = "medium"
	RiskHigh   Risk = "high"
)

// StepType identifies a candidate plan step.
type StepType string

const (
	StepTypeRetrieve StepType = "retrieve"
	StepTypeRerank   StepType = "rerank"
	StepTypeCompress StepType = "compress"
	StepTypeRoute    StepType = "route"
	StepTypeReason   StepType = "reason"
	StepTypeDelegate StepType = "delegate"
	StepTypeVerify   StepType = "verify"
)

// Cache indicates whether a plan or trace used the plan cache.
type Cache string

const (
	CacheHit    Cache = "hit"
	CacheMiss   Cache = "miss"
	CacheBypass Cache = "bypass"
)

// NodeKind identifies an execution graph node.
type NodeKind string

const (
	NodeKindRetrieve NodeKind = "retrieve"
	NodeKindRerank   NodeKind = "rerank"
	NodeKindCompress NodeKind = "compress"
	NodeKindRoute    NodeKind = "route"
	NodeKindReason   NodeKind = "reason"
	NodeKindDelegate NodeKind = "delegate"
	NodeKindVerify   NodeKind = "verify"
	NodeKindBranch   NodeKind = "branch"
	NodeKindLoop     NodeKind = "loop"
	NodeKindApproval NodeKind = "approval"
	NodeKindMerge    NodeKind = "merge"
	NodeKindRollback NodeKind = "rollback"
)

// EdgeKind identifies an execution graph edge.
type EdgeKind string

const (
	EdgeKindThen        EdgeKind = "then"
	EdgeKindOnSuccess   EdgeKind = "on_success"
	EdgeKindOnFailure   EdgeKind = "on_failure"
	EdgeKindOnCondition EdgeKind = "on_condition"
	EdgeKindParallel    EdgeKind = "parallel"
)

// SpanKind identifies a trace span category.
type SpanKind string

const (
	SpanKindIntent    SpanKind = "intent"
	SpanKindCandidate SpanKind = "candidate"
	SpanKindOptimize  SpanKind = "optimize"
	SpanKindSchedule  SpanKind = "schedule"
	SpanKindRetrieve  SpanKind = "retrieve"
	SpanKindReason    SpanKind = "reason"
	SpanKindCompress  SpanKind = "compress"
	SpanKindVerify    SpanKind = "verify"
	SpanKindDelegate  SpanKind = "delegate"
	SpanKindCache     SpanKind = "cache"
)

// Constraints captures hard ceilings and soft requirements for a goal.
type Constraints struct {
	MaxCostUSD          *float64           `json:"max_cost_usd"`
	MaxLatencySeconds   *float64           `json:"max_latency_seconds"`
	MaxTokens           *int               `json:"max_tokens"`
	RequireCitations    bool               `json:"require_citations"`
	RequireVerification bool               `json:"require_verification"`
	Sensitivity         Sensitivity        `json:"sensitivity"`
	WeightOverrides     map[string]float64 `json:"weight_overrides"`
}

// SourceRef identifies a source available to satisfy a goal.
type SourceRef struct {
	Name    string     `json:"name"`
	Kind    SourceKind `json:"kind"`
	URI     *string    `json:"uri"`
	Version *string    `json:"version"`
}

// Goal is the request-side objective supplied to the runtime.
type Goal struct {
	Text           string      `json:"text"`
	Sources        []SourceRef `json:"sources"`
	Constraints    Constraints `json:"constraints"`
	ConversationID *string     `json:"conversation_id"`
}

// Intent is the normalized planning intent derived from a goal.
type Intent struct {
	Bucket     IntentBucket `json:"bucket"`
	Entities   []string     `json:"entities"`
	Risk       Risk         `json:"risk"`
	Normalized string       `json:"normalized"`
	Confidence float64      `json:"confidence"`
}

// StepSpec describes one candidate plan step.
type StepSpec struct {
	Type   StepType       `json:"type"`
	Params map[string]any `json:"params"`
	Plugin *string        `json:"plugin"`
}

// Candidate is an ordered set of steps with a model tier.
type Candidate struct {
	Steps     []StepSpec `json:"steps"`
	ModelTier string     `json:"model_tier"`
}

// PlanScore is the optimizer objective and feasibility estimate for a plan.
type PlanScore struct {
	ExpectedAccuracy         float64 `json:"expected_accuracy"`
	CacheHitProbability      float64 `json:"cache_hit_probability"`
	VerificationConfidence   float64 `json:"verification_confidence"`
	CostUSD                  float64 `json:"cost_usd"`
	LatencySeconds           float64 `json:"latency_seconds"`
	Risk                     float64 `json:"risk"`
	HallucinationProbability float64 `json:"hallucination_probability"`
	ContextLoss              float64 `json:"context_loss"`
	Total                    float64 `json:"total"`
	Feasible                 bool    `json:"feasible"`
}

// Plan is the selected runtime plan and its optimizer metadata.
type Plan struct {
	Intent      Intent         `json:"intent"`
	Chosen      Candidate      `json:"chosen"`
	Score       PlanScore      `json:"score"`
	Rejected    [][2]any       `json:"rejected"`
	Cache       Cache          `json:"cache"`
	ID          string         `json:"id"`
	SpecVersion string         `json:"spec_version"`
	Extra       map[string]any `json:"extra"`
}

// Hit is one retrieved chunk returned by a retrieval adapter.
type Hit struct {
	ChunkID   string         `json:"chunk_id"`
	Filename  string         `json:"filename"`
	Text      string         `json:"text"`
	Score     float64        `json:"score"`
	CreatedAt *string        `json:"created_at"`
	Source    *string        `json:"source"`
	Meta      map[string]any `json:"meta"`
}

// ModelCapabilities describes model-side capabilities relevant to planning.
type ModelCapabilities struct {
	MaxContextTokens  int  `json:"max_context_tokens"`
	PromptCache       bool `json:"prompt_cache"`
	ToolCalling       bool `json:"tool_calling"`
	StructuredOutputs bool `json:"structured_outputs"`
	Vision            bool `json:"vision"`
}

// ModelRequest is the request sent to a model adapter.
type ModelRequest struct {
	Messages   []map[string]string `json:"messages"`
	Capability string              `json:"capability"`
	MaxTokens  int                 `json:"max_tokens"`
	System     *string             `json:"system"`
	Tools      []map[string]any    `json:"tools"`
}

// ModelResult is the response returned from a model adapter.
type ModelResult struct {
	Text             string   `json:"text"`
	Model            string   `json:"model"`
	Tier             string   `json:"tier"`
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	EstCostUSD       float64  `json:"est_cost_usd"`
	CacheHit         bool     `json:"cache_hit"`
	ModelsUsed       []string `json:"models_used"`
}

// GraphNode is a node in the execution graph IR.
type GraphNode struct {
	Kind         NodeKind       `json:"kind"`
	Params       map[string]any `json:"params"`
	Plugin       *string        `json:"plugin"`
	BudgetTokens *int           `json:"budget_tokens"`
	ID           string         `json:"id"`
}

// GraphEdge is a directed edge in the execution graph IR.
type GraphEdge struct {
	Src       string   `json:"src"`
	Dst       string   `json:"dst"`
	Kind      EdgeKind `json:"kind"`
	Condition *string  `json:"condition"`
}

// ExecutionGraph is the persisted executable graph generated from a plan.
type ExecutionGraph struct {
	Nodes       []GraphNode    `json:"nodes"`
	Edges       []GraphEdge    `json:"edges"`
	PlanID      string         `json:"plan_id"`
	ID          string         `json:"id"`
	SpecVersion string         `json:"spec_version"`
	Extra       map[string]any `json:"extra"`
}

// Span is one observability span in a trace.
type Span struct {
	Name     string         `json:"name"`
	Kind     SpanKind       `json:"kind"`
	Start    string         `json:"start"`
	End      string         `json:"end"`
	ParentID *string        `json:"parent_id"`
	Attrs    map[string]any `json:"attrs"`
	ID       string         `json:"id"`
}

// Trace records execution observations for a plan.
type Trace struct {
	PlanID               string         `json:"plan_id"`
	GoalText             string         `json:"goal_text"`
	Spans                []Span         `json:"spans"`
	ActualCostUSD        float64        `json:"actual_cost_usd"`
	ActualLatencySeconds float64        `json:"actual_latency_seconds"`
	ActualTokens         int            `json:"actual_tokens"`
	Citations            []string       `json:"citations"`
	VerificationPassed   *bool          `json:"verification_passed"`
	Cache                Cache          `json:"cache"`
	ID                   string         `json:"id"`
	SpecVersion          string         `json:"spec_version"`
	Extra                map[string]any `json:"extra"`
}
