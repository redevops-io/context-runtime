package plugins

import "github.com/redevops-io/context-runtime/crtypes"

type pluginStub struct{}

var _ Plugin = (*pluginStub)(nil)

func (*pluginStub) Info() PluginInfo {
	return PluginInfo{Name: "plugin", Kind: PluginKindPlanner}
}

type intentAnalyzerStub struct{}

var _ IntentAnalyzer = (*intentAnalyzerStub)(nil)

func (*intentAnalyzerStub) Analyze(goal crtypes.Goal) crtypes.Intent {
	return crtypes.Intent{}
}

type candidateGeneratorStub struct{}

var _ CandidateGenerator = (*candidateGeneratorStub)(nil)

func (*candidateGeneratorStub) Generate(intent crtypes.Intent, goal crtypes.Goal) []crtypes.Candidate {
	return nil
}

func (*candidateGeneratorStub) Prune(candidates []crtypes.Candidate, goal crtypes.Goal) []crtypes.Candidate {
	return candidates
}

type costOptimizerStub struct{}

var _ CostOptimizer = (*costOptimizerStub)(nil)

func (*costOptimizerStub) Score(candidate crtypes.Candidate, goal crtypes.Goal) crtypes.PlanScore {
	return crtypes.PlanScore{}
}

func (*costOptimizerStub) Select(scored []ScoredCandidate, goal crtypes.Goal) crtypes.Plan {
	return crtypes.Plan{}
}

type costEstimatorStub struct{}

var _ CostEstimator = (*costEstimatorStub)(nil)

func (*costEstimatorStub) Estimate(candidate crtypes.Candidate, goal crtypes.Goal) crtypes.PlanScore {
	return crtypes.PlanScore{}
}

func (*costEstimatorStub) Statistics(bucket *string) CostModelStatistics {
	return CostModelStatistics{}
}

func (*costEstimatorStub) Observe(plan crtypes.Plan, trace crtypes.Trace) {}

type modelPluginStub struct{}

var _ ModelPlugin = (*modelPluginStub)(nil)

func (*modelPluginStub) Complete(req crtypes.ModelRequest) crtypes.ModelResult {
	return crtypes.ModelResult{}
}

func (*modelPluginStub) Capabilities(model string) crtypes.ModelCapabilities {
	return crtypes.ModelCapabilities{}
}

func (*modelPluginStub) CountTokens(text string, model string) int {
	return 0
}

func (*modelPluginStub) Info() PluginInfo {
	return PluginInfo{Name: "model", Kind: PluginKindModel}
}

type reasonerPluginStub struct{}

var _ ReasonerPlugin = (*reasonerPluginStub)(nil)

func (*reasonerPluginStub) Reason(req ReasonRequest) crtypes.ModelResult {
	return crtypes.ModelResult{}
}

func (*reasonerPluginStub) Info() PluginInfo {
	return PluginInfo{Name: "reasoner", Kind: PluginKindReasoner}
}

type retrieverPluginStub struct{}

var _ RetrieverPlugin = (*retrieverPluginStub)(nil)

func (*retrieverPluginStub) Search(query string, k int, method Retrieval) []crtypes.Hit {
	return nil
}

func (*retrieverPluginStub) Info() PluginInfo {
	return PluginInfo{Name: "retriever", Kind: PluginKindRetriever}
}

type storePluginStub struct{}

var _ StorePlugin = (*storePluginStub)(nil)

func (*storePluginStub) Index(path string) map[string]any {
	return nil
}

func (*storePluginStub) Info() PluginInfo {
	return PluginInfo{Name: "store", Kind: PluginKindStore}
}

type schedulerPluginStub struct{}

var _ SchedulerPlugin = (*schedulerPluginStub)(nil)

func (*schedulerPluginStub) Schedule(graph crtypes.ExecutionGraph, constraints crtypes.Constraints) Schedule {
	return Schedule{}
}

func (*schedulerPluginStub) Info() PluginInfo {
	return PluginInfo{Name: "scheduler", Kind: PluginKindScheduler}
}

type compressionPluginStub struct{}

var _ CompressionPlugin = (*compressionPluginStub)(nil)

func (*compressionPluginStub) Compress(text string, targetTokens int) Compressed {
	return Compressed{}
}

type verifierPluginStub struct{}

var _ VerifierPlugin = (*verifierPluginStub)(nil)

func (*verifierPluginStub) Verify(result crtypes.ModelResult, plan crtypes.Plan, ctx BuiltContext) Verdict {
	return Verdict{}
}

type toolPluginStub struct{}

var _ ToolPlugin = (*toolPluginStub)(nil)

func (*toolPluginStub) Spec() ToolSpec {
	return ToolSpec{Name: "tool"}
}

func (*toolPluginStub) Run(args map[string]any) ToolResult {
	return ToolResult{OK: true}
}

type traceExporterStub struct{}

var _ TraceExporter = (*traceExporterStub)(nil)

func (*traceExporterStub) Export(trace crtypes.Trace) {}
