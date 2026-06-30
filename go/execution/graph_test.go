package execution

import (
	"strings"
	"testing"

	"github.com/redevops-io/context-runtime/crtypes"
)

func TestBuildLinearGraphValidateAccepts(t *testing.T) {
	plugin := "retriever"
	candidate := crtypes.Candidate{
		Steps: []crtypes.StepSpec{
			{Type: crtypes.StepTypeRetrieve, Params: map[string]any{"target_tokens": 512}, Plugin: &plugin},
			{Type: crtypes.StepTypeRerank, Params: map[string]any{"top_k": 5}},
			{Type: crtypes.StepTypeCompress, Params: map[string]any{"target_tokens": 256}},
			{Type: crtypes.StepTypeReason, Params: map[string]any{}},
		},
	}

	graph, err := Build("plan_test", candidate)
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if err := Validate(graph); err != nil {
		t.Fatalf("Validate rejected linear graph: %v", err)
	}
	if graph.PlanID != "plan_test" {
		t.Fatalf("PlanID = %q, want %q", graph.PlanID, "plan_test")
	}
	if len(graph.Nodes) != len(candidate.Steps) {
		t.Fatalf("node count = %d, want %d", len(graph.Nodes), len(candidate.Steps))
	}
	if len(graph.Edges) != len(candidate.Steps)-1 {
		t.Fatalf("edge count = %d, want %d", len(graph.Edges), len(candidate.Steps)-1)
	}
	if graph.Nodes[0].Kind != crtypes.NodeKindRetrieve || graph.Nodes[len(graph.Nodes)-1].Kind != crtypes.NodeKindReason {
		t.Fatalf("unexpected endpoint kinds: first=%q last=%q", graph.Nodes[0].Kind, graph.Nodes[len(graph.Nodes)-1].Kind)
	}
	if graph.Nodes[0].BudgetTokens == nil || *graph.Nodes[0].BudgetTokens != 512 {
		t.Fatalf("first node BudgetTokens = %v, want 512", graph.Nodes[0].BudgetTokens)
	}
	for i, edge := range graph.Edges {
		if edge.Kind != crtypes.EdgeKindThen {
			t.Fatalf("edge %d kind = %q, want %q", i, edge.Kind, crtypes.EdgeKindThen)
		}
		if edge.Src != graph.Nodes[i].ID || edge.Dst != graph.Nodes[i+1].ID {
			t.Fatalf("edge %d = %s->%s, want %s->%s", i, edge.Src, edge.Dst, graph.Nodes[i].ID, graph.Nodes[i+1].ID)
		}
	}
}

func TestValidateRejectsDanglingEdge(t *testing.T) {
	graph, err := Build("plan_dangling", crtypes.Candidate{
		Steps: []crtypes.StepSpec{
			{Type: crtypes.StepTypeRetrieve, Params: map[string]any{}},
			{Type: crtypes.StepTypeReason, Params: map[string]any{}},
		},
	})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	graph.Edges = append(graph.Edges, crtypes.GraphEdge{
		Src:  graph.Nodes[0].ID,
		Dst:  "node_missing",
		Kind: crtypes.EdgeKindThen,
	})

	err = Validate(graph)
	if err == nil {
		t.Fatal("Validate accepted graph with dangling edge")
	}
	if !strings.Contains(err.Error(), "unknown node") {
		t.Fatalf("Validate error = %q, want unknown node", err)
	}
}

func TestValidateRejectsUnguardedCycle(t *testing.T) {
	graph := crtypes.ExecutionGraph{
		Nodes: []crtypes.GraphNode{
			{ID: "node_a", Kind: crtypes.NodeKindRetrieve, Params: map[string]any{}},
			{ID: "node_b", Kind: crtypes.NodeKindReason, Params: map[string]any{}},
		},
		Edges: []crtypes.GraphEdge{
			{Src: "node_a", Dst: "node_b", Kind: crtypes.EdgeKindThen},
			{Src: "node_b", Dst: "node_a", Kind: crtypes.EdgeKindThen},
		},
	}

	err := Validate(graph)
	if err == nil {
		t.Fatal("Validate accepted graph with unguarded cycle")
	}
	if !strings.Contains(err.Error(), "unguarded cycle") {
		t.Fatalf("Validate error = %q, want unguarded cycle", err)
	}
}
