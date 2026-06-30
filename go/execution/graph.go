package execution

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"math"

	"github.com/redevops-io/context-runtime/crtypes"
)

var stepToNodeKind = map[crtypes.StepType]crtypes.NodeKind{
	crtypes.StepTypeRetrieve: crtypes.NodeKindRetrieve,
	crtypes.StepTypeRerank:   crtypes.NodeKindRerank,
	crtypes.StepTypeCompress: crtypes.NodeKindCompress,
	crtypes.StepTypeRoute:    crtypes.NodeKindRoute,
	crtypes.StepTypeReason:   crtypes.NodeKindReason,
	crtypes.StepTypeDelegate: crtypes.NodeKindDelegate,
	crtypes.StepTypeVerify:   crtypes.NodeKindVerify,
}

// Build compiles an ordered candidate into a linear execution graph.
func Build(planID string, candidate crtypes.Candidate) (crtypes.ExecutionGraph, error) {
	nodes := make([]crtypes.GraphNode, 0, len(candidate.Steps))
	for _, step := range candidate.Steps {
		kind, ok := stepToNodeKind[step.Type]
		if !ok {
			return crtypes.ExecutionGraph{}, fmt.Errorf("unknown step type %q", step.Type)
		}

		nodeID, err := newID("node")
		if err != nil {
			return crtypes.ExecutionGraph{}, fmt.Errorf("create node id: %w", err)
		}
		params := cloneParams(step.Params)
		nodes = append(nodes, crtypes.GraphNode{
			Kind:         kind,
			Params:       params,
			Plugin:       step.Plugin,
			BudgetTokens: budgetTokens(params),
			ID:           nodeID,
		})
	}

	edges := make([]crtypes.GraphEdge, 0)
	if len(nodes) > 1 {
		edges = make([]crtypes.GraphEdge, 0, len(nodes)-1)
		for i := 0; i < len(nodes)-1; i++ {
			edges = append(edges, crtypes.GraphEdge{
				Src:  nodes[i].ID,
				Dst:  nodes[i+1].ID,
				Kind: crtypes.EdgeKindThen,
			})
		}
	}

	graphID, err := newID("xg")
	if err != nil {
		return crtypes.ExecutionGraph{}, fmt.Errorf("create execution graph id: %w", err)
	}
	graph := crtypes.ExecutionGraph{
		Nodes:       nodes,
		Edges:       edges,
		PlanID:      planID,
		ID:          graphID,
		SpecVersion: crtypes.SpecVersion,
		Extra:       map[string]any{},
	}
	if err := Validate(graph); err != nil {
		return crtypes.ExecutionGraph{}, err
	}
	return graph, nil
}

// Validate enforces execution graph validity before scheduling or caching.
func Validate(graph crtypes.ExecutionGraph) error {
	ids := make(map[string]struct{}, len(graph.Nodes))
	for _, node := range graph.Nodes {
		if _, exists := ids[node.ID]; exists {
			return errors.New("duplicate node ids")
		}
		ids[node.ID] = struct{}{}
	}

	for _, edge := range graph.Edges {
		if _, ok := ids[edge.Src]; !ok {
			return fmt.Errorf("edge references unknown node: %s->%s", edge.Src, edge.Dst)
		}
		if _, ok := ids[edge.Dst]; !ok {
			return fmt.Errorf("edge references unknown node: %s->%s", edge.Src, edge.Dst)
		}
		if edge.Kind == crtypes.EdgeKindOnCondition && edge.Condition == nil {
			return errors.New("on_condition edge missing condition guard")
		}
	}

	for _, node := range graph.Nodes {
		if node.Kind == crtypes.NodeKindLoop {
			if _, ok := node.Params["max_iters"]; !ok {
				return fmt.Errorf("loop node %s missing max_iters guard", node.ID)
			}
		}
		if node.Kind == crtypes.NodeKindRollback {
			if _, ok := node.Params["compensates"]; !ok {
				return fmt.Errorf("rollback node %s must name nodes it compensates", node.ID)
			}
		}
	}

	return checkAcyclic(graph)
}

func checkAcyclic(graph crtypes.ExecutionGraph) error {
	adj := make(map[string][]string, len(graph.Nodes))
	loopIDs := make(map[string]struct{})
	for _, node := range graph.Nodes {
		adj[node.ID] = nil
		if node.Kind == crtypes.NodeKindLoop {
			loopIDs[node.ID] = struct{}{}
		}
	}

	for _, edge := range graph.Edges {
		if _, isLoop := loopIDs[edge.Dst]; isLoop && edge.Kind == crtypes.EdgeKindOnCondition {
			continue
		}
		adj[edge.Src] = append(adj[edge.Src], edge.Dst)
	}

	color := make(map[string]int, len(graph.Nodes))
	var dfs func(string) error
	dfs = func(nodeID string) error {
		color[nodeID] = 1
		for _, nextID := range adj[nodeID] {
			switch color[nextID] {
			case 1:
				return errors.New("execution graph has an unguarded cycle")
			case 0:
				if err := dfs(nextID); err != nil {
					return err
				}
			}
		}
		color[nodeID] = 2
		return nil
	}

	for _, node := range graph.Nodes {
		if color[node.ID] == 0 {
			if err := dfs(node.ID); err != nil {
				return err
			}
		}
	}
	return nil
}

func cloneParams(params map[string]any) map[string]any {
	cloned := make(map[string]any, len(params))
	for key, value := range params {
		cloned[key] = value
	}
	return cloned
}

func budgetTokens(params map[string]any) *int {
	value, ok := params["target_tokens"]
	if !ok || value == nil {
		return nil
	}

	switch tokens := value.(type) {
	case int:
		return intPtr(tokens)
	case int8:
		return intPtr(int(tokens))
	case int16:
		return intPtr(int(tokens))
	case int32:
		return intPtr(int(tokens))
	case int64:
		if int64(int(tokens)) == tokens {
			return intPtr(int(tokens))
		}
	case uint:
		converted := int(tokens)
		if converted >= 0 && uint(converted) == tokens {
			return intPtr(converted)
		}
	case uint8:
		return intPtr(int(tokens))
	case uint16:
		return intPtr(int(tokens))
	case uint32:
		converted := int(tokens)
		if converted >= 0 && uint32(converted) == tokens {
			return intPtr(converted)
		}
	case uint64:
		converted := int(tokens)
		if converted >= 0 && uint64(converted) == tokens {
			return intPtr(converted)
		}
	case float32:
		return wholeFloatPtr(float64(tokens))
	case float64:
		return wholeFloatPtr(tokens)
	}
	return nil
}

func wholeFloatPtr(value float64) *int {
	if math.IsNaN(value) || math.IsInf(value, 0) || value != math.Trunc(value) {
		return nil
	}
	converted := int(value)
	if float64(converted) != value {
		return nil
	}
	return &converted
}

func intPtr(value int) *int {
	return &value
}

func newID(kind string) (string, error) {
	var random [6]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "", err
	}
	return kind + "_" + hex.EncodeToString(random[:]), nil
}
