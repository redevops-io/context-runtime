"""Narration may not omit or change what a structured result says. Domain-free: finance, inventory and logistics facts."""
from context_runtime.verification.narration import Fact, facts_from_hits, serve_verified, verify_narration
from context_runtime.types import Hit

FACTS = [Fact("number", "1250000.50", "USD", "total invoiced"), Fact("number", "42", None, "invoices"), Fact("identifier", "INV-00042"), Fact("date", "2024-06-30")]
STRUCT = "Total invoiced 1,250,000.50 USD across 42 invoices; largest INV-00042 dated 2024-06-30."


def test_a_faithful_narration_passes_and_is_served_as_written():
    ans = "Customers were invoiced $1,250,000.50 USD over 42 invoices. The largest was INV-00042 on 2024-06-30."
    served, rep = serve_verified(ans, FACTS, STRUCT, question="What was invoiced?")
    assert rep.status == "ok" and served == ans


def test_an_omitted_figure_replaces_the_narration_with_the_structured_result():
    served, rep = serve_verified("About 42 invoices, the largest was INV-00042 on 2024-06-30.", FACTS, STRUCT)
    assert rep.status == "replaced" and served == STRUCT and any("total invoiced" in m for m in rep.missing)


def test_a_changed_figure_is_caught_even_if_the_right_one_is_also_present():
    ans = "Total 1,250,000.50 USD over 24 invoices; largest INV-00042 on 2024-06-30."      # 42 -> 24
    served, rep = serve_verified(ans, FACTS, STRUCT)
    assert rep.status == "replaced" and "24" in rep.unsupported_numbers and served == STRUCT


def test_matching_is_on_token_boundaries_not_substrings():
    # "42" is a substring of 00042 and of 142: neither may satisfy the invoice count
    rep = verify_narration("Total 1,250,000.50 USD; INV-00042 on 2024-06-30; 142 lines.", FACTS, structured_text=STRUCT)
    assert rep.status == "replaced" and any("invoices=42" in m for m in rep.missing)
    assert not verify_narration("Total 1,250,000.50 USD; 42 invoices; INV-000421 on 2024-06-30.", FACTS, structured_text=STRUCT).passed


def test_a_unit_that_is_declared_must_accompany_the_number():
    rep = verify_narration("Total 1250000.50 over 42 invoices, INV-00042, 2024-06-30.", FACTS, structured_text=STRUCT)
    assert rep.status == "replaced" and any("USD" in m for m in rep.missing)


def test_an_explicit_abstention_is_not_forced_to_repeat_the_facts_but_may_not_invent_figures():
    assert verify_narration("The source does not disclose an approved amount for that request.", FACTS, structured_text=STRUCT).status == "abstained"
    assert verify_narration("The source does not disclose it; it was probably 9,999.", FACTS, structured_text=STRUCT).status == "replaced"


def test_citations_and_numbers_from_the_question_are_not_flagged():
    ans = "For 2024 Q2 [1]: 1,250,000.50 USD, 42 invoices, INV-00042, 2024-06-30."
    assert verify_narration(ans, FACTS, structured_text=STRUCT, question="What was invoiced in 2024 Q2?").status == "ok"


def test_no_facts_means_no_check_and_facts_come_from_hit_meta():
    assert verify_narration("anything 7", [], structured_text="").status == "no_facts"
    hs = [Hit(chunk_id="a", filename="f", text="t", score=1.0, source="s", meta={"facts": [{"kind": "number", "value": "3"}]}), Hit(chunk_id="b", filename="f", text="t", score=1.0, source="s")]
    assert facts_from_hits(hs) == [Fact("number", "3")]
