"""
knowledge_graph.py
In-Memory Knowledge Graph Engine for Groww Mutual Fund RAG.
Built with NetworkX to model entities (Schemes, AMCs, Benchmarks, Lock-in Periods,
Expense Ratios, Exit Loads, Tax Rules, Investment Modes) and directed relations.
Enables deterministic multi-hop reasoning (e.g. Section 80C -> Groww ELSS -> NIFTY 500 TRI).
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Set, Any
import networkx as nx
from rag.retriever import Passage

class MutualFundKnowledgeGraph:
    """
    Constructs and queries an entity-relationship graph of mutual fund schemes,
    regulatory attributes, and operational rules.
    """
    def __init__(self, sources_dir: Optional[str] = None):
        if sources_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sources_dir = os.path.join(base_dir, "sources")
        self.sources_dir = sources_dir
        self.graph = nx.DiGraph()
        self.scheme_to_doc: Dict[str, str] = {}
        self.build_graph()

    def build_graph(self):
        """Constructs graph nodes and edges from structured mutual fund knowledge."""
        self.graph.clear()
        
        # 1. Core Schemes & Metadata
        schemes = [
            {
                "id": "groww_elss_tax_saver",
                "name": "Groww ELSS Tax Saver Fund",
                "category": "ELSS / Tax Saver",
                "benchmark": "NIFTY 500 Total Return Index",
                "lock_in": "3 Years",
                "expense_ratio": "1.47%",
                "exit_load": "Nil",
                "min_sip": "₹500",
                "min_lump": "₹500",
                "tax_section": "Section 80C",
                "doc_id": "groww_scheme_elss_tax_saver"
            },
            {
                "id": "groww_nifty_total_market",
                "name": "Groww Nifty Total Market Index Fund",
                "category": "Index Fund / Large & Mid & Small",
                "benchmark": "Nifty Total Market Total Return Index",
                "lock_in": "Nil",
                "expense_ratio": "0.25%",
                "exit_load": "Nil",
                "min_sip": "₹100",
                "min_lump": "₹100",
                "tax_section": "Equity STCG 20% / LTCG 12.5%",
                "doc_id": "groww_scheme_nifty_total_market"
            },
            {
                "id": "groww_multicap_fund",
                "name": "Groww Multicap Fund",
                "category": "Equity / Multi Cap",
                "benchmark": "NIFTY 500 Multicap 50:25:25 Total Return Index",
                "lock_in": "Nil",
                "expense_ratio": "0.35%",
                "exit_load": "1% if redeemed within 365 days",
                "min_sip": "₹500",
                "min_lump": "₹500",
                "tax_section": "Equity STCG 20% / LTCG 12.5%",
                "doc_id": "groww_scheme_multicap_fund"
            },
            {
                "id": "groww_liquid_fund",
                "name": "Groww Liquid Fund",
                "category": "Debt / Liquid",
                "benchmark": "CRISIL Liquid Debt Index",
                "lock_in": "Nil",
                "expense_ratio": "0.11%",
                "exit_load": "0.0070% tiered schedule within 7 days",
                "min_sip": "₹500",
                "min_lump": "₹500",
                "tax_section": "Debt Slab Rate",
                "doc_id": "groww_scheme_liquid_fund"
            },
            {
                "id": "groww_banking_financial_services",
                "name": "Groww Banking & Financial Services Fund",
                "category": "Equity / Sectoral",
                "benchmark": "NIFTY Financial Services Total Return Index",
                "lock_in": "Nil",
                "expense_ratio": "0.65%",
                "exit_load": "1% if redeemed within 365 days",
                "min_sip": "₹500",
                "min_lump": "₹500",
                "tax_section": "Equity STCG 20% / LTCG 12.5%",
                "doc_id": "groww_scheme_banking_financial_services"
            }
        ]

        # AMC Entity Node
        amc_name = "Groww Asset Management Limited"
        self.graph.add_node(amc_name, type="AMC", doc_id="groww_guide_what_is_an_amc")

        for s in schemes:
            s_name = s["name"]
            doc_id = s["doc_id"]
            self.scheme_to_doc[s_name.lower()] = doc_id
            self.scheme_to_doc[s["id"]] = doc_id

            # Add Scheme Node
            self.graph.add_node(s_name, type="Scheme", doc_id=doc_id, category=s["category"])
            self.graph.add_edge(amc_name, s_name, relation="MANAGES")
            self.graph.add_edge(s_name, amc_name, relation="MANAGED_BY")

            # Benchmark Relation
            bench = s["benchmark"]
            self.graph.add_node(bench, type="Benchmark", doc_id="groww_concept_benchmark")
            self.graph.add_edge(s_name, bench, relation="BENCHMARKED_TO")
            self.graph.add_edge(bench, s_name, relation="BENCHMARK_FOR")

            # Lock-In Relation
            lock = s["lock_in"]
            self.graph.add_node(lock, type="LockIn")
            self.graph.add_edge(s_name, lock, relation="HAS_LOCK_IN")
            self.graph.add_edge(lock, s_name, relation="LOCK_IN_OF")

            # Expense Ratio Relation
            exp = s["expense_ratio"]
            self.graph.add_node(exp, type="ExpenseRatio", doc_id="groww_concept_expense_ratio")
            self.graph.add_edge(s_name, exp, relation="HAS_EXPENSE_RATIO")

            # Exit Load Relation
            load = s["exit_load"]
            self.graph.add_node(load, type="ExitLoad", doc_id="groww_faq_exit_load_explained")
            self.graph.add_edge(s_name, load, relation="HAS_EXIT_LOAD")

            # Tax Section Relation
            tax = s["tax_section"]
            self.graph.add_node(tax, type="TaxSection", doc_id="groww_guide_mutual_fund_taxation_capital_gains")
            self.graph.add_edge(s_name, tax, relation="TAX_TREATMENT")
            self.graph.add_edge(tax, s_name, relation="APPLIES_TO")

        # 2. Conceptual Mutual Fund Entities & Relational Triples
        self._add_concept_relationships()

    def _add_concept_relationships(self):
        """Adds conceptual mutual fund edges and domain triples."""
        # Section 80C -> ELSS link
        self.graph.add_edge("Section 80C", "Groww ELSS Tax Saver Fund", relation="TAX_DEDUCTION_UPTO_1.5L")
        self.graph.add_edge("Section 80C", "3 Years", relation="STATUTORY_MINIMUM_LOCK_IN")

        # Direct Plan vs Regular Plan
        direct_node = "Direct Plan"
        regular_node = "Regular Plan"
        self.graph.add_node(direct_node, type="PlanVariant", doc_id="groww_guide_direct_vs_regular_mutual_funds")
        self.graph.add_node(regular_node, type="PlanVariant", doc_id="groww_guide_direct_vs_regular_mutual_funds")
        self.graph.add_edge(direct_node, "0% Distributor Commission", relation="COMMISSION_STRUCTURE")
        self.graph.add_edge(direct_node, "Lower Expense Ratio", relation="COST_ADVANTAGE")
        self.graph.add_edge(regular_node, "Distributor Trail Commission", relation="COMMISSION_STRUCTURE")

        # Investment Modes: SIP, Lump Sum, STP, SWP
        sip_node = "Systematic Investment Plan (SIP)"
        lump_node = "Lump Sum"
        stp_node = "Systematic Transfer Plan (STP)"
        swp_node = "Systematic Withdrawal Plan (SWP)"
        for mode in [sip_node, lump_node, stp_node, swp_node]:
            self.graph.add_node(mode, type="InvestmentMode", doc_id="groww_guide_sip_lumpsum_stp_swp")

        self.graph.add_edge(sip_node, "Rupee-Cost Averaging", relation="BENEFIT")
        self.graph.add_edge(sip_node, "Power of Compounding", relation="BENEFIT")
        self.graph.add_edge(direct_node, sip_node, relation="WORKS_WITH_AS_DIRECT_SIP")
        self.graph.add_edge(sip_node, direct_node, relation="AVAILABLE_IN_DIRECT_MODE")

        # Growth Option vs IDCW Option
        growth_node = "Growth Option"
        idcw_node = "IDCW Option"
        self.graph.add_node(growth_node, type="PlanOption", doc_id="groww_guide_growth_vs_idcw_options")
        self.graph.add_node(idcw_node, type="PlanOption", doc_id="groww_guide_growth_vs_idcw_options")
        self.graph.add_edge(growth_node, "Reinvested Profits Compounding", relation="PROFIT_TREATMENT")
        self.graph.add_edge(idcw_node, "Periodic Capital Distribution", relation="PROFIT_TREATMENT")

        # Cut-Off Timings
        cutoff_eq = "2:00 PM IST"
        cutoff_liq = "12:00 PM IST"
        self.graph.add_node(cutoff_eq, type="CutOffTime", doc_id="groww_guide_transacting_buying_selling_nav")
        self.graph.add_node(cutoff_liq, type="CutOffTime", doc_id="groww_guide_transacting_buying_selling_nav")
        self.graph.add_edge("Equity Mutual Funds", cutoff_eq, relation="PURCHASE_REDEMPTION_CUTOFF")
        self.graph.add_edge("Debt Mutual Funds", cutoff_eq, relation="PURCHASE_REDEMPTION_CUTOFF")
        self.graph.add_edge("Liquid Funds", cutoff_liq, relation="PURCHASE_REDEMPTION_CUTOFF")

        # Asset Class Comparison
        mf_node = "Mutual Funds"
        stocks_node = "Direct Stocks"
        fd_node = "Fixed Deposits (FD)"
        for asset in [mf_node, stocks_node, fd_node]:
            self.graph.add_node(asset, type="AssetClass", doc_id="groww_guide_mutual_funds_vs_stocks_fd")
        self.graph.add_edge(mf_node, "Automatic Portfolio Diversification", relation="FEATURE")
        self.graph.add_edge(mf_node, "Professional SEBI Fund Managers", relation="MANAGEMENT")
        self.graph.add_edge(stocks_node, "Individual Company Single-Stock Risk", relation="RISK")
        self.graph.add_edge(fd_node, "Guaranteed Fixed Interest with DICGC ₹5L Insurance", relation="SAFETY")

    def query_graph(self, query: str) -> Dict[str, Any]:
        """
        Parses query, extracts matched graph entities, traverses connected edges,
        and returns structured entity facts and prioritized document IDs.
        """
        q_lower = query.lower()
        matched_nodes = []
        
        # 1. Extract matching nodes from query
        for node in self.graph.nodes():
            n_lower = str(node).lower()
            if len(n_lower) >= 3 and n_lower in q_lower:
                matched_nodes.append(node)
            elif n_lower == "section 80c" and ("80c" in q_lower or "tax saver" in q_lower or "tax saving" in q_lower):
                matched_nodes.append(node)
            elif n_lower == "groww elss tax saver fund" and ("elss" in q_lower or "tax saver" in q_lower):
                matched_nodes.append(node)
            elif n_lower == "groww liquid fund" and "liquid" in q_lower:
                matched_nodes.append(node)
            elif n_lower == "groww multicap fund" and "multicap" in q_lower:
                matched_nodes.append(node)
            elif n_lower == "groww nifty total market index fund" and ("total market" in q_lower or "nifty total" in q_lower):
                matched_nodes.append(node)
            elif n_lower == "direct plan" and "direct" in q_lower:
                matched_nodes.append(node)
            elif n_lower == "systematic investment plan (sip)" and "sip" in q_lower:
                matched_nodes.append(node)
            elif n_lower == "idcw option" and "idcw" in q_lower:
                matched_nodes.append(node)
            elif n_lower == "growth option" and "growth" in q_lower and "direct growth" not in q_lower:
                matched_nodes.append(node)

        # 2. Multi-hop traversal across adjacent edges
        triples = []
        target_doc_ids: Set[str] = set()
        
        for node in matched_nodes:
            # Check node's own doc_id
            doc_id = self.graph.nodes[node].get("doc_id")
            if doc_id:
                target_doc_ids.add(doc_id)

            # Traverse outgoing edges
            for successor in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, successor)
                rel = edge_data.get("relation", "RELATED_TO")
                triples.append((node, rel, successor))
                succ_doc = self.graph.nodes[successor].get("doc_id")
                if succ_doc:
                    target_doc_ids.add(succ_doc)

            # Traverse incoming edges (for reverse multi-hop queries, e.g. "which fund has 3-year lock-in")
            for predecessor in self.graph.predecessors(node):
                edge_data = self.graph.get_edge_data(predecessor, node)
                rel = edge_data.get("relation", "RELATED_TO")
                triples.append((predecessor, rel, node))
                pred_doc = self.graph.nodes[predecessor].get("doc_id")
                if pred_doc:
                    target_doc_ids.add(pred_doc)

        return {
            "matched_entities": matched_nodes,
            "triples": triples[:8],
            "doc_ids": list(target_doc_ids)
        }
