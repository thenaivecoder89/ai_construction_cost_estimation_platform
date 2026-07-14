# retrieval_sow_qa_tests.py

import json
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

import global_rag.scripts.retrieve_chunks as ret
import global_rag.scripts.config as config


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_text(value):
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip().lower()


def get_output_folder():
    output_folder = getattr(config, "output_folder", None)

    if output_folder is None:
        output_folder = Path.cwd() / "outputs"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def result_text_blob(results):
    text_parts = []

    for item in results:
        text_parts.append(clean_text(item.get("chunk_text")))
        text_parts.append(clean_text(item.get("section_heading")))
        text_parts.append(clean_text(item.get("source_reference")))
        text_parts.append(clean_text(item.get("corpus_zone")))
        text_parts.append(clean_text(item.get("corpus_pack")))
        text_parts.append(clean_text(item.get("document_id")))

    return " ".join(text_parts)


def term_found(text_blob, term):
    return clean_text(term) in text_blob


def validate_term_groups(text_blob, required_term_groups):
    """
    required_term_groups format:
    [
        ["term A", "term B"],   # at least one term in this group must be present
        ["term C"],             # term C must be present
    ]
    """

    missing_groups = []

    for group in required_term_groups:
        group_passed = False

        for term in group:
            if term_found(text_blob, term):
                group_passed = True
                break

        if not group_passed:
            missing_groups.append(group)

    return missing_groups


def validate_traceability(results):
    """
    Every material result should have:
    - chunk_id
    - document_id
    - source_reference
    - source_reference should identify source_type
    """

    traceability_issues = []

    for item in results:
        chunk_id = item.get("chunk_id")
        document_id = item.get("document_id")
        source_reference = item.get("source_reference")

        if not chunk_id:
            traceability_issues.append("Missing chunk_id in one or more results.")

        if not document_id:
            traceability_issues.append(f"Missing document_id for chunk_id={chunk_id}.")

        if not source_reference:
            traceability_issues.append(f"Missing source_reference for chunk_id={chunk_id}.")

        if source_reference and "source_type=" not in source_reference:
            traceability_issues.append(
                f"source_reference does not contain source_type for chunk_id={chunk_id}."
            )

    return traceability_issues


def validate_corpus_filter(results, expected_corpus_zone):
    corpus_issues = []

    if expected_corpus_zone is None:
        return corpus_issues

    for item in results:
        actual_corpus_zone = item.get("corpus_zone")

        if actual_corpus_zone != expected_corpus_zone:
            corpus_issues.append(
                f"Expected corpus_zone={expected_corpus_zone}, "
                f"but found corpus_zone={actual_corpus_zone} "
                f"for chunk_id={item.get('chunk_id')}."
            )

    return corpus_issues


def summarize_best_chunks(results, max_items=5):
    best_chunks = []

    for item in results[:max_items]:
        best_chunks.append(
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "corpus_zone": item.get("corpus_zone"),
                "corpus_pack": item.get("corpus_pack"),
                "section_heading": item.get("section_heading"),
                "source_reference": item.get("source_reference"),
                "similarity_score": item.get("similarity_score"),
                "hybrid_score": item.get("hybrid_score"),
                "keyword_score": item.get("keyword_score"),
            }
        )

    return best_chunks


# ---------------------------------------------------------------------
# SOW-aligned retrieval QA test definitions
# ---------------------------------------------------------------------

SOW_RETRIEVAL_TESTS = [
    {
        "test_id": "T01_COMPLETENESS_READINESS",
        "sow_area": "5.1 Completeness & Readiness Check",
        "query": (
            "required submission components completeness readiness missing weak open items "
            "conditions precedent final investment approval metric reconciliation merchant price "
            "offtaker credit combined downside BESS augmentation permit cyber decommissioning"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["condition", "conditions", "conditions precedent", "open items"],
            ["reconcile", "reconciliation", "metric reconciliation"],
            ["merchant-price", "merchant price", "merchant curve"],
            ["offtaker", "credit"],
            ["combined downside"],
            ["BESS augmentation", "battery augmentation"],
            ["permit", "permitting", "land"],
            ["cyber", "cybersecurity"],
            ["decommissioning"],
        ],
        "failure_meaning": (
            "The corpus is not surfacing enough evidence to support the SOW completeness/readiness check."
        ),
    },
    {
        "test_id": "T02_STRATEGY_FIT",
        "sow_area": "5.2 Strategy and Fit Assessment",
        "query": (
            "strategy fit growth priorities geography technology portfolio concentration "
            "capital allocation control rights Thailand solar storage BESS contracted revenue"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["strategy", "strategic", "growth priorities"],
            ["Thailand", "Southeast Asian", "geography"],
            ["solar", "PV"],
            ["storage", "BESS", "battery"],
            ["control", "controlling", "80%"],
            ["contracted revenue", "70%"],
            ["portfolio concentration", "counterparty concentration", "diversification"],
        ],
        "failure_meaning": (
            "Strategy-fit evidence is weak or missing. report_generation.py may need a strategy criteria config/table."
        ),
    },
    {
        "test_id": "T03_HISTORICAL_IC_QUESTIONS",
        "sow_area": "5.3 Historical IC Questioning & Institutional Memory",
        "query": (
            "historical IC questions recurring IC themes merchant exposure construction technology risk "
            "counterparty credit downside protection risk allocation merchant price curves business case "
            "seller assumptions IRR bridge accounting impact book values follow up actions"
        ),
        "corpus_zone": None,
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["IC question", "IC questions", "investment committee", "committee"],
            ["merchant exposure", "merchant price", "merchant curve"],
            ["counterparty", "offtaker credit", "creditworthiness"],
            ["downside", "downside protection"],
            ["construction", "technology risk"],
            ["IRR bridge"],
            ["accounting", "book value", "book values"],
            ["follow-up", "follow up", "follow-up actions"],
        ],
        "failure_meaning": (
            "Historical IC question memory is not sufficiently evidenced. "
            "This may require loading historical IC Q&A logs / prior follow-up actions."
        ),
    },
    {
        "test_id": "T04_RISK_IDENTIFICATION_PROMPTS",
        "sow_area": "5.4 Risk Identification and Challenge Prompts",
        "query": (
            "principal risks mitigants risk rating merchant exposure grid cost schedule BESS degradation "
            "offtaker credit land permits cybersecurity decommissioning curtailment construction technology risk"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["risk", "risks"],
            ["mitigation", "mitigants", "mitigation / status"],
            ["merchant exposure", "merchant price exposure"],
            ["grid", "grid cost", "grid interconnection"],
            ["BESS degradation", "battery degradation", "augmentation"],
            ["offtaker", "payment security"],
            ["land", "permit", "permitting"],
            ["cyber", "cybersecurity"],
            ["decommissioning"],
        ],
        "failure_meaning": (
            "Risk register/challenge prompt evidence is insufficient."
        ),
    },
    {
        "test_id": "T05_FINANCIAL_METRICS_INCONSISTENCIES",
        "sow_area": "5.1 Inconsistency Detection + 5.5 Financial Challenge",
        "query": (
            "main financial metrics returns project cost project IRR equity IRR NPV minimum DSCR "
            "EV EBITDA debt total cost year one revenue metrics differ deck memo workbook"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["total project cost", "project cost"],
            ["project IRR"],
            ["equity IRR"],
            ["NPV"],
            ["minimum DSCR", "DSCR"],
            ["EV / EBITDA", "EV EBITDA"],
            ["deck", "memo", "workbook", "metrics differ"],
            ["USD", "mn"],
        ],
        "failure_meaning": (
            "Financial metric evidence is insufficient for deterministic reconciliation."
        ),
    },
    {
        "test_id": "T06_SENSITIVITY_DOWNSIDE",
        "sow_area": "5.3 Historical IC Themes + 5.4 Downside Protection",
        "query": (
            "sensitivity results downside cases merchant price generation capex COD delay FX depreciation "
            "combined downside BESS degradation augmentation equity IRR minimum DSCR included in submission"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["sensitivity", "sensitivities"],
            ["merchant price -15", "merchant price"],
            ["generation -5", "generation"],
            ["capex +10", "capex"],
            ["COD delay", "delay"],
            ["combined downside"],
            ["BESS degradation", "augmentation"],
            ["equity IRR"],
            ["minimum DSCR", "Min_DSCR"],
            ["Included_In_Submission", "included in submission", "not included"],
        ],
        "failure_meaning": (
            "Sensitivity/downside evidence is insufficient."
        ),
    },
    {
        "test_id": "T07_MARKET_OFFTAKE_REVENUE",
        "sow_area": "Market, Offtake and Revenue Assumptions",
        "query": (
            "market offtake revenue assumptions contracted share merchant share PPA price "
            "merchant price BESS revenue payment security offtaker credit merchant curve revenue projection EBITDA"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 10000,
        "minimum_results": 5,
        "required_term_groups": [
            ["contracted share", "70%"],
            ["merchant share", "30%"],
            ["PPA price", "PPA"],
            ["merchant price", "merchant curve"],
            ["BESS revenue", "ancillary services", "capacity", "shifting"],
            ["payment security"],
            ["offtaker", "credit"],
            ["revenue projection", "total revenue", "EBITDA"],
        ],
        "failure_meaning": (
            "Market/offtake/revenue evidence is insufficient."
        ),
    },
    {
        "test_id": "T08_VALUATION_CLIENT_SIDE",
        "sow_area": "5.5 Valuation and Client-Side Benchmark Challenge",
        "query": (
            "valuation assumptions NPV EV EBITDA enterprise value first full year EBITDA "
            "discount rate cost per MW BESS equipment grid interconnection overhead contingency benchmark"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 10000,
        "minimum_results": 5,
        "required_term_groups": [
            ["valuation", "EV / EBITDA", "EV EBITDA"],
            ["NPV"],
            ["first full year EBITDA", "EBITDA"],
            ["project cost", "cost"],
            ["BESS equipment", "BESS"],
            ["grid interconnection", "grid"],
            ["overhead", "contingency", "advisory"],
            ["benchmark", "upper quartile", "high"],
        ],
        "failure_meaning": (
            "Client-side valuation/cost benchmark evidence is insufficient."
        ),
    },
    {
        "test_id": "T09_EXTERNAL_COST_BENCHMARK",
        "sow_area": "5.5 Benchmark Analysis from Whitelisted Sources",
        "query": (
            "utility scale PV plus battery overnight capital cost grid connection cost "
            "fixed operating expense variable operating expense battery storage solar PV benchmark"
        ),
        "corpus_zone": "corpus_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 10000,
        "minimum_results": 5,
        "required_term_groups": [
            ["Utility-Scale Battery Storage", "utility-scale battery"],
            ["Utility-Scale PV", "utility scale PV", "Solar - Utility PV"],
            ["PV-Plus-Battery", "PV plus battery"],
            ["Overnight Capital Cost"],
            ["Grid Connection Cost", "Spur Line Cost"],
            ["Fixed Operating Expenses", "fixed operating"],
            ["Variable Operating Expenses", "variable operating"],
        ],
        "failure_meaning": (
            "External cost benchmark evidence is insufficient."
        ),
    },
    {
        "test_id": "T10_EXTERNAL_EV_EBITDA_BENCHMARK",
        "sow_area": "5.5 EV/EBITDA / Valuation Multiple Benchmark",
        "query": (
            "renewable energy solar storage EV EBITDA enterprise value valuation multiple "
            "transaction benchmark comparable transactions listed company multiples"
        ),
        "corpus_zone": "corpus_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 10000,
        "minimum_results": 3,
        "required_term_groups": [
            ["EV / EBITDA", "EV EBITDA"],
            ["enterprise value"],
            ["valuation multiple", "multiple"],
            ["transaction benchmark", "comparable transaction", "listed company"],
        ],
        "failure_meaning": (
            "EV/EBITDA valuation benchmark evidence was not found. "
            "The final report should not claim EV/EBITDA is externally benchmarked."
        ),
    },
    {
        "test_id": "T11_CONDITIONS_PRECEDENT_OPEN_ITEMS",
        "sow_area": "5.1 Readiness + 6.1 Output Open Items",
        "query": (
            "conditions precedent open items final investment approval metric reconciliation "
            "merchant price offtaker credit combined downside BESS augmentation grid overhead advisory "
            "land permit cyber decommissioning actions"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["approve progression", "final documentation", "final investment approval"],
            ["metric reconciliation", "reconcile financial metrics", "reconciliation"],
            ["merchant-price", "merchant price", "merchant curve"],
            ["offtaker-credit", "offtaker credit", "offtaker"],
            ["combined downside"],
            ["BESS augmentation", "battery augmentation"],
            ["grid", "overhead", "advisory"],
            ["land", "permit"],
            ["cyber", "cybersecurity"],
            ["decommissioning"],
        ],
        "failure_meaning": (
            "Conditions precedent / open item evidence is insufficient."
        ),
    },
    {
        "test_id": "T12_PREVIOUS_REQUEST_ACCOMMODATION",
        "sow_area": "5.1 Previous Request Accommodation",
        "query": (
            "previous request missing data accommodated prior IC request follow up action "
            "previously requested included in current memo not accommodated"
        ),
        "corpus_zone": None,
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 3,
        "required_term_groups": [
            ["previous request", "prior request", "previously requested"],
            ["accommodated", "not accommodated", "partially accommodated"],
            ["follow-up", "follow up", "follow up action"],
        ],
        "failure_meaning": (
            "Previous request accommodation cannot be tested from current corpus. "
            "Load prior request / follow-up logs or create a tracking table."
        ),
    },
    {
        "test_id": "T13_PRIOR_DEAL_COMPARISON",
        "sow_area": "5.5 Historical Deal Comparison",
        "query": (
            "comparable prior investments historical deal comparison similarities differences "
            "prior renewable investments platform solar storage wind BESS geography stage"
        ),
        "corpus_zone": None,
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 3,
        "required_term_groups": [
            ["comparable", "similarities", "differences"],
            ["prior investment", "historical deal", "previous transaction"],
            ["solar", "storage", "BESS", "wind"],
            ["geography", "stage", "platform"],
        ],
        "failure_meaning": (
            "Comparable prior-deal evidence was not found. "
            "Load historical transaction metadata or prior IC materials."
        ),
    },
    {
        "test_id": "T14_MACRO_FX_GDP_DATA",
        "sow_area": "2 Scope - Macro / FX / GDP Data",
        "query": (
            "Thailand macroeconomic data FX projections exchange rate GDP growth inflation "
            "renewable energy market country risk"
        ),
        "corpus_zone": None,
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 3,
        "required_term_groups": [
            ["Thailand"],
            ["FX", "exchange rate", "currency"],
            ["GDP", "GDP growth"],
            ["inflation", "macroeconomic", "macro"],
            ["country risk", "market risk"],
        ],
        "failure_meaning": (
            "Macro / FX / GDP evidence was not found. "
            "Load macro datasets or connect a whitelisted macro source."
        ),
    },
    {
        "test_id": "T15_TRACEABILITY",
        "sow_area": "6.2 Traceability",
        "query": (
            "financial metrics returns risks conditions merchant price offtaker credit sensitivity "
            "project cost PPA price BESS grid permit"
        ),
        "corpus_zone": "client_data",
        "top_k": 15,
        "mode": "hybrid",
        "max_chunk_chars": 8000,
        "minimum_results": 5,
        "required_term_groups": [
            ["source_type="],
        ],
        "failure_meaning": (
            "Traceability metadata is insufficient. Every finding must cite slide/page/table/row/cell references."
        ),
    },
]


# ---------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------

def run_single_test(test_config):
    test_id = test_config["test_id"]

    test_result = {
        "test_id": test_id,
        "sow_area": test_config["sow_area"],
        "query": test_config["query"],
        "corpus_zone_filter": test_config.get("corpus_zone"),
        "status": "FAIL",
        "failure_details": [],
        "results_returned": 0,
        "best_chunks": [],
    }

    try:
        output = ret.retrieve_chunks(
            query_text=test_config["query"],
            top_k=test_config.get("top_k", 10),
            mode=test_config.get("mode", "hybrid"),
            corpus_zone=test_config.get("corpus_zone"),
            corpus_pack=test_config.get("corpus_pack"),
            document_id=test_config.get("document_id"),
            max_chunk_chars=test_config.get("max_chunk_chars", 8000),
        )

        results = output.get("results", [])
        test_result["results_returned"] = len(results)
        test_result["best_chunks"] = summarize_best_chunks(results)

        failure_details = []

        minimum_results = test_config.get("minimum_results", 1)

        if len(results) < minimum_results:
            failure_details.append(
                f"Only {len(results)} results returned; expected at least {minimum_results}."
            )

        corpus_issues = validate_corpus_filter(
            results=results,
            expected_corpus_zone=test_config.get("corpus_zone"),
        )
        failure_details.extend(corpus_issues)

        traceability_issues = validate_traceability(results)
        failure_details.extend(traceability_issues)

        text_blob = result_text_blob(results)

        missing_groups = validate_term_groups(
            text_blob=text_blob,
            required_term_groups=test_config.get("required_term_groups", []),
        )

        for missing_group in missing_groups:
            failure_details.append(
                "Missing expected evidence term group. "
                f"At least one of these terms was expected: {missing_group}"
            )

        if failure_details:
            failure_details.append(test_config.get("failure_meaning", "No additional failure meaning provided."))
            test_result["status"] = "FAIL"
            test_result["failure_details"] = failure_details
        else:
            test_result["status"] = "PASS"
            test_result["failure_details"] = []

    except Exception as e:
        test_result["status"] = "FAIL"
        test_result["failure_details"] = [
            f"Exception while running test {test_id}: {type(e).__name__}: {str(e)}",
            test_config.get("failure_meaning", "No additional failure meaning provided."),
        ]

    return test_result


def run_all_tests(sleep_seconds=1):
    all_results = []

    for test_config in SOW_RETRIEVAL_TESTS:
        print(f"Running {test_config['test_id']} - {test_config['sow_area']}")

        result = run_single_test(test_config)
        all_results.append(result)

        print(f"Result: {result['status']} | Results returned: {result['results_returned']}")
        print("-" * 80)

        time.sleep(sleep_seconds)

    return all_results


def build_summary_dataframe(all_results):
    rows = []

    for result in all_results:
        best_chunk_ids = []

        for item in result.get("best_chunks", []):
            if item.get("chunk_id"):
                best_chunk_ids.append(item.get("chunk_id"))

        rows.append(
            {
                "test_id": result.get("test_id"),
                "sow_area": result.get("sow_area"),
                "status": result.get("status"),
                "results_returned": result.get("results_returned"),
                "corpus_zone_filter": result.get("corpus_zone_filter"),
                "best_chunk_ids": ", ".join(best_chunk_ids),
                "failure_details": " | ".join(result.get("failure_details", [])),
            }
        )

    return pd.DataFrame(rows)


def build_overall_summary(all_results):
    total_tests = len(all_results)
    passed_tests = sum(1 for item in all_results if item["status"] == "PASS")
    failed_tests = total_tests - passed_tests

    overall_status = "PASS" if failed_tests == 0 else "FAIL"

    return {
        "overall_status": overall_status,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "pass_rate_pct": round((passed_tests / total_tests) * 100, 2) if total_tests else 0,
        "generated_at": datetime.utcnow().isoformat(),
    }


def save_outputs(all_results):
    output_folder = get_output_folder()

    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    summary_df = build_summary_dataframe(all_results)
    overall_summary = build_overall_summary(all_results)

    csv_path = output_folder / f"sow_retrieval_qa_summary_{run_timestamp}.csv"
    json_path = output_folder / f"sow_retrieval_qa_results_{run_timestamp}.json"

    summary_df.to_csv(csv_path, index=False)

    output_json = {
        "overall_summary": overall_summary,
        "test_results": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    return overall_summary, csv_path, json_path, summary_df


def print_summary(overall_summary, summary_df, csv_path, json_path):
    print("\n")
    print("=" * 100)
    print("SOW RETRIEVAL QA SUMMARY")
    print("=" * 100)

    print(f"Overall status : {overall_summary['overall_status']}")
    print(f"Total tests    : {overall_summary['total_tests']}")
    print(f"Passed tests   : {overall_summary['passed_tests']}")
    print(f"Failed tests   : {overall_summary['failed_tests']}")
    print(f"Pass rate      : {overall_summary['pass_rate_pct']}%")

    print("\nTest summary:")
    print(summary_df[["test_id", "sow_area", "status", "results_returned"]].to_string(index=False))

    failed_df = summary_df[summary_df["status"] == "FAIL"]

    if len(failed_df) > 0:
        print("\nFailed test details:")
        for _, row in failed_df.iterrows():
            print("-" * 100)
            print(f"Test ID: {row['test_id']}")
            print(f"SOW Area: {row['sow_area']}")
            print(f"Failure details: {row['failure_details']}")

    print("\nOutput files:")
    print(f"CSV summary : {csv_path}")
    print(f"JSON details: {json_path}")
    print("=" * 100)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    all_results = run_all_tests(sleep_seconds=1)

    overall_summary, csv_path, json_path, summary_df = save_outputs(all_results)

    print_summary(
        overall_summary=overall_summary,
        summary_df=summary_df,
        csv_path=csv_path,
        json_path=json_path,
    )

    return {
        "overall_summary": overall_summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summary_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    main()