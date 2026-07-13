from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from seminar_demo.ui_adapter import (
    ROOT,
    ReplayRun,
    discover_replay_runs,
    load_live_dataset,
    medrac_only_download_payload,
    normalize_artifact_sample,
    normalized_sample_markdown,
    run_live_dataset_sample,
    validate_live_configuration,
)


st.set_page_config(
    page_title="MedRaC Pipeline Lab",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #132a2f;
        --muted: #51666a;
        --paper: #f5f1e8;
        --teal: #0c6b64;
        --amber: #c8782b;
        --line: #d8d1c2;
    }
    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(12,107,100,.09), transparent 24rem),
            linear-gradient(180deg, #fbf9f4 0%, var(--paper) 100%);
        color: var(--ink);
    }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: var(--ink); }
    [data-testid="stSidebar"] { background: #e9e4d8; border-right: 1px solid var(--line); }
    .eyebrow { color: var(--teal); font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    .subtitle { color: var(--muted); max-width: 58rem; font-size: 1.05rem; }
    .research-note { border-left: 4px solid var(--amber); background: rgba(255,255,255,.62); padding: .9rem 1rem; border-radius: 0 .45rem .45rem 0; }
    .pipeline-grid { display: grid; grid-template-columns: repeat(7, minmax(7rem, 1fr)); gap: .45rem; margin: 1rem 0 1.5rem; }
    .pipeline-stage { border: 1px solid var(--line); border-top: 3px solid var(--teal); background: rgba(255,255,255,.72); padding: .7rem .55rem; border-radius: .45rem; text-align: center; font-size: .78rem; font-weight: 700; }
    .provenance { color: var(--muted); font-size: .78rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.66); border: 1px solid var(--line); padding: .65rem; border-radius: .45rem; }
    div[data-testid="stMetric"] * { color: var(--ink) !important; }
    [data-testid="stSidebar"] * { color: var(--ink); }
    textarea:disabled { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; opacity: 1 !important; }
    .stButton > button, .stDownloadButton > button { border-radius: .35rem; font-weight: 700; }
    :focus-visible { outline: 3px solid #d99142 !important; outline-offset: 2px; }
    @media (max-width: 900px) { .pipeline-grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3, show_spinner=False)
def cached_runs() -> tuple[list[ReplayRun], list[Path]]:
    return discover_replay_runs()


@st.cache_data(show_spinner=False)
def cached_live_dataset():
    return load_live_dataset()


def sample_label(sample: dict[str, Any]) -> str:
    index = sample.get("dataframe_index", sample.get("DataFrame Index", "?"))
    calculator_id = sample.get("calculator_id", sample.get("Calculator ID", "?"))
    name = sample.get("calculator_name", sample.get("Calculator Name", "Unknown calculator"))
    return f"Index {index} · ID {calculator_id} · {name}"


def display_input(normalized: dict[str, Any], source: str) -> None:
    st.subheader("Input sample")
    a, b, c = st.columns([1, 2, 2])
    a.metric("Source", source)
    b.metric("Calculator ID", normalized["calculator_id"])
    c.metric("Calculator", normalized.get("calculator_name") or "Unavailable")
    st.text_area(
        "Patient Note",
        value=str(normalized.get("patient_note") or ""),
        height=220,
        disabled=True,
        help=f"Artifact field: {normalized.get('patient_note_source') or 'unavailable'}",
    )
    st.text_area(
        "Question",
        value=str(normalized.get("question") or ""),
        height=130,
        disabled=True,
        help=f"Artifact field: {normalized.get('question_source') or 'unavailable'}",
    )
    if normalized["question_missing"]:
        st.warning("This artifact does not preserve a Question. It is not inferred from the Patient Note.")


def display_pipeline(normalized: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="pipeline-grid" aria-label="MedRaC pipeline stages">
          <div class="pipeline-stage">Question-only<br>RAG query</div>
          <div class="pipeline-stage">Formula<br>retrieval</div>
          <div class="pipeline-stage">Note + question<br>extraction</div>
          <div class="pipeline-stage">Code<br>generation</div>
          <div class="pipeline-stage">Safe child<br>execution</div>
          <div class="pipeline-stage">Final<br>answer</div>
          <div class="pipeline-stage">F · E · C · A<br>evaluation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    retrieval = normalized["retrieval"]
    extraction = normalized["extraction"]
    code_generation = normalized["code_generation"]
    execution = normalized["execution"]
    stepwise = normalized["stepwise"]
    tabs = st.tabs(
        [
            "Formula Retrieval",
            "Value Extraction",
            "Code Generation",
            "Safe Execution",
            "Output",
            "Step-wise Evaluation",
        ]
    )
    with tabs[0]:
        st.caption(
            "The implementation retrieves with Question only. Patient Note is used in the following extraction stage."
        )
        st.text_area(
            "Actual retrieval query",
            value=str(retrieval.get("query") or "Unavailable"),
            height=120,
            disabled=True,
        )
        st.caption(f"Query provenance: {normalized['retrieval_query_source']}")
        x, y, z = st.columns(3)
        x.metric("Status", retrieval.get("status", "unavailable"))
        y.metric("Score", retrieval.get("score", "unavailable"))
        z.metric("Rank", retrieval.get("rank", "unavailable"))
        st.text_area(
            "Retrieved formula",
            value=str(retrieval.get("formula") or "Unavailable"),
            height=220,
            disabled=True,
        )
        config = normalized["configuration"]
        st.caption(f"Embedding model: {config.get('embedding_model', 'unavailable')}")
        if normalized["index_warning"]:
            st.warning(normalized["index_warning"])
        with st.expander("Index provenance"):
            st.json(normalized["index_metadata"])
    with tabs[1]:
        st.caption("Extraction input: Patient Note + Question + retrieved formula.")
        st.metric("Status", extraction.get("status", "unavailable"))
        values = extraction.get("extracted_values")
        if isinstance(values, list) and all(isinstance(item, dict) for item in values):
            st.dataframe(values, width="stretch", hide_index=True)
        else:
            st.json(values)
        st.caption("Units and missing-variable fields are shown only when stored; current artifacts do not infer them.")
    with tabs[2]:
        st.caption("Code-generation input: retrieved formula + extracted values + Question.")
        st.metric("Status", code_generation.get("status", "unavailable"))
        st.code(str(code_generation.get("code") or "# No generated code stored"), language="python")
    with tabs[3]:
        x, y, z = st.columns(3)
        x.metric("Validation", execution.get("validation_status", "unavailable"))
        y.metric("Execution", execution.get("status", "unavailable"))
        z.metric("Duration", f"{execution.get('duration_ms')} ms" if execution.get("duration_ms") is not None else "unavailable")
        st.code(str(execution.get("result")), language=None)
        if execution.get("error"):
            st.error(str(execution["error"]))
        st.caption(f"Execution mode: {execution.get('execution_mode', 'unavailable')}")
    with tabs[4]:
        answer, truth, rule = st.columns(3)
        answer.metric("Final answer", normalized.get("final_answer") if normalized.get("final_answer") is not None else "Unavailable")
        truth.metric("Ground truth", normalized.get("ground_truth_answer") if normalized.get("ground_truth_answer") is not None else "Unavailable")
        rule.metric("Rule evaluation", normalized["rule_evaluation"].get("result", "Unavailable"))
        if normalized["runtime_error"]:
            st.error(normalized["runtime_error"])
        st.subheader("Runtime metadata")
        left, right = st.columns(2)
        left.json(normalized["token_usage"])
        right.json(normalized["request_counts"])
        with st.expander("Model configuration"):
            st.json(normalized["configuration"])
    with tabs[5]:
        if not stepwise["complete"]:
            st.info(
                f"Step-wise evaluation status: {stepwise['status']}. F/E/C/A and strict correctness are unavailable, not incorrect."
            )
        evaluation = stepwise.get("evaluation") or {}
        columns = st.columns(4)
        labels = {
            "formula": "Formula (F)",
            "extracted_values": "Extraction (E)",
            "calculation": "Calculation (C)",
            "answer": "Answer (A)",
        }
        for column, key in zip(columns, labels):
            item = evaluation.get(key) if isinstance(evaluation, dict) else None
            item = item if isinstance(item, dict) else {}
            column.metric(labels[key], item.get("result", "Unavailable"))
            if item.get("explanation"):
                column.caption(str(item["explanation"]))
        strict, first = st.columns(2)
        strict.metric(
            "Strict all-steps correct",
            "Yes" if stepwise["strict_correct"] is True else "No" if stepwise["strict_correct"] is False else "Unavailable",
        )
        first.metric("First failed step", stepwise["first_failed_step"] or ("None — all correct" if stepwise["complete"] else "Unavailable"))
        st.caption(
            "Answer correctness is hybrid: an LLM normalizes the answer, then RegEvaluator assigns the verdict."
        )


def downloads(run: ReplayRun, raw_sample: dict[str, Any], normalized: dict[str, Any]) -> None:
    st.subheader("Downloads")
    payload = medrac_only_download_payload(raw_sample)
    report = normalized_sample_markdown(
        normalized,
        run_id=str(run.manifest.get("run_id", run.path.name)),
        status=str(run.manifest.get("status", "unknown")),
    )
    left, right = st.columns(2)
    left.download_button(
        "Download MedRaC JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        file_name=f"{run.path.name}-medrac-sample.json",
        mime="application/json",
        width="stretch",
    )
    right.download_button(
        "Download Markdown report",
        data=report,
        file_name=f"{run.path.name}-medrac-sample.md",
        mime="text/markdown",
        width="stretch",
    )


st.markdown('<div class="eyebrow">Evidence-based medical calculations</div>', unsafe_allow_html=True)
st.title("MedRaC Pipeline Lab")
st.markdown(
    '<p class="subtitle">A transparent seminar view of retrieval, extraction, generated calculation code, safe execution, and F–E–C–A evaluation.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="research-note"><strong>Research demonstration only.</strong> This interface is not a clinical calculator and must not be used for patient-care decisions. Model and evaluator outputs may be incorrect.</div>',
    unsafe_allow_html=True,
)

if "live_run_path" not in st.session_state:
    st.session_state.live_run_path = None
if "live_fingerprint" not in st.session_state:
    st.session_state.live_fingerprint = None

mode = st.sidebar.radio("Mode", ["Replay", "Dataset live demo"])
selected_run: ReplayRun | None = None
selected_raw_sample: dict[str, Any] | None = None
source_label = mode

if mode == "Replay":
    runs, skipped = cached_runs()
    if skipped:
        st.sidebar.warning(f"Skipped {len(skipped)} incomplete or invalid artifact directories.")
    if not runs:
        st.error("No complete MedRaC artifacts were found.")
        st.stop()
    default_run = next(
        (
            index
            for index, run in enumerate(runs)
            if str(run.manifest.get("status", "")).startswith("completed")
        ),
        0,
    )
    selected_run = st.sidebar.selectbox(
        "Artifact run",
        runs,
        index=default_run,
        format_func=lambda run: run.label,
    )
    selected_raw_sample = st.sidebar.selectbox(
        "Sample",
        selected_run.samples,
        format_func=sample_label,
    )
    source_label = f"Replay · {selected_run.artifact_type}"
    st.sidebar.caption(f"Provenance: {selected_run.manifest.get('result_provenance', 'unknown')}")
else:
    live_status = validate_live_configuration()
    st.sidebar.subheader("Live configuration")
    st.sidebar.write(f"Token: {'configured' if live_status.token_configured else 'missing'}")
    st.sidebar.write(f"Dataset manifest: {'ready' if live_status.dataset_ready else 'invalid'}")
    st.sidebar.write(f"Canonical index: {'ready' if live_status.index_ready else 'invalid'}")
    for message in live_status.messages:
        st.sidebar.caption(message)
    manifest, selected = cached_live_dataset()
    records = selected.to_dict(orient="records")
    selected_row = st.sidebar.selectbox("Deterministic sample", records, format_func=sample_label)
    enable_stepwise = st.sidebar.checkbox(
        "Enable step-wise evaluation",
        value=False,
        help="Adds four evaluator chat requests.",
    )
    if enable_stepwise:
        st.sidebar.warning("Expected total: 6 chat requests + 1 embedding request.")
    else:
        st.sidebar.info("Expected total: 2 chat requests + 1 embedding request.")
    fingerprint = f"{selected_row['DataFrame Index']}:{enable_stepwise}:openai/gpt-4.1:42"
    run_clicked = st.sidebar.button(
        "Run one MedRaC sample",
        type="primary",
        disabled=not live_status.ready,
        width="stretch",
    )
    if st.sidebar.button("Reset live result", width="stretch"):
        st.session_state.live_run_path = None
        st.session_state.live_fingerprint = None
        st.rerun()
    if run_clicked:
        if st.session_state.live_fingerprint == fingerprint and st.session_state.live_run_path:
            st.warning("This exact live request already ran. Reset the live result before running it again.")
        else:
            status_box = st.status("Starting deterministic MedRaC run…", expanded=True)
            stage_labels = {
                "retrieval": "Formula retrieved from the Question query",
                "extraction": "Values extracted from Patient Note + Question + formula",
                "code_generation": "Calculation code generated",
                "execution": "Generated code processed by the safe executor",
                "rule_evaluation": "Rule-based final answer evaluated",
                "stepwise_evaluation": "F–E–C–A evaluation completed",
                "failed": "Run stopped with a preserved partial result",
                "completed": "MedRaC run completed",
            }

            def update_progress(stage: str, _: dict[str, Any]) -> None:
                status_box.update(label=stage_labels.get(stage, stage))

            try:
                selected_run = run_live_dataset_sample(
                    int(selected_row["DataFrame Index"]),
                    use_llm_evaluation=enable_stepwise,
                    progress_callback=update_progress,
                )
                st.session_state.live_run_path = str(selected_run.path)
                st.session_state.live_fingerprint = fingerprint
                final_state = "complete" if selected_run.manifest["status"] == "completed" else "error"
                status_box.update(label=f"Saved {selected_run.path.name}", state=final_state, expanded=False)
            except Exception as exc:
                status_box.update(label="Live run could not start", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
    if selected_run is None and st.session_state.live_run_path:
        path = Path(st.session_state.live_run_path)
        manifest_payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        sample_payload = json.loads((path / "samples.json").read_text(encoding="utf-8"))
        selected_run = ReplayRun(path, "demo", manifest_payload, sample_payload)
    if selected_run is not None:
        selected_raw_sample = selected_run.samples[0]
        source_label = "New Streamlit live run"
    else:
        preview = {
            "dataframe_index": selected_row["DataFrame Index"],
            "row_number": selected_row["Row Number"],
            "calculator_id": selected_row["Calculator ID"],
            "calculator_name": selected_row["Calculator Name"],
            "category": selected_row["Category"],
            "input": {
                "patient_note": selected_row["Patient Note"],
                "question": selected_row["Question"],
            },
            "retrieval": {"status": "not_run", "query": selected_row["Question"]},
            "extraction": {"status": "not_run", "extracted_values": None},
            "code_generation": {"status": "not_run", "code": None},
            "execution": {"status": "not_run", "validation_status": "not_run", "execution_mode": "safe_ast_child_process"},
            "rule_evaluation": {"status": "not_run", "result": None},
            "stepwise_evaluation": {"status": "not_run", "evaluation": None},
            "first_failed_step": None,
            "final_answer": None,
            "ground_truth_answer": selected_row["Ground Truth Answer"],
            "token_usage": {},
            "request_counts": {},
        }
        selected_raw_sample = preview
        selected_run = ReplayRun(
            ROOT,
            "preview",
            {
                "run_id": "not-run",
                "status": "not_run",
                "result_provenance": "deterministic dataset preview",
                "configuration": {
                    "generator_model": "openai/gpt-4.1",
                    "evaluator_model": "openai/gpt-4.1",
                    "embedding_model": "openai/text-embedding-3-small",
                    "use_rag": True,
                    "use_llm_evaluation": enable_stepwise,
                    "execution_mode": "safe_ast_child_process",
                },
            },
            [preview],
        )
        source_label = "Dataset preview · not run"

normalized = normalize_artifact_sample(selected_raw_sample, selected_run.manifest)
display_input(normalized, source_label)
display_pipeline(normalized)
downloads(selected_run, selected_raw_sample, normalized)

st.caption(
    "MedRaC uses Question for formula retrieval, then uses Patient Note + Question + retrieved formula for value extraction."
)
