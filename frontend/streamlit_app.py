"""
Social Support AI Chatbot
=========================
Single conversational interface replacing the 5-step wizard.

Conversation phases:
  GREETING  → welcome + explain
  INTAKE    → collect applicant data via natural conversation
  CONFIRM   → show summary card, ask for confirmation
  UPLOAD    → guided document uploads within chat
  ASSESS    → trigger AI pipeline, stream agent progress
  RESULTS   → rich inline decision cards
  QA        → open-ended Q&A with full context
"""
import streamlit as st
import requests
import json
import time

API_BASE = "http://localhost:8000/api"

# ── Phase constants ──────────────────────────────────────────────────────────
PHASE_GREETING = "GREETING"
PHASE_INTAKE   = "INTAKE"
PHASE_CONFIRM  = "CONFIRM"
PHASE_UPLOAD   = "UPLOAD"
PHASE_ASSESS   = "ASSESS"
PHASE_RESULTS  = "RESULTS"
PHASE_QA       = "QA"

REQUIRED_FIELDS = [
    "full_name", "emirates_id", "age", "gender", "nationality",
    "marital_status", "family_size", "dependents",
    "education_level", "employment_status", "monthly_income",
    "total_assets", "total_liabilities", "years_of_experience",
]

FIELD_LABELS = {
    "full_name": "Full Name", "emirates_id": "Emirates ID",
    "age": "Age", "gender": "Gender", "nationality": "Nationality",
    "marital_status": "Marital Status", "family_size": "Family Size",
    "dependents": "Dependents", "education_level": "Education",
    "employment_status": "Employment Status", "monthly_income": "Monthly Income (AED)",
    "total_assets": "Total Assets (AED)", "total_liabilities": "Total Liabilities (AED)",
    "years_of_experience": "Years of Experience",
}

DOC_TYPES = {
    "bank_statement":   ("Bank Statement",       "CSV, XLSX, or PDF"),
    "emirates_id":      ("Emirates ID",          "PNG, JPG, or PDF"),
    "resume":           ("Resume / CV",          "PDF or TXT"),
    "assets_liabilities": ("Assets & Liabilities", "XLSX, CSV, or PDF"),
    "credit_report":    ("Credit Report",        "PDF or TXT"),
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Support AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Chat container */
  .stChatMessage { border-radius: 12px; }

  /* Decision cards */
  .card-approve  { background:#d4edda; border:2px solid #28a745; border-radius:12px; padding:1.2rem; margin:.5rem 0; }
  .card-decline  { background:#f8d7da; border:2px solid #dc3545; border-radius:12px; padding:1.2rem; margin:.5rem 0; }
  .card-review   { background:#fff3cd; border:2px solid #ffc107; border-radius:12px; padding:1.2rem; margin:.5rem 0; }
  .card-title    { font-size:1.5rem; font-weight:700; margin-bottom:.3rem; }
  .card-subtitle { font-size:.95rem; color:#555; }

  /* Summary card */
  .summary-card  { background:#e8f4fd; border:1px solid #2196F3; border-radius:10px; padding:1rem; margin:.5rem 0; font-size:.9rem; }

  /* Agent step */
  .agent-step    { background:#f0f0f0; border-left:4px solid #2196F3; padding:.5rem .8rem; border-radius:4px; margin:.2rem 0; font-size:.85rem; }

  /* Progress pill */
  .phase-pill    { display:inline-block; background:#2196F3; color:white; border-radius:20px; padding:.2rem .8rem; font-size:.75rem; font-weight:600; }

  /* Upload zone */
  .upload-hint   { background:#f8f9fa; border:1px dashed #ccc; border-radius:8px; padding:.8rem; text-align:center; font-size:.85rem; color:#666; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase":            PHASE_GREETING,
        "messages":         [],           # {"role": "user|assistant", "content": str}
        "collected":        {},           # applicant fields gathered so far
        "applicant_id":     None,
        "applicant_name":   None,
        "docs_uploaded":    {},           # {doc_type: bool}
        "assessment_result": None,
        "show_uploader":    False,
        "current_doc_type": list(DOC_TYPES.keys())[0],
        "welcomed":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API helpers ──────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        fn = getattr(requests, method)
        r = fn(f"{API_BASE}{path}", timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"error": "Cannot connect to backend. Ensure FastAPI is running on port 8000."}
    except requests.Timeout:
        return {"error": "Request timed out. The LLM may still be processing — please wait."}
    except requests.HTTPError as e:
        try:
            return {"error": e.response.json().get("detail", str(e))}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


# ── Message helpers ──────────────────────────────────────────────────────────
def add_bot(content: str):
    st.session_state.messages.append({"role": "assistant", "content": content})


def add_user(content: str):
    st.session_state.messages.append({"role": "user", "content": content})


def set_phase(phase: str):
    st.session_state.phase = phase


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## 🏛️ Social Support AI")
    st.sidebar.caption("UAE Government — AI-Powered Assessment System")
    st.sidebar.markdown("---")

    phase = st.session_state.phase
    phase_labels = {
        PHASE_GREETING: "Welcome",
        PHASE_INTAKE:   "Collecting Information",
        PHASE_CONFIRM:  "Confirming Details",
        PHASE_UPLOAD:   "Uploading Documents",
        PHASE_ASSESS:   "Running Assessment",
        PHASE_RESULTS:  "Assessment Complete",
        PHASE_QA:       "Open Assistance",
    }
    st.sidebar.markdown(f"**Status:** `{phase_labels.get(phase, phase)}`")

    if st.session_state.applicant_name:
        st.sidebar.success(f"Applicant: **{st.session_state.applicant_name}**\nID: `{st.session_state.applicant_id}`")

    if st.session_state.docs_uploaded:
        uploaded = sum(1 for v in st.session_state.docs_uploaded.values() if v)
        st.sidebar.info(f"Documents: **{uploaded}/{len(DOC_TYPES)}** uploaded")

    # Field collection progress
    if phase in (PHASE_INTAKE, PHASE_CONFIRM):
        collected = st.session_state.collected
        done = sum(1 for f in REQUIRED_FIELDS if f in collected and collected[f] not in ("", None))
        pct = int(done / len(REQUIRED_FIELDS) * 100)
        st.sidebar.progress(pct / 100, text=f"Fields: {done}/{len(REQUIRED_FIELDS)}")

    st.sidebar.markdown("---")

    # Langfuse observability link
    st.sidebar.markdown("**Observability**")
    st.sidebar.markdown("[Langfuse Dashboard →](http://localhost:3000)", unsafe_allow_html=False)

    st.sidebar.markdown("---")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New Application", use_container_width=True):
            _reset_session()
            st.rerun()
    with col2:
        if st.button("View All", use_container_width=True):
            st.session_state._show_all = True
            st.rerun()


def _reset_session():
    keys_to_clear = [
        "phase", "messages", "collected", "applicant_id", "applicant_name",
        "docs_uploaded", "assessment_result", "show_uploader", "welcomed",
        "current_doc_type", "_show_all",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


# ── All applications view ────────────────────────────────────────────────────
def render_all_applications():
    st.markdown("## All Applications")
    if st.button("← Back to Chat"):
        st.session_state._show_all = False
        st.rerun()

    data = api("get", "/applicants")
    if "error" in data:
        st.error(data["error"])
        return
    if not data:
        st.info("No applications submitted yet.")
        return

    import pandas as pd
    df = pd.DataFrame(data)
    df.columns = ["ID", "Full Name", "Emirates ID", "Employment", "Monthly Income (AED)", "Has Decision"]
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Greeting ─────────────────────────────────────────────────────────────────
GREETING_MSG = """**Welcome to the UAE Social Support AI Assistant** 🏛️

I'm your AI-powered guide for the social support application process. I can help you:

- **Apply** for financial and economic support
- **Upload** your supporting documents (bank statement, Emirates ID, CV, etc.)
- **Get an instant AI assessment** of your eligibility
- **Understand** your results and recommended programs
- **Ask questions** about support programs and next steps

Everything is processed locally and securely.

To get started, could you please tell me your **full name**?"""


# ── Intake: call /api/chat-intake ────────────────────────────────────────────
def handle_intake_message(user_msg: str) -> str:
    collected = st.session_state.collected

    result = api("post", "/chat-intake", json={
        "message": user_msg,
        "collected_fields": collected,
        "conversation_history": st.session_state.messages[-6:],
    })

    if "error" in result:
        return f"I had trouble processing that. {result['error']} Could you please try again?"

    # Merge extracted fields
    extracted = result.get("extracted_fields", {})
    for k, v in extracted.items():
        if v not in ("", None):
            st.session_state.collected[k] = v

    if result.get("is_complete"):
        set_phase(PHASE_CONFIRM)
        return _build_confirm_message()

    return result.get("next_question", "Could you please share more details?")


def _build_confirm_message() -> str:
    c = st.session_state.collected
    lines = ["Great! Here's a summary of what I've collected. Please confirm everything looks correct:\n"]
    lines.append("---")
    for field in REQUIRED_FIELDS:
        label = FIELD_LABELS.get(field, field)
        val = c.get(field, "—")
        if field == "monthly_income" and isinstance(val, (int, float)):
            val = f"AED {float(val):,.0f}"
        lines.append(f"**{label}:** {val}")
    lines.append("---")
    lines.append("\nIs all of this correct? Reply **Yes** to submit your application, or tell me what needs to be corrected.")
    return "\n".join(lines)


# ── Confirm ──────────────────────────────────────────────────────────────────
def handle_confirm_message(user_msg: str) -> str:
    msg_lower = user_msg.lower().strip()

    # User confirmed
    if any(word in msg_lower for word in ("yes", "correct", "confirm", "ok", "looks good", "right")):
        return _submit_application()

    # User wants correction
    set_phase(PHASE_INTAKE)
    return (
        "No problem! Let's correct that. Which field needs updating? "
        "For example, you can say \"My age is 32\" or \"I am actually Part-Time employed\"."
    )


def _submit_application() -> str:
    c = st.session_state.collected
    payload = {
        "full_name":          str(c.get("full_name", "")),
        "emirates_id":        str(c.get("emirates_id", "")),
        "age":                int(c.get("age", 0)),
        "gender":             str(c.get("gender", "Male")),
        "nationality":        str(c.get("nationality", "UAE")),
        "marital_status":     str(c.get("marital_status", "Single")),
        "family_size":        int(c.get("family_size", 1)),
        "dependents":         int(c.get("dependents", 0)),
        "education_level":    str(c.get("education_level", "High School")),
        "employment_status":  str(c.get("employment_status", "Unemployed")),
        "monthly_income":     float(c.get("monthly_income", 0)),
        "years_of_experience": float(c.get("years_of_experience", 0)),
    }

    result = api("post", "/submit-application", json=payload)
    if "error" in result:
        # Handle duplicate Emirates ID gracefully
        if "already exists" in result["error"]:
            return (
                "It looks like an application with this Emirates ID already exists in our system. "
                "Would you like me to look up your existing application? "
                "If you want to start fresh, please say 'reset'."
            )
        return f"I couldn't submit the application: {result['error']} Please check your details."

    st.session_state.applicant_id = result["id"]
    st.session_state.applicant_name = result["full_name"]
    set_phase(PHASE_UPLOAD)
    st.session_state.show_uploader = True
    st.session_state.docs_uploaded = {k: False for k in DOC_TYPES}

    return (
        f"**Application submitted!** ✅ Your reference number is **#{result['id']}**.\n\n"
        "Now I need some supporting documents. Please upload them using the panel below. "
        "You can upload any or all of these:\n\n"
        + "\n".join(f"- **{label}** ({fmt})" for label, fmt in DOC_TYPES.values())
        + "\n\nWhen you're ready, say **\"done\"** or **\"run assessment\"** to proceed."
    )


# ── Upload phase ─────────────────────────────────────────────────────────────
def handle_upload_message(user_msg: str) -> str:
    msg_lower = user_msg.lower()
    if any(word in msg_lower for word in ("done", "ready", "assess", "run", "next", "proceed", "start")):
        uploaded_count = sum(1 for v in st.session_state.docs_uploaded.values() if v)
        if uploaded_count == 0:
            return (
                "You haven't uploaded any documents yet. **At least one document is required** "
                "before I can run the assessment.\n\n"
                "Please upload one or more of these:\n\n"
                + "\n".join(f"- **{label}** ({fmt})" for label, fmt in DOC_TYPES.values())
            )
        set_phase(PHASE_ASSESS)
        st.session_state.show_uploader = False
        return (
            f"I have {uploaded_count} document(s) uploaded. "
            "Starting the AI assessment now — this takes about 30–90 seconds depending on your hardware.\n\n"
            "I'll update you as each agent completes its work... ⏳"
        )
    return (
        "Please upload your documents using the panel below. "
        "When you're done, say **\"run assessment\"** or click the button above."
    )


def render_upload_panel():
    """Render the document upload widget below the chat."""
    if st.session_state.phase != PHASE_UPLOAD or not st.session_state.show_uploader:
        return

    st.markdown("---")
    st.markdown("### 📎 Upload Documents")

    cols = st.columns([2, 3, 2])
    with cols[0]:
        doc_options = {label: key for key, (label, _) in DOC_TYPES.items()}
        selected_label = st.selectbox(
            "Document Type",
            options=list(doc_options.keys()),
            key="doc_type_select",
        )
        selected_key = doc_options[selected_label]

    with cols[1]:
        uploaded_file = st.file_uploader(
            f"Choose {selected_label}",
            type=["csv", "xlsx", "xls", "txt", "pdf", "png", "jpg", "jpeg"],
            key=f"uploader_{selected_key}",
            label_visibility="collapsed",
        )

    with cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if uploaded_file and st.button("Upload", key=f"btn_{selected_key}", type="primary", use_container_width=True):
            with st.spinner(f"Uploading {selected_label}..."):
                result = api(
                    "post",
                    f"/upload-document/{st.session_state.applicant_id}",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"doc_type": selected_key},
                )
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.docs_uploaded[selected_key] = True
                add_bot(f"✅ **{selected_label}** uploaded and queued for processing.")
                st.rerun()

    # Show upload status
    status_cols = st.columns(len(DOC_TYPES))
    for i, (key, (label, _)) in enumerate(DOC_TYPES.items()):
        with status_cols[i]:
            icon = "✅" if st.session_state.docs_uploaded.get(key) else "⬜"
            st.caption(f"{icon} {label}")

    # Quick-proceed button
    uploaded_count = sum(1 for v in st.session_state.docs_uploaded.values() if v)
    if uploaded_count == 0:
        st.button("⬜ Upload at least 1 document to proceed", type="secondary", use_container_width=True, disabled=True)
    else:
        if st.button("✅ Done uploading — Run Assessment", type="primary", use_container_width=True):
            set_phase(PHASE_ASSESS)
            st.session_state.show_uploader = False
            add_bot(
                f"Running AI assessment with {uploaded_count} document(s)... ⏳"
            )
            st.rerun()

    st.markdown("---")


# ── Assessment ────────────────────────────────────────────────────────────────
def run_assessment() -> str:
    """Call the /api/assess endpoint and return a formatted result message."""
    applicant_id = st.session_state.applicant_id

    # Show live agent progress as chat messages
    agent_steps = [
        ("🔍 Agent 1/4: Document Processing Agent", "Extracting data from uploaded files using ReAct reasoning..."),
        ("🔎 Agent 2/4: Data Validation Agent",    "Cross-checking information with Reflexion self-critique..."),
        ("🧮 Agent 3/4: Eligibility Assessment Agent", "Scoring with ML (GradientBoosting) + LLM reasoning..."),
        ("💡 Agent 4/4: Enablement Recommender Agent", "Querying policy database (RAG) for suitable programs..."),
    ]

    progress_placeholder = st.empty()
    for label, detail in agent_steps:
        progress_placeholder.info(f"{label}\n\n_{detail}_")
        time.sleep(0.4)

    # Call assessment API
    result = api("post", f"/assess/{applicant_id}")
    progress_placeholder.empty()

    if "error" in result:
        if "already assessed" in result["error"].lower():
            # Fetch existing decision
            result = api("get", f"/decision/{applicant_id}")
            if "error" not in result:
                st.session_state.assessment_result = result
                set_phase(PHASE_RESULTS)
                return _format_results_message(result)
        return f"Assessment error: {result['error']}"

    decision = result.get("decision", result)
    st.session_state.assessment_result = decision
    set_phase(PHASE_RESULTS)
    return _format_results_message(decision)


def _format_results_message(decision: dict) -> str:
    """Format the decision as a rich chat message."""
    rec = decision.get("recommendation", "REVIEW")
    score = decision.get("eligibility_score", 0)
    tier = decision.get("support_tier", "")
    confidence = decision.get("confidence_score", 0)

    # Recommendation header
    if rec == "APPROVE":
        header = f"✅ **APPROVED — {tier}**"
        status_line = f"Your application has been approved for government social support."
    elif rec == "SOFT_DECLINE":
        header = f"⚠️ **SOFT DECLINE — {tier}**"
        status_line = "You currently don't qualify for direct financial support, but economic enablement programs are available."
    else:
        header = "🔍 **MANUAL REVIEW REQUIRED**"
        status_line = "Your application has been flagged for human review due to data inconsistencies."

    # Score breakdown
    scores = [
        ("Income",      decision.get("income_score", 0),      "30%"),
        ("Employment",  decision.get("employment_score", 0),  "25%"),
        ("Wealth",      decision.get("wealth_score", 0),      "20%"),
        ("Family",      decision.get("family_score", 0),      "15%"),
        ("Demographic", decision.get("demographic_score", 0), "10%"),
    ]
    score_lines = "\n".join(
        f"  • {name} ({weight}): **{val:.0f}/100**" for name, val, weight in scores
    )

    # Programs
    programs = decision.get("enablement_recommendations", [])
    prog_lines = ""
    if programs:
        prog_items = []
        for p in programs[:4]:
            if isinstance(p, dict):
                priority = p.get("priority", "medium")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                prog_items.append(f"  {icon} **{p.get('program_name', 'Program')}** — {p.get('reason', '')}")
            else:
                prog_items.append(f"  • {p}")
        prog_lines = "\n\n**Recommended Programmes:**\n" + "\n".join(prog_items)

    # Validation flags
    flags = decision.get("validation_flags", [])
    flag_lines = ""
    if flags:
        flag_lines = "\n\n**⚠️ Validation Flags:**\n" + "\n".join(f"  - {f}" for f in flags)

    reasoning = decision.get("reasoning", "")
    reasoning_section = f"\n\n**AI Reasoning:**\n{reasoning}" if reasoning else ""

    return (
        f"{header}\n\n"
        f"{status_line}\n\n"
        f"**Eligibility Score: {score:.1f}/100** (Confidence: {confidence:.0f}%)\n\n"
        f"**Score Breakdown:**\n{score_lines}"
        f"{prog_lines}"
        f"{flag_lines}"
        f"{reasoning_section}\n\n"
        "---\n"
        "Feel free to ask me anything about your results, the recommended programs, "
        "or how to improve your eligibility. I'm here to help! 💬"
    )


# ── Q&A phase ─────────────────────────────────────────────────────────────────
def handle_qa_message(user_msg: str) -> str:
    """Route Q&A messages to /api/chat with full context."""
    decision = st.session_state.assessment_result or {}
    result = api("post", "/chat", json={
        "applicant_id": st.session_state.applicant_id,
        "message": user_msg,
        "chat_history": st.session_state.messages[-10:],
    })
    if "error" in result:
        return f"I had trouble processing that. {result['error']}"
    return result.get("response", "I'm here to help! Could you rephrase that?")


# ── Score visualisation (rendered after results) ──────────────────────────────
def render_score_chart():
    """Display a score bar chart below the chat if results are available."""
    if st.session_state.phase not in (PHASE_RESULTS, PHASE_QA):
        return
    decision = st.session_state.assessment_result
    if not decision:
        return

    import pandas as pd
    scores_df = pd.DataFrame({
        "Criterion": ["Income (30%)", "Employment (25%)", "Wealth (20%)", "Family (15%)", "Demographic (10%)"],
        "Score": [
            decision.get("income_score", 0),
            decision.get("employment_score", 0),
            decision.get("wealth_score", 0),
            decision.get("family_score", 0),
            decision.get("demographic_score", 0),
        ],
    })

    with st.expander("📊 Score Breakdown Chart", expanded=False):
        st.bar_chart(scores_df.set_index("Criterion"))
        st.caption(f"Overall Eligibility Score: **{decision.get('eligibility_score', 0):.1f}/100**")

    # Agent trace
    trace = decision.get("agent_trace", [])
    if trace:
        with st.expander("🔍 Agent Execution Trace (ReAct + Reflexion)", expanded=False):
            for entry in trace:
                agent = entry.get("agent", "")
                action = entry.get("action", "")
                framework = entry.get("reasoning_framework", "")
                framework_tag = f" [{framework}]" if framework else ""
                st.markdown(
                    f'<div class="agent-step"><b>{agent}</b>{framework_tag} → <code>{action}</code></div>',
                    unsafe_allow_html=True,
                )
                for k, v in entry.items():
                    if k not in ("agent", "action", "reasoning_framework", "step"):
                        st.caption(f"  {k}: {v}")


# ── Main chat renderer ────────────────────────────────────────────────────────
def render_chat():
    # Render all past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Score chart + trace (after results)
    render_score_chart()

    # Upload panel (only in UPLOAD phase)
    render_upload_panel()

    # Assessment auto-trigger
    if st.session_state.phase == PHASE_ASSESS:
        with st.chat_message("assistant"):
            with st.spinner("Running AI assessment... (30–90 seconds)"):
                result_msg = run_assessment()
        add_bot(result_msg)
        st.rerun()

    # Chat input
    phase = st.session_state.phase
    placeholders = {
        PHASE_GREETING: "Say hello to start your application...",
        PHASE_INTAKE:   "Answer the question above...",
        PHASE_CONFIRM:  "Reply Yes to confirm, or tell me what to correct...",
        PHASE_UPLOAD:   "Say 'done' when finished uploading...",
        PHASE_ASSESS:   "",
        PHASE_RESULTS:  "Ask me anything about your results...",
        PHASE_QA:       "Ask me about programs, eligibility, or anything else...",
    }
    placeholder = placeholders.get(phase, "Type your message...")

    if phase == PHASE_ASSESS:
        return  # no input during assessment

    user_input = st.chat_input(placeholder)
    if not user_input:
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    add_user(user_input)

    # Route to phase handler
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if phase == PHASE_GREETING:
                # First message from user — greet and move to intake
                set_phase(PHASE_INTAKE)
                response = handle_intake_message(user_input)

            elif phase == PHASE_INTAKE:
                response = handle_intake_message(user_input)

            elif phase == PHASE_CONFIRM:
                response = handle_confirm_message(user_input)

            elif phase == PHASE_UPLOAD:
                response = handle_upload_message(user_input)

            elif phase in (PHASE_RESULTS, PHASE_QA):
                set_phase(PHASE_QA)
                response = handle_qa_message(user_input)

            else:
                response = "I'm not sure how to handle that right now. Could you try again?"

        st.markdown(response)

    add_bot(response)
    st.rerun()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    init_state()
    render_sidebar()

    # All-applications view
    if st.session_state.get("_show_all"):
        render_all_applications()
        return

    # Page header
    st.markdown("## 🏛️ Social Support AI Assistant")
    st.caption("Powered by Ollama (llama3) · LangGraph · ChromaDB · Scikit-learn · Langfuse")
    st.markdown("---")

    # Auto-greet on first load
    if not st.session_state.welcomed:
        add_bot(GREETING_MSG)
        st.session_state.welcomed = True

    render_chat()


if __name__ == "__main__":
    main()
