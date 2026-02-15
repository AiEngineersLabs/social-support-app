"""
Streamlit Frontend for Social Support Application System
Guided step-by-step wizard UI for application submission and AI assessment.
"""
import streamlit as st
import requests
import time
import pandas as pd

API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="Social Support Application System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1E3A5F; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .score-card { padding: 1.2rem; border-radius: 10px; text-align: center; margin: 1rem 0; }
    .approve { background-color: #d4edda; border: 2px solid #28a745; }
    .decline { background-color: #f8d7da; border: 2px solid #dc3545; }
    .review { background-color: #fff3cd; border: 2px solid #ffc107; }

    /* Step indicator styles */
    .step-container { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
    .step-box {
        flex: 1; padding: 0.7rem 0.5rem; border-radius: 8px;
        text-align: center; font-size: 0.8rem; font-weight: 600;
    }
    .step-done { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .step-active { background: #cce5ff; color: #004085; border: 2px solid #004085; }
    .step-locked { background: #e9ecef; color: #999; border: 1px solid #dee2e6; }
    .step-number { font-size: 1.1rem; font-weight: 700; display: block; }

    .next-hint {
        background: #e8f4fd; border-left: 4px solid #2196F3;
        padding: 0.8rem 1rem; border-radius: 4px; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
def init_state():
    defaults = {
        "current_step": 1,
        "applicant_id": None,
        "applicant_name": None,
        "docs_uploaded": 0,
        "assessment_done": False,
        "chat_messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Step Progress Indicator
STEPS = [
    ("1", "New Application"),
    ("2", "Upload Documents"),
    ("3", "AI Assessment"),
    ("4", "Decision Dashboard"),
    ("5", "Chat Assistant"),
]


def render_step_indicator():
    current = st.session_state.current_step
    cols = st.columns(len(STEPS))
    for i, (num, label) in enumerate(STEPS):
        step_num = i + 1
        with cols[i]:
            if step_num < current:
                css = "step-done"
                icon = "&#10003;"
            elif step_num == current:
                css = "step-active"
                icon = num
            else:
                css = "step-locked"
                icon = num
            st.markdown(
                f'<div class="step-box {css}">'
                f'<span class="step-number">{icon}</span>{label}</div>',
                unsafe_allow_html=True,
            )


def go_to_step(step: int):
    st.session_state.current_step = step
    st.rerun()


# Sidebar
def render_sidebar():
    st.sidebar.markdown("### Social Support System")
    st.sidebar.markdown("AI-Powered Application Assessment")
    st.sidebar.markdown("---")

    current = st.session_state.current_step

    # Show completed info
    if st.session_state.applicant_id:
        st.sidebar.success(f"Applicant: **{st.session_state.applicant_name}**\nID: **{st.session_state.applicant_id}**")

    if st.session_state.docs_uploaded > 0:
        st.sidebar.info(f"Documents uploaded: **{st.session_state.docs_uploaded}/5**")

    if st.session_state.assessment_done:
        st.sidebar.success("AI Assessment: **Complete**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Steps")

    # Step navigation — allow going back to completed steps
    for i, (num, label) in enumerate(STEPS):
        step_num = i + 1
        if step_num < current:
            if st.sidebar.button(f"Step {num}: {label}", key=f"nav_{step_num}", use_container_width=True):
                go_to_step(step_num)
        elif step_num == current:
            st.sidebar.markdown(f"**-> Step {num}: {label}**")
        else:
            st.sidebar.markdown(f"<span style='color:#999'>Step {num}: {label}</span>", unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # View all applications (always accessible)
    if st.sidebar.button("View All Applications", use_container_width=True):
        st.session_state._show_all = True
        st.rerun()

    # Reset button
    if st.sidebar.button("Start New Application", use_container_width=True):
        for k in ["applicant_id", "applicant_name", "docs_uploaded", "assessment_done", "chat_messages", "_show_all"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.current_step = 1
        st.rerun()


# Step 1: New Application
def render_step1():
    st.markdown('<div class="main-header">Step 1: Submit New Application</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fill in the applicant details to begin the assessment process</div>', unsafe_allow_html=True)

    with st.form("application_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            full_name = st.text_input("Full Name *", placeholder="e.g., Ahmed Al Maktoum")
            emirates_id = st.text_input("Emirates ID *", placeholder="e.g., 784-1990-1234567-1")
            age = st.number_input("Age *", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender *", ["Male", "Female"])
            nationality = st.selectbox("Nationality *", ["UAE", "India", "Pakistan", "Philippines", "Egypt", "Jordan", "Syria", "Bangladesh", "Other"])
            marital_status = st.selectbox("Marital Status *", ["Single", "Married", "Divorced", "Widowed"])
            family_size = st.number_input("Family Size *", min_value=1, max_value=20, value=3)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=1)

        with col2:
            st.subheader("Employment & Financial")
            education_level = st.selectbox("Education Level *", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
            employment_status = st.selectbox("Employment Status *", ["Unemployed", "Part-Time", "Self-Employed", "Employed"])
            current_employer = st.text_input("Current Employer (if applicable)", placeholder="e.g., Company Name")
            years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
            monthly_income = st.number_input("Monthly Income (AED) *", min_value=0.0, max_value=100000.0, value=3000.0, step=100.0)
            st.subheader("Contact")
            address = st.text_area("Address", placeholder="Full address", height=68)
            phone = st.text_input("Phone", placeholder="+971-50-XXXXXXX")
            email = st.text_input("Email", placeholder="email@example.com")

        submitted = st.form_submit_button("Submit Application & Go to Step 2", use_container_width=True, type="primary")

        if submitted:
            if not full_name or not emirates_id:
                st.error("Full Name and Emirates ID are required.")
                return

            payload = {
                "full_name": full_name, "emirates_id": emirates_id,
                "age": age, "gender": gender, "nationality": nationality,
                "marital_status": marital_status, "family_size": family_size,
                "dependents": dependents, "education_level": education_level,
                "employment_status": employment_status,
                "current_employer": current_employer or None,
                "years_of_experience": years_of_experience,
                "monthly_income": monthly_income,
                "address": address or None, "phone": phone or None,
                "email": email or None,
            }

            try:
                resp = requests.post(f"{API_BASE}/submit-application", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.applicant_id = data["id"]
                    st.session_state.applicant_name = data["full_name"]
                    st.success(f"Application submitted! Applicant ID: **{data['id']}**")
                    time.sleep(1)
                    go_to_step(2)
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            except requests.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI server is running on port 8000.")


# Step 2: Upload Documents
def render_step2():
    st.markdown('<div class="main-header">Step 2: Upload Supporting Documents</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Upload documents for <b>{st.session_state.applicant_name}</b> '
        f'(ID: {st.session_state.applicant_id})</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    Upload the following documents. Each will be analyzed by the AI Document Processing Agent.
    Supported formats: **CSV, XLSX, TXT, PDF, PNG, JPG**
    """)

    doc_types = {
        "Bank Statement": ("bank_statement", "Bank statement showing recent transactions (CSV or Excel)"),
        "Emirates ID": ("emirates_id", "Emirates ID card scan or text file"),
        "Resume / CV": ("resume", "Applicant's resume or CV (text or PDF)"),
        "Assets & Liabilities": ("assets_liabilities", "Financial statement of assets and liabilities (Excel)"),
        "Credit Report": ("credit_report", "Credit bureau report (text or PDF)"),
    }

    uploaded_count = 0
    applicant_id = st.session_state.applicant_id

    for label, (doc_type, description) in doc_types.items():
        with st.expander(f"{label}", expanded=True):
            st.caption(description)
            file = st.file_uploader(
                f"Choose {label} file",
                key=f"upload_{doc_type}",
                type=["csv", "xlsx", "xls", "txt", "pdf", "png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if file:
                uploaded_count += 1
                if st.button(f"Upload {label}", key=f"btn_{doc_type}", use_container_width=True):
                    try:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        data = {"doc_type": doc_type}
                        resp = requests.post(
                            f"{API_BASE}/upload-document/{applicant_id}",
                            files=files, data=data, timeout=30,
                        )
                        if resp.status_code == 200:
                            st.success(f"{label} uploaded successfully!")
                            st.session_state.docs_uploaded = max(
                                st.session_state.docs_uploaded,
                                uploaded_count,
                            )
                        else:
                            st.error(f"Error: {resp.json().get('detail', 'Upload failed')}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to backend server.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Step 1", use_container_width=True):
            go_to_step(1)
    with col2:
        if st.button("Proceed to Step 3: AI Assessment", type="primary", use_container_width=True):
            st.session_state.docs_uploaded = max(st.session_state.docs_uploaded, uploaded_count)
            go_to_step(3)

    st.markdown(
        '<div class="next-hint">You can upload some or all documents. '
        "The AI will work with whatever is available. Click <b>Proceed to Step 3</b> when ready.</div>",
        unsafe_allow_html=True,
    )


# Step 3: AI Assessment
def render_step3():
    st.markdown('<div class="main-header">Step 3: AI-Powered Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Run the multi-agent AI pipeline for <b>{st.session_state.applicant_name}</b> '
        f'(ID: {st.session_state.applicant_id})</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    The assessment runs through **4 specialized AI agents** in sequence:
    """)

    agent_col1, agent_col2, agent_col3, agent_col4 = st.columns(4)
    with agent_col1:
        st.markdown("**Agent 1**\n\nDocument Processing\n\n_Extracts data from uploaded files_")
    with agent_col2:
        st.markdown("**Agent 2**\n\nData Validation\n\n_Cross-checks all information_")
    with agent_col3:
        st.markdown("**Agent 3**\n\nEligibility Assessment\n\n_ML + LLM scoring_")
    with agent_col4:
        st.markdown("**Agent 4**\n\nEnablement Recommender\n\n_Suggests programs via RAG_")

    st.markdown("---")

    if st.session_state.assessment_done:
        st.success("Assessment already completed for this applicant.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Step 2", use_container_width=True):
                go_to_step(2)
        with col2:
            if st.button("View Results (Step 4)", type="primary", use_container_width=True):
                go_to_step(4)
        return

    if st.button("Run AI Assessment", type="primary", use_container_width=True):
        applicant_id = st.session_state.applicant_id
        progress = st.progress(0)
        status = st.empty()

        agent_steps = [
            (15, "Agent 1/4: Document Processing Agent — extracting data from documents..."),
            (35, "Agent 2/4: Data Validation Agent — cross-checking information..."),
            (60, "Agent 3/4: Eligibility Assessment Agent — scoring with ML + LLM..."),
            (80, "Agent 4/4: Enablement Recommender Agent — finding suitable programs..."),
        ]

        for pct, msg in agent_steps:
            status.text(msg)
            progress.progress(pct)
            time.sleep(0.3)

        try:
            resp = requests.post(f"{API_BASE}/assess/{applicant_id}", timeout=300)
            progress.progress(100)
            status.text("Assessment complete!")

            if resp.status_code == 200:
                result = resp.json()
                decision = result.get("decision", {})
                trace = result.get("agent_trace", [])

                st.session_state.assessment_done = True

                # Show result banner
                rec = decision.get("recommendation", "REVIEW")
                if rec == "APPROVE":
                    st.markdown(
                        f'<div class="score-card approve"><h2>APPROVED</h2>'
                        f'<p>{decision.get("support_tier", "")}</p>'
                        f'<p>Eligibility Score: {decision.get("eligibility_score", 0):.1f}/100</p></div>',
                        unsafe_allow_html=True,
                    )
                elif rec == "SOFT_DECLINE":
                    st.markdown(
                        f'<div class="score-card decline"><h2>SOFT DECLINE</h2>'
                        f'<p>{decision.get("support_tier", "")}</p>'
                        f'<p>Eligibility Score: {decision.get("eligibility_score", 0):.1f}/100</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="score-card review"><h2>MANUAL REVIEW</h2>'
                        "<p>Flagged for human review due to data discrepancies</p></div>",
                        unsafe_allow_html=True,
                    )

                # Agent trace
                with st.expander("View Agent Execution Trace (Observability)"):
                    for entry in trace:
                        agent = entry.get("agent", "Unknown")
                        action = entry.get("action", "")
                        st.markdown(f"**{agent}** &rarr; `{action}`")
                        for k, v in entry.items():
                            if k not in ("agent", "action"):
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{k}: `{v}`")

                st.markdown("---")
                if st.button("View Full Decision Dashboard (Step 4)", type="primary", use_container_width=True):
                    go_to_step(4)
            else:
                st.error(f"Assessment failed: {resp.json().get('detail', 'Unknown error')}")
        except requests.ConnectionError:
            st.error("Cannot connect to backend server.")
        except requests.Timeout:
            st.error("Assessment timed out. The LLM may still be processing — try again in a moment.")

    st.markdown("---")
    if st.button("Back to Step 2", use_container_width=True):
        go_to_step(2)


# Step 4: Decision Dashboard
def render_step4():
    st.markdown('<div class="main-header">Step 4: Decision Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Detailed assessment results for <b>{st.session_state.applicant_name}</b> '
        f'(ID: {st.session_state.applicant_id})</div>',
        unsafe_allow_html=True,
    )

    applicant_id = st.session_state.applicant_id

    try:
        resp = requests.get(f"{API_BASE}/decision/{applicant_id}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            _display_decision(data)
        else:
            st.warning("No decision found. Please complete Step 3 first.")
            if st.button("Go to Step 3", type="primary"):
                go_to_step(3)
            return
    except requests.ConnectionError:
        st.error("Cannot connect to backend server.")
        return

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Step 3", use_container_width=True):
            go_to_step(3)
    with col2:
        if st.button("Proceed to Step 5: Chat with AI", type="primary", use_container_width=True):
            go_to_step(5)


def _display_decision(data: dict):
    """Display the full decision dashboard."""
    rec = data.get("recommendation", "REVIEW")

    # Top recommendation banner
    if rec == "APPROVE":
        st.success(
            f"**Recommendation: APPROVED** | Eligibility Score: {data.get('eligibility_score', 0):.1f}/100 "
            f"| Confidence: {data.get('confidence_score', 0):.1f}%"
        )
    elif rec == "SOFT_DECLINE":
        st.error(
            f"**Recommendation: SOFT DECLINE** | Eligibility Score: {data.get('eligibility_score', 0):.1f}/100 "
            f"| Confidence: {data.get('confidence_score', 0):.1f}%"
        )
    else:
        st.warning(f"**Recommendation: MANUAL REVIEW** | Eligibility Score: {data.get('eligibility_score', 0):.1f}/100")

    # Score breakdown
    st.markdown("### Score Breakdown")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Income", f"{data.get('income_score', 0):.0f}/100", help="Lower income = higher need score")
    with col2:
        st.metric("Employment", f"{data.get('employment_score', 0):.0f}/100", help="Unemployed = higher need score")
    with col3:
        st.metric("Family", f"{data.get('family_score', 0):.0f}/100", help="Larger family = higher need score")
    with col4:
        st.metric("Wealth", f"{data.get('wealth_score', 0):.0f}/100", help="Lower net worth = higher need score")
    with col5:
        st.metric("Demographic", f"{data.get('demographic_score', 0):.0f}/100", help="Age and vulnerability factors")

    # Score bar chart
    scores_df = pd.DataFrame({
        "Criterion": ["Income (30%)", "Employment (25%)", "Wealth (20%)", "Family (15%)", "Demographic (10%)"],
        "Score": [
            data.get("income_score", 0),
            data.get("employment_score", 0),
            data.get("wealth_score", 0),
            data.get("family_score", 0),
            data.get("demographic_score", 0),
        ],
    })
    st.bar_chart(scores_df.set_index("Criterion"))

    # Reasoning
    st.markdown("### AI Reasoning")
    st.markdown(data.get("reasoning", "No reasoning available."))

    # Enablement recommendations
    recs = data.get("enablement_recommendations", [])
    if recs:
        st.markdown("### Recommended Enablement Programs")
        for i, prog in enumerate(recs, 1):
            if isinstance(prog, dict):
                name = prog.get("program_name", "Unknown Program")
                priority = prog.get("priority", "medium")
                reason = prog.get("reason", "")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                st.markdown(f"{icon} **{i}. {name}** ({priority} priority)")
                if reason:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;_{reason}_")
            else:
                st.markdown(f"- {prog}")

    # Agent trace
    trace = data.get("agent_trace", [])
    if trace:
        with st.expander("Agent Execution Trace (Observability)"):
            for entry in trace:
                st.json(entry)


# Step 5: Chat Assistant
def render_step5():
    st.markdown('<div class="main-header">Step 5: Chat with AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Ask questions about the application, decision, or available support programs '
        f'for <b>{st.session_state.applicant_name}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Try asking:**
    - _"Why was my application approved/declined?"_
    - _"What training programs can I join?"_
    - _"How do I improve my eligibility?"_
    - _"Tell me about job matching services"_
    """)

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your application, eligibility, or support programs..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/chat",
                        json={"applicant_id": st.session_state.applicant_id, "message": prompt},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        answer = resp.json().get("response", "Sorry, I couldn't process your request.")
                    else:
                        answer = "Error communicating with the AI system."
                except requests.ConnectionError:
                    answer = "Cannot connect to the backend. Please ensure the server is running."
                except requests.Timeout:
                    answer = "The request timed out. Please try again."

                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    st.markdown("---")
    if st.button("Back to Decision Dashboard", use_container_width=True):
        go_to_step(4)


# All Applications View
def render_all_applications():
    st.markdown('<div class="main-header">All Applications</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Overview of all submitted applications</div>', unsafe_allow_html=True)

    if st.button("Back to Wizard", use_container_width=True):
        st.session_state._show_all = False
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/applicants", timeout=10)
        if resp.status_code == 200:
            applicants = resp.json()
            if not applicants:
                st.info("No applications submitted yet. Start with Step 1.")
                return

            df = pd.DataFrame(applicants)
            df.columns = ["ID", "Full Name", "Emirates ID", "Employment", "Monthly Income (AED)", "Has Decision"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to load applications.")
    except requests.ConnectionError:
        st.error("Cannot connect to backend server. Make sure FastAPI is running.")


# Main Router
def main():
    init_state()
    render_sidebar()

    # Show All Applications view if toggled
    if st.session_state.get("_show_all", False):
        render_all_applications()
        return

    # Step progress indicator
    render_step_indicator()

    current = st.session_state.current_step

    # Gate: steps 2-5 require applicant_id
    if current >= 2 and not st.session_state.applicant_id:
        st.warning("Please complete Step 1 first — submit an application to continue.")
        if st.button("Go to Step 1", type="primary"):
            go_to_step(1)
        return

    # Gate: steps 4-5 require assessment
    if current >= 4 and not st.session_state.assessment_done:
        st.warning("Please complete Step 3 first — run the AI assessment to view results.")
        if st.button("Go to Step 3", type="primary"):
            go_to_step(3)
        return

    # Render current step
    if current == 1:
        render_step1()
    elif current == 2:
        render_step2()
    elif current == 3:
        render_step3()
    elif current == 4:
        render_step4()
    elif current == 5:
        render_step5()


if __name__ == "__main__":
    main()
