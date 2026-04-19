import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
import os

# Set Groq API key from user
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Import project-specific modules
from agents.graph import generate_care_plan
from utils.pdf_export import create_pdf

# --- PAGE SETUP ---
st.set_page_config(page_title="No-Show Predictor & Care Agent", page_icon="🏥", layout="wide")

st.markdown("""
<style>
/* ── Google Font ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Global spacing ──────────────────────────── */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero header ─────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0d9488 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header h1 {
    margin: 0 0 0.4rem 0;
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.hero-header p {
    margin: 0;
    color: rgba(255,255,255,0.8);
    font-size: 0.95rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(4px);
    color: #fff;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* ── Metric cards ────────────────────────────── */
.metric-row { display: flex; gap: 1.2rem; margin-bottom: 1.8rem; }
.metric-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 6px 16px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.metric-card .accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 14px 14px 0 0;
}
.metric-card .accent-bar.blue { background: linear-gradient(90deg, #3b82f6, #6366f1); }
.metric-card .accent-bar.red { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-card .accent-bar.green { background: linear-gradient(90deg, #10b981, #06b6d4); }
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
}
.metric-value.high { color: #dc2626; }
.metric-value.low { color: #059669; }
.metric-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.35rem;
}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: #f1f5f9;
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    font-size: 0.88rem;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Buttons ─────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0d9488, #0f766e) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(13,148,136,0.3);
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 6px 20px rgba(13,148,136,0.4);
    transform: translateY(-1px);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}

/* ── Info / Success / Alert boxes ────────────── */
.stAlert { border-radius: 12px !important; }

/* ── Section labels ──────────────────────────── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title .icon {
    font-size: 1.3rem;
}

/* ── Patient info card ───────────────────────── */
.patient-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}
.patient-card .field {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.88rem;
}
.patient-card .field:last-child { border-bottom: none; }
.patient-card .field .label { color: #64748b; font-weight: 500; }
.patient-card .field .value { color: #0f172a; font-weight: 700; }

/* ── Risk badge ──────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.risk-badge.high {
    background: #fee2e2;
    color: #dc2626;
}
.risk-badge.low {
    background: #dcfce7;
    color: #16a34a;
}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HERO HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🧬 AI-Powered Clinical Intelligence</div>
    <h1>Clinical Appointment No-Show Predictor</h1>
    <p>Upload patient data · Predict risk with ML · Generate AI care plans with LangGraph & Groq</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader("Upload Kaggle Dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown("""
    1️⃣ Upload appointment CSV  
    2️⃣ Run ML risk prediction  
    3️⃣ Review risk analysis  
    4️⃣ Generate AI care plans  
    5️⃣ Download PDF reports
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · LangGraph · Groq · Decision Tree")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # --- RUN PREDICTIONS ---
    st.write("---")
    if st.button("Run ML Risk Prediction", type="primary"):
        with st.spinner("Analyzing patient data with Decision Tree..."):
            try:
                # 1. Load Model & Scaler
                model = joblib.load('noshow_model.pkl')
                scaler = joblib.load('scaler.pkl')
                
                # 2. Preprocessing
                X = df.copy()
                X.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap'}, inplace=True)
                
                sched_dt = pd.to_datetime(X['ScheduledDay'])
                appt_dt = pd.to_datetime(X['AppointmentDay'])
                X['WaitDays'] = (appt_dt.dt.normalize() - sched_dt.dt.normalize()).dt.days
                X.loc[X['WaitDays'] < 0, 'WaitDays'] = 0
                X['Gender'] = X['Gender'].map({'M': 1, 'F': 0})
                
                # Keep original data for UI (including PatientId), but ensure we have expected columns for the model
                expected_cols = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'WaitDays']
                
                # We drop features not needed for scaler, like PatientId or IDs
                X_model = X[expected_cols]
                
                # 3. Scaling & Prediction
                X_scaled = scaler.transform(X_model)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                # 4. Store Results in Session State for Persistence
                # Saving back to original df to show in UI
                res_df = X.copy()
                res_df['PatientId'] = df.get('PatientId', df.index) # add PatientId fallback to index
                res_df['No-Show Probability'] = (probabilities * 100).round(2)
                res_df['Risk Level'] = ['High Risk' if p > 0.5 else 'Low Risk' for p in probabilities]
                
                st.session_state['results_df'] = res_df
                st.session_state['feature_importances'] = dict(zip(expected_cols, model.feature_importances_))
                
                st.success("✅ Analysis Complete! Scroll down to view results.")
                
            except Exception as e:
                st.error(f"🚨 Prediction Error: {e}")

    # --- DISPLAY RESULTS & AGENTIC CARE ---
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        
        # Calculate Metrics
        total_appointments = len(res_df)
        high_risk_count = int((res_df['Risk Level'] == 'High Risk').sum())
        low_risk_count = total_appointments - high_risk_count
        high_pct = round(high_risk_count / total_appointments * 100, 1) if total_appointments else 0
        low_pct = round(low_risk_count / total_appointments * 100, 1) if total_appointments else 0
        
        tab1, tab2 = st.tabs(["📊 Command Center", "🤖 Agentic Copilot"])
        
        # ─────────────────────────────────────────
        #  TAB 1 — Command Center
        # ─────────────────────────────────────────
        with tab1:
            # Render Metric Cards
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="accent-bar blue"></div>
                    <div class="metric-icon">📋</div>
                    <div class="metric-label">Total Appointments</div>
                    <div class="metric-value">{total_appointments:,}</div>
                    <div class="metric-sub">All records analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="accent-bar red"></div>
                    <div class="metric-icon">⚠️</div>
                    <div class="metric-label">High Risk Patients</div>
                    <div class="metric-value high">{high_risk_count:,}</div>
                    <div class="metric-sub">{high_pct}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="accent-bar green"></div>
                    <div class="metric-icon">✅</div>
                    <div class="metric-label">Low Risk Patients</div>
                    <div class="metric-value low">{low_risk_count:,}</div>
                    <div class="metric-sub">{low_pct}% of total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title"><span class="icon">📋</span> Patient Risk Table</div>', unsafe_allow_html=True)
            
            def highlight_high_risk(val):
                return 'background-color: #fee2e2' if val == 'High Risk' else ''
            
            display_cols = ['PatientId', 'Risk Level', 'No-Show Probability'] + [c for c in res_df.columns if c not in ['PatientId', 'Risk Level', 'No-Show Probability']]
            
            # Use .head(1000) to ensure Styler doesn't crash on high row counts
            styled_df = res_df[display_cols].head(1000).style.map(highlight_high_risk, subset=['Risk Level'])
            st.dataframe(styled_df, height=480)
            st.caption(f"Showing first 1,000 of {total_appointments:,} records.")
            
            # ── Feature Importance Chart ──────────
            st.markdown("---")
            st.markdown('<div class="section-title"><span class="icon">🔍</span> Global Feature Importance</div>', unsafe_allow_html=True)
            
            importances = st.session_state['feature_importances']
            importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).sort_values(by='Importance')
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.95, len(importance_df)))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, height=0.6, edgecolor='none')
            
            for bar, val in zip(bars, importance_df['Importance']):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', ha='left', fontsize=10, color='#94a3b8', fontweight='600')
            
            ax.set_xlabel('Importance Score', fontsize=11, color='#94a3b8', fontweight='600')
            ax.tick_params(axis='y', colors='#e2e8f0', labelsize=11)
            ax.tick_params(axis='x', colors='#64748b', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.set_xlim(0, importance_df['Importance'].max() * 1.25)
            
            plt.tight_layout()
            st.pyplot(fig)

        # ─────────────────────────────────────────
        #  TAB 2 — Agentic Copilot
        # ─────────────────────────────────────────
        with tab2:
            st.markdown('<div class="section-title"><span class="icon">🤖</span> Agentic Care Coordination</div>', unsafe_allow_html=True)
            st.info("Select a **High Risk** patient below to generate a personalized Care Plan using **LangGraph & Groq**.")
            
            # Filter for high-risk patients
            high_risk_patients = res_df[res_df['Risk Level'] == 'High Risk']
            
            if not high_risk_patients.empty:
                col_select, col_info = st.columns([1, 1])
                
                with col_select:
                    # Let user select by PatientId instead of index
                    selected_patient_id = st.selectbox("Select Patient ID:", high_risk_patients['PatientId'])
                    # Get that patient's row
                    patient_row = high_risk_patients[high_risk_patients['PatientId'] == selected_patient_id].iloc[0]
                
                with col_info:
                    gender_label = "Male" if patient_row['Gender'] == 1 else "Female"
                    st.markdown(f"""
                    <div class="patient-card">
                        <div class="field"><span class="label">Patient ID</span><span class="value">{selected_patient_id}</span></div>
                        <div class="field"><span class="label">Age</span><span class="value">{patient_row.get('Age', 'N/A')}</span></div>
                        <div class="field"><span class="label">Gender</span><span class="value">{gender_label}</span></div>
                        <div class="field"><span class="label">Risk Score</span><span class="value"><span class="risk-badge high">{patient_row.get('No-Show Probability', 'N/A')}%</span></span></div>
                        <div class="field"><span class="label">Wait Days</span><span class="value">{patient_row.get('WaitDays', 'N/A')}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("")
                
                if st.button("🚀 Execute AI Reasoning", type="primary"):
                    with st.spinner("Agent pipeline running (Analyze → Intervene → Compile)..."):
                        # Prepare data for Task 2 Agent (keeping numerical format)
                        patient_data_dict = {
                            "Age": patient_row['Age'],
                            "Gender": patient_row['Gender'], # already mapped to 1/0
                            "Scholarship": patient_row['Scholarship'],
                            "Hypertension": patient_row['Hypertension'],
                            "Diabetes": patient_row['Diabetes'],
                            "Alcoholism": patient_row['Alcoholism'],
                            "Handicap": patient_row['Handicap'],
                            "SMS_received": patient_row['SMS_received'],
                            "WaitDays": patient_row.get('WaitDays', 0)
                        }
                        
                        # Task 2: Call LangGraph
                        agent_resp = generate_care_plan(
                            patient_data=patient_data_dict,
                            risk_score=float(patient_row['No-Show Probability']),
                            risk_level=patient_row['Risk Level'],
                            feature_importances=st.session_state['feature_importances']
                        )
                        
                        if agent_resp.get("error"):
                            st.error(agent_resp["error"])
                        else:
                            report = agent_resp["final_report"]
                            
                            st.markdown("---")
                            st.markdown('<div class="section-title"><span class="icon">📄</span> AI-Generated Care Strategy</div>', unsafe_allow_html=True)
                            
                            r_col, i_col = st.columns(2)
                            with r_col:
                                st.info(f"**🔍 Risk Analysis**\n\n{report.get('risk_analysis', 'N/A')}")
                            with i_col:
                                st.success(f"**💊 Intervention Plan**\n\n{report.get('intervention_plan', 'N/A')}")
                            
                            st.caption(report.get('disclaimer', ''))
                            
                            # Task 3: PDF Generation
                            pdf_output = create_pdf(report)
                            st.download_button(
                                label="📥 Download Care Plan PDF",
                                data=pdf_output,
                                file_name=f"CarePlan_Patient_{selected_patient_id}.pdf",
                                mime="application/pdf"
                            )
            else:
                st.write("No High Risk patients identified in this batch.")

else:
    st.info("👈 Please upload the appointment CSV file in the sidebar to get started.")