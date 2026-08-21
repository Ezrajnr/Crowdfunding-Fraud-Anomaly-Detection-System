%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from sklearn.ensemble import IsolationForest

# Page Configuration
st.set_page_config(
    page_title="Crowdfund Fraud Guard Enterprise",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Enterprise Crowdfunding Fraud & Anomaly Detection System")
st.caption("AI-Powered Anomaly Detection & Hybrid Rule Engine | Enterprise Protection Suite")

# 1. Generate & Initialize Persistent Session Dataset in Naira (₦)
if 'df' not in st.session_state:
    np.random.seed(42)
    n_normal = 900
    n_anomalies = 100
    
    # Normal Behavior Data (₦5,000 - ₦500,000)
    amount_normal = np.random.exponential(scale=50000, size=n_normal) + 5000
    velocity_normal = np.random.poisson(lam=2, size=n_normal) + 1
    account_age_normal = np.random.randint(30, 1000, size=n_normal)
    ip_mismatch_normal = np.random.choice([0, 1], size=n_normal, p=[0.95, 0.05])
    failed_attempts_normal = np.random.choice([0, 1, 2], size=n_normal, p=[0.85, 0.12, 0.03])
    
    # Fraudulent / Anomalous Data (Extreme values up to ₦800 Million or micro card-testing)
    amount_fraud = np.concatenate([
        np.random.uniform(500000000, 900000000, size=50),
        np.random.uniform(100, 500, size=50)
    ])
    velocity_fraud = np.random.poisson(lam=12, size=n_anomalies) + 3
    account_age_fraud = np.random.randint(0, 5, size=n_anomalies)
    ip_mismatch_fraud = np.random.choice([0, 1], size=n_anomalies, p=[0.20, 0.80])
    failed_attempts_fraud = np.random.randint(3, 10, size=n_anomalies)
    
    df_normal = pd.DataFrame({
        'Amount (₦)': amount_normal,
        'Velocity (Tx/Hr)': velocity_normal,
        'Account Age (Days)': account_age_normal,
        'IP Mismatch': ip_mismatch_normal,
        'Failed Attempts': failed_attempts_normal
    })
    
    df_fraud = pd.DataFrame({
        'Amount (₦)': amount_fraud,
        'Velocity (Tx/Hr)': velocity_fraud,
        'Account Age (Days)': account_age_fraud,
        'IP Mismatch': ip_mismatch_fraud,
        'Failed Attempts': failed_attempts_fraud
    })
    
    df = pd.concat([df_normal, df_fraud]).sample(frac=1).reset_index(drop=True)
    
    # Train Initial Model
    features = ['Amount (₦)', 'Velocity (Tx/Hr)', 'Account Age (Days)', 'IP Mismatch', 'Failed Attempts']
    model = IsolationForest(contamination=0.10, random_state=42)
    model.fit(df[features])
    
    df['Risk Score (%)'] = [int(np.clip((0.5 - s) * 100, 5, 98)) for s in model.decision_function(df[features])]
    
    # Assign Initial Statuses based on score & threshold
    statuses = []
    for idx, row in df.iterrows():
        if row['Amount (₦)'] >= 500000000 or row['Risk Score (%)'] >= 70:
            statuses.append('Blocked (Fraud)')
        elif row['Risk Score (%)'] >= 40:
            statuses.append('Manual Review')
        else:
            statuses.append('Approved')
            
    df['Status'] = statuses
    df['Audit ID'] = [f"FRD-{random.randint(10000, 99999)}" for _ in range(len(df))]
    df['Timestamp'] = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(len(df))]
    
    st.session_state.df = df
    st.session_state.model = model

df = st.session_state.df
model = st.session_state.model
features = ['Amount (₦)', 'Velocity (Tx/Hr)', 'Account Age (Days)', 'IP Mismatch', 'Failed Attempts']

# 2. Executive KPI Header
total_monitored = len(df)
fraud_count = int((df['Status'] == 'Blocked (Fraud)').sum())
total_fraud_blocked_ngn = df[df['Status'] == 'Blocked (Fraud)']['Amount (₦)'].sum()
avg_risk_score = int(df['Risk Score (%)'].mean())

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Transactions Monitored", f"{total_monitored:,}")
kpi2.metric("Blocked Fraud Count", f"{fraud_count:,}", delta_color="inverse")
kpi3.metric("Prevented Fraud Volume", f"₦{total_fraud_blocked_ngn:,.2f}")
kpi4.metric("Avg System Threat Index", f"{avg_risk_score}%")

st.markdown("---")

# 3. Main Interface Tabs
tab1, tab2, tab3 = st.tabs([
    "🚨 Live Risk Inspector & Rules Engine", 
    "📊 Analytics & Threat Visualization", 
    "📋 Fraud Audit Log & Export"
])

# --- TAB 1: LIVE RISK INSPECTOR ---
with tab1:
    st.subheader("🔍 Live Transaction Risk Assessment")
    st.caption("Perform real-time analysis combining Isolation Forest ML and Automated Fraud Rules.")

    with st.form("inspection_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_amount = st.number_input("Donation Amount (₦)", min_value=100.0, max_value=1000000000.0, value=50000.0, step=5000.0)
            input_velocity = st.slider("Transaction Speed (Tx / Hour)", 1, 30, 2)
            
        with col2:
            input_age = st.number_input("Account Age (Days)", min_value=0, max_value=2000, value=180)
            input_ip = st.selectbox("IP / Billing Country Mismatch?", options=["No", "Yes"])
            
        with col3:
            input_failed = st.slider("Recent Failed Payment Retries", 0, 10, 0)
            st.write("")
            submit_btn = st.form_submit_button("🚨 Run Fraud Security Audit", use_container_width=True)

    if submit_btn:
        ip_val = 1 if input_ip == "Yes" else 0
        sample_data = [[input_amount, input_velocity, input_age, ip_val, input_failed]]
        
        raw_score = model.decision_function(sample_data)[0]
        calculated_risk_pct = int(np.clip((0.5 - raw_score) * 100, 5, 98))
        
        # Rule Overrides for Naira Values
        is_extreme_amount = input_amount >= 500000000.0  # ₦500 Million threshold
        is_new_account_high_risk = (input_age <= 3) and (input_amount > 10000000.0) # ₦10 Million on new account
        is_card_testing = (input_failed >= 4) and (input_velocity >= 10)
        
        if is_extreme_amount or is_new_account_high_risk or is_card_testing:
            calculated_risk_pct = max(calculated_risk_pct, 95)

        # Determine Status
        if calculated_risk_pct >= 70 or is_extreme_amount:
            final_status = "Blocked (Fraud)"
        elif calculated_risk_pct >= 40:
            final_status = "Manual Review"
        else:
            final_status = "Approved"

        # Log new transaction into persistent state
        new_record = pd.DataFrame([{
            'Audit ID': f"FRD-{random.randint(10000, 99999)}",
            'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Amount (₦)': input_amount,
            'Velocity (Tx/Hr)': input_velocity,
            'Account Age (Days)': input_age,
            'IP Mismatch': ip_val,
            'Failed Attempts': input_failed,
            'Risk Score (%)': calculated_risk_pct,
            'Status': final_status
        }])
        
        st.session_state.df = pd.concat([new_record, st.session_state.df]).reset_index(drop=True)

        # Output Results
        st.markdown("#### 📊 Security Verdict")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric("Calculated Threat Score", f"{calculated_risk_pct}%")
            st.progress(calculated_risk_pct / 100)
            
        with res_col2:
            if final_status == "Blocked (Fraud)":
                st.error("🚨 FRAUDULENT TRANSACTION DETECTED! TRANSACTION TERMINATED.")
                
                reasons = []
                if is_extreme_amount:
                    reasons.append("• **High-Value Threshold Violation:** Amount exceeds ₦500,000,000 threshold within a short window.")
                if is_new_account_high_risk:
                    reasons.append("• **New Account Velocity Risk:** High donation (>₦10,000,000) requested from an account <3 days old.")
                if is_card_testing:
                    reasons.append("• **Card Testing Signature:** Excessive retries combined with rapid velocity.")
                if ip_val == 1:
                    reasons.append("• **Geographic Anomaly:** IP country location does not match card billing record.")
                
                if reasons:
                    st.write("**Triggered Violations:**")
                    for r in reasons:
                        st.write(r)
                else:
                    st.write("**Triggered Violations:** ML Model detected anomalous multi-feature distribution.")
                    
                st.info("🔒 **Action Executed:** Account frozen. Incident logged to audit registry.")
                
            elif final_status == "Manual Review":
                st.warning("⚠️ MODERATE RISK SUSPICION - HELD FOR REVIEW")
                st.write("Transaction held temporarily. Automated 2-Factor Authentication request dispatched.")
            else:
                st.success("✅ TRANSACTION CLEAR - APPROVED")
                st.write("All parameters clear safety criteria.")

# --- TAB 2: ANALYTICS & VISUALIZATIONS ---
with tab2:
    st.subheader("📊 Threat Distribution & Pattern Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Amount (₦) vs Account Age (Colored by Status)**")
        st.scatter_chart(
            data=st.session_state.df,
            x='Account Age (Days)',
            y='Amount (₦)',
            color='Status'
        )
        
    with chart_col2:
        st.markdown("**Transaction Speed vs Failed Retries**")
        st.scatter_chart(
            data=st.session_state.df,
            x='Velocity (Tx/Hr)',
            y='Failed Attempts',
            color='Status'
        )

# --- TAB 3: AUDIT LOG & CSV EXPORT ---
with tab3:
    st.subheader("📋 System Audit Registry")
    
    filter_col1, filter_col2 = st.columns([2, 1])
    
    with filter_col1:
        selected_status = st.multiselect(
            "Filter Records by Security Status",
            options=["Blocked (Fraud)", "Manual Review", "Approved"],
            default=["Blocked (Fraud)", "Manual Review"]
        )
        
    filtered_df = st.session_state.df[st.session_state.df['Status'].isin(selected_status)]
    
    with filter_col2:
        st.write("")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit Log (CSV)",
            data=csv_data,
            file_name="crowdfunding_fraud_audit_log_ngn.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(
        filtered_df[['Audit ID', 'Timestamp', 'Amount (₦)', 'Velocity (Tx/Hr)', 'Account Age (Days)', 'IP Mismatch', 'Failed Attempts', 'Risk Score (%)', 'Status']],
        use_container_width=True
    )
