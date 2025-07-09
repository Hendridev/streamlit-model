import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots # FIX: Added missing import
from datetime import datetime
import ast # For safely evaluating string literals

# --- Page Configuration ---
st.set_page_config(
    page_title="Log Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dcdcdc;
        margin-bottom: 10px;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
    }
    .st-tabs-container {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Log Parsing Function ---
@st.cache_data
def parse_log_file(uploaded_file):
    """Parses the uploaded log file into a pandas DataFrame, handling multiple log formats."""
    log_entries = []
    if uploaded_file is None:
        return pd.DataFrame()

    log_content = uploaded_file.getvalue().decode("utf-8").splitlines()

    # Regex patterns for different log message structures
    log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(INFO|WARNING|ERROR)\] (.*)')
    summary_pattern = re.compile(r"Prediction results:\s*(\{.*\})")
    metrics_pattern = re.compile(r"METRICS\s*\|\s*(.*)")
    array_pattern = re.compile(r"Predictions:\s*(\[.*\])")
    
    for line in log_content:
        log_match = log_pattern.match(line)
        if log_match:
            timestamp, log_level, message = log_match.groups()
            dt_object = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
            
            summary_match = summary_pattern.search(message)
            metrics_match = metrics_pattern.search(message)
            array_match = array_pattern.search(message)

            entry = {
                "timestamp": dt_object,
                "level": log_level,
                "message": message.strip(),
                "prediction_summary": None,
                "metrics": None
            }

            if summary_match:
                try:
                    entry["prediction_summary"] = ast.literal_eval(summary_match.group(1))
                except:
                    entry["prediction_summary"] = None
            elif array_match:
                try:
                    array_str = array_match.group(1)
                    ones = array_str.count('1')
                    zeros = array_str.count('0')
                    entry["prediction_summary"] = {'Stay': zeros, 'Leave': ones}
                    entry["message"] = "Prediction complete."
                except:
                    entry["prediction_summary"] = None

            if metrics_match:
                try:
                    metrics_str = metrics_match.group(1)
                    metrics_dict = {k.strip(): float(v.strip()) for k, v in (item.split(':') for item in metrics_str.split('|'))}
                    entry["metrics"] = metrics_dict
                except:
                    entry["metrics"] = None

            log_entries.append(entry)

    if not log_entries:
        return pd.DataFrame()

    df = pd.DataFrame(log_entries)
    df = df.sort_values(by="timestamp", ascending=False)
    return df

# --- Main Application ---
st.markdown('<h1 class="main-header">Interactive Log Analysis Dashboard</h1>', unsafe_allow_html=True)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Dashboard Controls")
    
    uploaded_file = st.file_uploader("Upload your app.log file", type=["log", "txt"])
    
    if uploaded_file:
        log_df = parse_log_file(uploaded_file)

        if not log_df.empty:
            st.subheader("Filter by Date")
            min_date, max_date = log_df['timestamp'].min().date(), log_df['timestamp'].max().date()
            start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

            st.subheader("Filter by Log Level")
            log_levels = log_df['level'].unique().tolist()
            selected_levels = st.multiselect("Select log levels", options=log_levels, default=log_levels)

            st.subheader("Search Logs")
            search_keyword = st.text_input("Enter keyword to search...")
            
            filtered_df = log_df[
                (log_df['timestamp'].dt.date >= start_date) &
                (log_df['timestamp'].dt.date <= end_date) &
                (log_df['level'].isin(selected_levels))
            ]

            if search_keyword:
                filtered_df = filtered_df[filtered_df['message'].str.contains(search_keyword, case=False, na=False)]
        else:
            st.warning("Could not parse any valid log entries from the file.")
            filtered_df = pd.DataFrame()
    else:
        st.info("Please upload a log file to begin analysis.")
        filtered_df = pd.DataFrame()

# --- Dashboard Body ---
if not filtered_df.empty:
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Log Entries</div>
            <div class="metric-value">{len(filtered_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        prediction_count = filtered_df[filtered_df['message'].str.contains('Prediction complete', na=False)].shape[0]
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Predictions Made</div>
            <div class="metric-value">{prediction_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        error_count = filtered_df[filtered_df['level'] == 'ERROR'].shape[0]
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Errors</div>
            <div class="metric-value">{error_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        model_loads = filtered_df[filtered_df['message'] == 'Model loaded successfully.'].shape[0]
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Model Loads</div>
            <div class="metric-value">{model_loads}</div>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Log Timeline", "Prediction Analysis", "Performance & Drift", "Preprocessing Insights", "Raw Log Data"])

    with tab1:
        st.markdown("### Log Entries Over Time")
        time_series_df = filtered_df.set_index('timestamp').resample('H').size().reset_index(name='count')
        fig = px.line(time_series_df, x='timestamp', y='count', title='Log Volume per Hour')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Prediction Distribution Analysis")
        summary_logs = filtered_df.dropna(subset=['prediction_summary'])
        if not summary_logs.empty:
            latest_summary = summary_logs.iloc[0]['prediction_summary']
            if isinstance(latest_summary, dict):
                pred_df = pd.DataFrame(list(latest_summary.items()), columns=['Prediction', 'Count'])
                fig_pie = px.pie(pred_df, names='Prediction', values='Count', 
                                title='Distribution of Latest Predictions', hole=0.3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Latest prediction log does not contain a valid summary dictionary.")
        else:
            st.info("No prediction summary data found for the selected filters. Chart cannot be generated.")

    with tab3:
        st.markdown("### Model Performance & Data Drift Over Time")
        metrics_logs = filtered_df.dropna(subset=['metrics']).copy()
        if not metrics_logs.empty:
            metrics_df = pd.json_normalize(metrics_logs['metrics'])
            metrics_df.index = metrics_logs.index
            
            metrics_logs = metrics_logs.join(metrics_df)
            metrics_logs = metrics_logs.sort_values('timestamp')

            fig_perf = make_subplots(rows=2, cols=2, subplot_titles=("F1-Score", "Accuracy", "ROC AUC", "Data Drift Score"))
            
            if 'f1' in metrics_logs.columns:
                fig_perf.add_trace(go.Scatter(x=metrics_logs['timestamp'], y=metrics_logs['f1'], mode='lines+markers', name='F1-Score'), row=1, col=1)
            if 'accuracy' in metrics_logs.columns:
                fig_perf.add_trace(go.Scatter(x=metrics_logs['timestamp'], y=metrics_logs['accuracy'], mode='lines+markers', name='Accuracy'), row=1, col=2)
            if 'roc_auc' in metrics_logs.columns:
                fig_perf.add_trace(go.Scatter(x=metrics_logs['timestamp'], y=metrics_logs['roc_auc'], mode='lines+markers', name='ROC AUC'), row=2, col=1)
            if 'drift_score' in metrics_logs.columns:
                fig_perf.add_trace(go.Scatter(x=metrics_logs['timestamp'], y=metrics_logs['drift_score'], mode='lines+markers', name='Drift Score'), row=2, col=2)

            fig_perf.update_layout(height=600, showlegend=False, title_text="Performance Metrics")
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No performance metrics data found for the selected filters.")

    with tab4:
        st.markdown("### Data Preprocessing Insights")
        preprocessing_logs = filtered_df[filtered_df['message'].str.contains("Filled missing", na=False)]
        if not preprocessing_logs.empty:
            st.write("#### Missing Value Imputation Logs")
            st.dataframe(preprocessing_logs[['timestamp', 'message']], use_container_width=True)
        else:
            st.info("No data preprocessing logs found for the selected filters.")

    with tab5:
        st.markdown("### Raw Log Data")
        display_df = filtered_df[['timestamp', 'level', 'message']]
        st.dataframe(display_df, use_container_width=True, height=500)
        
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Log Data as CSV",
            data=csv,
            file_name='filtered_log_data.csv',
            mime='text/csv',
        )

else:
    if uploaded_file:
        st.warning("No log data matches the selected filters. Try adjusting the date range or log level filters in the sidebar.")
