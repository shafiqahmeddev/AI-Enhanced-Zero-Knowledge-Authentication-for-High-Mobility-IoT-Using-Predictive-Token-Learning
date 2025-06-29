"""
Task 4.4: Real-Time Analytics Dashboard
======================================

This module provides a comprehensive real-time analytics dashboard for the
ZKPAS MLOps pipeline. It visualizes federated learning progress, privacy metrics,
model performance, and system health in real-time.

Key Features:
- Real-time federated learning progress tracking
- Privacy budget monitoring and alerts
- Model performance visualization
- Client participation analytics
- System health monitoring
- Interactive dashboard with Streamlit
- Export capabilities for reports
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import sqlite3
from pathlib import Path
import pickle

# Import from our existing modules
from app.events import EventBus, Event, EventType
from app.federated_learning import FederatedLearningCoordinator
from app.mlflow_tracking import ZKPASMLflowTracker
from app.model_interpretability import ModelInterpretabilityManager

# Set up logging
logger = logging.getLogger(__name__)

# Dashboard configuration
DASHBOARD_CONFIG = {
    "refresh_interval": 5,  # seconds
    "max_data_points": 1000,
    "chart_height": 400,
    "privacy_alert_threshold": 0.8,  # 80% of budget
    "performance_alert_threshold": 0.05  # 5% drop
}


class DashboardDataManager:
    """
    Manages data collection and storage for the dashboard.
    
    Collects metrics from various ZKPAS components and stores them
    in a local database for efficient querying and visualization.
    """
    
    def __init__(self, db_path: str = "dashboard_data.db"):
        """
        Initialize dashboard data manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()
        
        # Data caches
        self.federated_metrics = []
        self.privacy_metrics = []
        self.client_metrics = []
        self.system_metrics = []
        
        # Event tracking
        self.last_update = datetime.now()
        self.event_counts = {}
        
        logger.info(f"Dashboard data manager initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS federated_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                round_number INTEGER,
                participants INTEGER,
                accuracy REAL,
                loss REAL,
                f1_score REAL,
                privacy_spent REAL,
                convergence_metric REAL,
                duration_seconds REAL
            );
            
            CREATE TABLE IF NOT EXISTS client_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                client_id TEXT,
                round_number INTEGER,
                sample_count INTEGER,
                local_accuracy REAL,
                local_loss REAL,
                gradient_norm REAL,
                privacy_spent REAL,
                update_size_mb REAL
            );
            
            CREATE TABLE IF NOT EXISTS privacy_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_budget REAL,
                budget_spent REAL,
                budget_remaining REAL,
                rounds_completed INTEGER,
                estimated_rounds_remaining INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                network_latency REAL,
                active_clients INTEGER,
                queue_size INTEGER,
                error_rate REAL
            );
            
            CREATE TABLE IF NOT EXISTS model_interpretability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                explanation_count INTEGER,
                avg_confidence REAL,
                feature_importance TEXT,
                explanation_type TEXT,
                privacy_preserving BOOLEAN
            );
        """)
        
        self.conn.commit()
        logger.info("Database tables initialized")
    
    def record_federated_round(self, round_data: Dict[str, Any]):
        """Record federated learning round metrics."""
        query = """
            INSERT INTO federated_rounds 
            (round_number, participants, accuracy, loss, f1_score, 
             privacy_spent, convergence_metric, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            round_data.get('round_number', 0),
            round_data.get('participants', 0),
            round_data.get('accuracy', 0.0),
            round_data.get('loss', 0.0),
            round_data.get('f1_score', 0.0),
            round_data.get('privacy_spent', 0.0),
            round_data.get('convergence_metric', 0.0),
            round_data.get('duration_seconds', 0.0)
        )
        
        self.conn.execute(query, params)
        self.conn.commit()
        
        # Update cache
        self.federated_metrics.append({
            'timestamp': datetime.now(),
            **round_data
        })
        
        # Limit cache size
        if len(self.federated_metrics) > DASHBOARD_CONFIG["max_data_points"]:
            self.federated_metrics = self.federated_metrics[-DASHBOARD_CONFIG["max_data_points"]:]
    
    def record_client_update(self, client_data: Dict[str, Any]):
        """Record client update metrics."""
        query = """
            INSERT INTO client_updates 
            (client_id, round_number, sample_count, local_accuracy, local_loss,
             gradient_norm, privacy_spent, update_size_mb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            client_data.get('client_id', ''),
            client_data.get('round_number', 0),
            client_data.get('sample_count', 0),
            client_data.get('local_accuracy', 0.0),
            client_data.get('local_loss', 0.0),
            client_data.get('gradient_norm', 0.0),
            client_data.get('privacy_spent', 0.0),
            client_data.get('update_size_mb', 0.0)
        )
        
        self.conn.execute(query, params)
        self.conn.commit()
    
    def record_privacy_metrics(self, privacy_data: Dict[str, Any]):
        """Record privacy tracking metrics."""
        query = """
            INSERT INTO privacy_tracking 
            (total_budget, budget_spent, budget_remaining, rounds_completed, estimated_rounds_remaining)
            VALUES (?, ?, ?, ?, ?)
        """
        
        params = (
            privacy_data.get('total_budget', 0.0),
            privacy_data.get('budget_spent', 0.0),
            privacy_data.get('budget_remaining', 0.0),
            privacy_data.get('rounds_completed', 0),
            privacy_data.get('estimated_rounds_remaining', 0)
        )
        
        self.conn.execute(query, params)
        self.conn.commit()
    
    def record_system_health(self, health_data: Dict[str, Any]):
        """Record system health metrics."""
        query = """
            INSERT INTO system_health 
            (cpu_usage, memory_usage, network_latency, active_clients, queue_size, error_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        params = (
            health_data.get('cpu_usage', 0.0),
            health_data.get('memory_usage', 0.0),
            health_data.get('network_latency', 0.0),
            health_data.get('active_clients', 0),
            health_data.get('queue_size', 0),
            health_data.get('error_rate', 0.0)
        )
        
        self.conn.execute(query, params)
        self.conn.commit()
    
    def get_recent_federated_metrics(self, limit: int = 100) -> pd.DataFrame:
        """Get recent federated learning metrics."""
        query = """
            SELECT * FROM federated_rounds 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def get_recent_privacy_metrics(self, limit: int = 100) -> pd.DataFrame:
        """Get recent privacy metrics."""
        query = """
            SELECT * FROM privacy_tracking 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    
    def get_client_participation(self, hours: int = 24) -> pd.DataFrame:
        """Get client participation metrics for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT client_id, COUNT(*) as update_count, 
                   AVG(local_accuracy) as avg_accuracy,
                   AVG(privacy_spent) as avg_privacy_spent,
                   MAX(timestamp) as last_seen
            FROM client_updates 
            WHERE timestamp > ?
            GROUP BY client_id
            ORDER BY update_count DESC
        """
        
        df = pd.read_sql_query(query, self.conn, params=(cutoff,))
        if not df.empty:
            df['last_seen'] = pd.to_datetime(df['last_seen'])
        return df
    
    def get_system_health_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get system health summary for the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        query = """
            SELECT AVG(cpu_usage) as avg_cpu,
                   AVG(memory_usage) as avg_memory,
                   AVG(network_latency) as avg_latency,
                   MAX(active_clients) as max_clients,
                   AVG(error_rate) as avg_error_rate
            FROM system_health 
            WHERE timestamp > ?
        """
        
        result = self.conn.execute(query, (cutoff,)).fetchone()
        
        if result:
            return {
                'avg_cpu_usage': result[0] or 0.0,
                'avg_memory_usage': result[1] or 0.0,
                'avg_network_latency': result[2] or 0.0,
                'max_active_clients': result[3] or 0,
                'avg_error_rate': result[4] or 0.0
            }
        else:
            return {
                'avg_cpu_usage': 0.0,
                'avg_memory_usage': 0.0,
                'avg_network_latency': 0.0,
                'max_active_clients': 0,
                'avg_error_rate': 0.0
            }


class DashboardApp:
    """
    Streamlit-based real-time analytics dashboard for ZKPAS.
    
    Provides interactive visualization of federated learning progress,
    privacy metrics, and system health monitoring.
    """
    
    def __init__(self, data_manager: DashboardDataManager):
        """
        Initialize dashboard app.
        
        Args:
            data_manager: Data manager for metric collection
        """
        self.data_manager = data_manager
        
        # Set page config
        st.set_page_config(
            page_title="ZKPAS Analytics Dashboard",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        logger.info("Dashboard app initialized")
    
    def run(self):
        """Run the Streamlit dashboard."""
        # Title and header
        st.title("🔒 ZKPAS Analytics Dashboard")
        st.markdown("Real-time monitoring for Zero-Knowledge Federated Learning")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Auto-refresh logic
        if st.session_state.get('auto_refresh', True):
            time.sleep(DASHBOARD_CONFIG["refresh_interval"])
            st.rerun()
        
        # Main dashboard layout
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔄 Federated Learning", 
            "🔐 Privacy Tracking", 
            "👥 Client Analytics", 
            "🖥️ System Health"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_federated_learning_tab()
        
        with tab3:
            self._render_privacy_tracking_tab()
        
        with tab4:
            self._render_client_analytics_tab()
        
        with tab5:
            self._render_system_health_tab()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        st.session_state.auto_refresh = auto_refresh
        
        # Refresh interval
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)", 
            min_value=1, 
            max_value=60, 
            value=DASHBOARD_CONFIG["refresh_interval"]
        )
        DASHBOARD_CONFIG["refresh_interval"] = refresh_interval
        
        # Data range controls
        st.sidebar.subheader("📅 Data Range")
        hours_back = st.sidebar.selectbox(
            "Show data from last:", 
            [1, 6, 12, 24, 48, 72],
            index=3
        )
        st.session_state.hours_back = hours_back
        
        # Export controls
        st.sidebar.subheader("📤 Export Data")
        if st.sidebar.button("Export CSV"):
            self._export_data()
        
        # System status
        st.sidebar.subheader("🚦 System Status")
        health_summary = self.data_manager.get_system_health_summary()
        
        # Status indicators
        cpu_status = "🟢" if health_summary['avg_cpu_usage'] < 80 else "🟡" if health_summary['avg_cpu_usage'] < 95 else "🔴"
        memory_status = "🟢" if health_summary['avg_memory_usage'] < 80 else "🟡" if health_summary['avg_memory_usage'] < 95 else "🔴"
        
        st.sidebar.write(f"{cpu_status} CPU: {health_summary['avg_cpu_usage']:.1f}%")
        st.sidebar.write(f"{memory_status} Memory: {health_summary['avg_memory_usage']:.1f}%")
        st.sidebar.write(f"🌐 Active Clients: {health_summary['max_active_clients']}")
        
        # Last update time
        st.sidebar.write(f"🕒 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def _render_overview_tab(self):
        """Render overview dashboard tab."""
        st.header("📊 System Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Get recent data
        fed_metrics = self.data_manager.get_recent_federated_metrics(10)
        privacy_metrics = self.data_manager.get_recent_privacy_metrics(10)
        client_participation = self.data_manager.get_client_participation(24)
        
        with col1:
            if not fed_metrics.empty:
                latest_accuracy = fed_metrics.iloc[-1]['accuracy']
                st.metric(
                    "Latest Accuracy", 
                    f"{latest_accuracy:.3f}",
                    delta=f"{latest_accuracy - fed_metrics.iloc[-2]['accuracy']:.3f}" if len(fed_metrics) > 1 else None
                )
            else:
                st.metric("Latest Accuracy", "N/A")
        
        with col2:
            if not fed_metrics.empty:
                total_rounds = fed_metrics.iloc[-1]['round_number']
                st.metric("Completed Rounds", f"{total_rounds}")
            else:
                st.metric("Completed Rounds", "0")
        
        with col3:
            if not privacy_metrics.empty:
                budget_remaining = privacy_metrics.iloc[-1]['budget_remaining']
                total_budget = privacy_metrics.iloc[-1]['total_budget']
                budget_pct = (budget_remaining / total_budget) * 100 if total_budget > 0 else 0
                st.metric(
                    "Privacy Budget", 
                    f"{budget_pct:.1f}%",
                    delta=f"-{privacy_metrics.iloc[-1]['budget_spent']:.2f}" if not privacy_metrics.empty else None
                )
            else:
                st.metric("Privacy Budget", "N/A")
        
        with col4:
            active_clients = len(client_participation)
            st.metric("Active Clients", f"{active_clients}")
        
        # Recent activity chart
        if not fed_metrics.empty:
            st.subheader("📈 Recent Training Progress")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Model Accuracy', 'Training Loss'),
                vertical_spacing=0.1
            )
            
            # Accuracy plot
            fig.add_trace(
                go.Scatter(
                    x=fed_metrics['round_number'],
                    y=fed_metrics['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            # Loss plot
            fig.add_trace(
                go.Scatter(
                    x=fed_metrics['round_number'],
                    y=fed_metrics['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Round Number", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_federated_learning_tab(self):
        """Render federated learning monitoring tab."""
        st.header("🔄 Federated Learning Progress")
        
        fed_metrics = self.data_manager.get_recent_federated_metrics(100)
        
        if fed_metrics.empty:
            st.warning("No federated learning data available yet.")
            return
        
        # Performance metrics over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Performance")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fed_metrics['round_number'],
                y=fed_metrics['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=fed_metrics['round_number'],
                y=fed_metrics['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Model Performance Over Rounds",
                xaxis_title="Round Number",
                yaxis_title="Score",
                height=DASHBOARD_CONFIG["chart_height"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Training Loss")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fed_metrics['round_number'],
                y=fed_metrics['loss'],
                mode='lines+markers',
                name='Loss',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Training Loss Over Rounds",
                xaxis_title="Round Number",
                yaxis_title="Loss",
                height=DASHBOARD_CONFIG["chart_height"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Convergence and participation metrics
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("⚡ Convergence Metrics")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fed_metrics['round_number'],
                y=fed_metrics['convergence_metric'],
                mode='lines+markers',
                name='Convergence',
                line=dict(color='purple')
            ))
            
            fig.update_layout(
                title="Model Convergence",
                xaxis_title="Round Number",
                yaxis_title="Convergence Metric",
                height=DASHBOARD_CONFIG["chart_height"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("👥 Client Participation")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=fed_metrics['round_number'],
                y=fed_metrics['participants'],
                name='Participants',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Clients per Round",
                xaxis_title="Round Number",
                yaxis_title="Number of Clients",
                height=DASHBOARD_CONFIG["chart_height"]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Round Metrics")
        
        # Show last 10 rounds
        recent_data = fed_metrics.tail(10)[['round_number', 'participants', 'accuracy', 'loss', 'f1_score', 'privacy_spent', 'duration_seconds']]
        recent_data.columns = ['Round', 'Participants', 'Accuracy', 'Loss', 'F1 Score', 'Privacy Spent', 'Duration (s)']
        
        st.dataframe(recent_data, use_container_width=True)
    
    def _render_privacy_tracking_tab(self):
        """Render privacy tracking tab."""
        st.header("🔐 Privacy Budget Tracking")
        
        privacy_metrics = self.data_manager.get_recent_privacy_metrics(100)
        
        if privacy_metrics.empty:
            st.warning("No privacy tracking data available yet.")
            return
        
        # Current privacy status
        latest_privacy = privacy_metrics.iloc[-1]
        budget_used_pct = (latest_privacy['budget_spent'] / latest_privacy['total_budget']) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Budget Used", 
                f"{budget_used_pct:.1f}%",
                delta=f"-{latest_privacy['budget_spent']:.2f}"
            )
        
        with col2:
            st.metric(
                "Budget Remaining", 
                f"{latest_privacy['budget_remaining']:.2f}",
                delta=f"{latest_privacy['budget_remaining'] - latest_privacy['budget_spent']:.2f}"
            )
        
        with col3:
            st.metric(
                "Estimated Rounds Left", 
                f"{latest_privacy['estimated_rounds_remaining']}"
            )
        
        # Privacy budget visualization
        st.subheader("📊 Privacy Budget Over Time")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Budget Consumption', 'Budget Remaining'),
            vertical_spacing=0.1
        )
        
        # Budget spent over time
        fig.add_trace(
            go.Scatter(
                x=privacy_metrics['timestamp'],
                y=privacy_metrics['budget_spent'],
                mode='lines+markers',
                name='Budget Spent',
                line=dict(color='red'),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Budget remaining over time
        fig.add_trace(
            go.Scatter(
                x=privacy_metrics['timestamp'],
                y=privacy_metrics['budget_remaining'],
                mode='lines+markers',
                name='Budget Remaining',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Budget Spent", row=1, col=1)
        fig.update_yaxes(title_text="Budget Remaining", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Privacy alerts
        if budget_used_pct > DASHBOARD_CONFIG["privacy_alert_threshold"] * 100:
            st.error(f"⚠️ Privacy Budget Alert: {budget_used_pct:.1f}% of budget used!")
        
        # Privacy efficiency metrics
        st.subheader("⚡ Privacy Efficiency")
        
        if not privacy_metrics.empty:
            efficiency = latest_privacy['rounds_completed'] / latest_privacy['budget_spent'] if latest_privacy['budget_spent'] > 0 else 0
            st.metric("Rounds per Privacy Unit", f"{efficiency:.2f}")
    
    def _render_client_analytics_tab(self):
        """Render client analytics tab."""
        st.header("👥 Client Analytics")
        
        client_data = self.data_manager.get_client_participation(24)
        
        if client_data.empty:
            st.warning("No client data available yet.")
            return
        
        # Client summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Active Clients", len(client_data))
        
        with col2:
            avg_updates = client_data['update_count'].mean()
            st.metric("Avg Updates per Client", f"{avg_updates:.1f}")
        
        with col3:
            avg_accuracy = client_data['avg_accuracy'].mean()
            st.metric("Avg Client Accuracy", f"{avg_accuracy:.3f}")
        
        with col4:
            total_privacy = client_data['avg_privacy_spent'].sum()
            st.metric("Total Privacy Spent", f"{total_privacy:.2f}")
        
        # Client participation visualization
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("📊 Client Update Frequency")
            
            fig = px.bar(
                client_data.head(10), 
                x='client_id', 
                y='update_count',
                title="Updates per Client (Top 10)"
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=DASHBOARD_CONFIG["chart_height"])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            st.subheader("🎯 Client Accuracy Distribution")
            
            fig = px.histogram(
                client_data, 
                x='avg_accuracy',
                nbins=20,
                title="Distribution of Client Accuracies"
            )
            fig.update_layout(height=DASHBOARD_CONFIG["chart_height"])
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Client details table
        st.subheader("📋 Client Details")
        
        # Format the data for display
        display_data = client_data.copy()
        display_data['avg_accuracy'] = display_data['avg_accuracy'].round(3)
        display_data['avg_privacy_spent'] = display_data['avg_privacy_spent'].round(3)
        display_data.columns = ['Client ID', 'Updates', 'Avg Accuracy', 'Avg Privacy Spent', 'Last Seen']
        
        st.dataframe(display_data, use_container_width=True)
    
    def _render_system_health_tab(self):
        """Render system health monitoring tab."""
        st.header("🖥️ System Health")
        
        health_summary = self.data_manager.get_system_health_summary(60)
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = health_summary['avg_cpu_usage']
            cpu_color = "normal" if cpu_usage < 80 else "warning" if cpu_usage < 95 else "error"
            st.metric("Avg CPU Usage", f"{cpu_usage:.1f}%")
        
        with col2:
            memory_usage = health_summary['avg_memory_usage']
            memory_color = "normal" if memory_usage < 80 else "warning" if memory_usage < 95 else "error"
            st.metric("Avg Memory Usage", f"{memory_usage:.1f}%")
        
        with col3:
            latency = health_summary['avg_network_latency']
            st.metric("Avg Network Latency", f"{latency:.2f}ms")
        
        with col4:
            st.metric("Max Active Clients", f"{health_summary['max_active_clients']}")
        
        # System status indicators
        st.subheader("🚦 System Status")
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            if cpu_usage < 80:
                st.success("🟢 CPU Status: Normal")
            elif cpu_usage < 95:
                st.warning("🟡 CPU Status: High")
            else:
                st.error("🔴 CPU Status: Critical")
        
        with col6:
            if memory_usage < 80:
                st.success("🟢 Memory Status: Normal")
            elif memory_usage < 95:
                st.warning("🟡 Memory Status: High")
            else:
                st.error("🔴 Memory Status: Critical")
        
        with col7:
            error_rate = health_summary['avg_error_rate']
            if error_rate < 0.01:
                st.success("🟢 Error Rate: Low")
            elif error_rate < 0.05:
                st.warning("🟡 Error Rate: Medium")
            else:
                st.error("🔴 Error Rate: High")
        
        # Mock real-time data for demonstration
        st.subheader("📈 Real-Time Metrics")
        
        # Generate mock time series data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=30),
            end=datetime.now(),
            freq='1min'
        )
        
        mock_data = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': np.random.normal(cpu_usage, 10, len(timestamps)),
            'memory_usage': np.random.normal(memory_usage, 5, len(timestamps)),
            'network_latency': np.random.exponential(latency, len(timestamps))
        })
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Network Latency (ms)'),
            vertical_spacing=0.08
        )
        
        # CPU usage
        fig.add_trace(
            go.Scatter(
                x=mock_data['timestamp'],
                y=mock_data['cpu_usage'],
                mode='lines',
                name='CPU',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Memory usage
        fig.add_trace(
            go.Scatter(
                x=mock_data['timestamp'],
                y=mock_data['memory_usage'],
                mode='lines',
                name='Memory',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Network latency
        fig.add_trace(
            go.Scatter(
                x=mock_data['timestamp'],
                y=mock_data['network_latency'],
                mode='lines',
                name='Latency',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _export_data(self):
        """Export dashboard data to CSV files."""
        try:
            # Get all data
            fed_metrics = self.data_manager.get_recent_federated_metrics(1000)
            privacy_metrics = self.data_manager.get_recent_privacy_metrics(1000)
            client_data = self.data_manager.get_client_participation(168)  # 1 week
            
            # Create export directory
            export_dir = Path("dashboard_exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export files
            if not fed_metrics.empty:
                fed_metrics.to_csv(export_dir / f"federated_metrics_{timestamp}.csv", index=False)
            
            if not privacy_metrics.empty:
                privacy_metrics.to_csv(export_dir / f"privacy_metrics_{timestamp}.csv", index=False)
            
            if not client_data.empty:
                client_data.to_csv(export_dir / f"client_analytics_{timestamp}.csv", index=False)
            
            st.success(f"Data exported to {export_dir}")
            
        except Exception as e:
            st.error(f"Export failed: {e}")


def generate_mock_data(data_manager: DashboardDataManager, num_rounds: int = 10):
    """Generate mock data for dashboard demonstration."""
    
    # Simulate federated learning rounds
    for round_num in range(1, num_rounds + 1):
        # Mock federated round data
        fed_data = {
            'round_number': round_num,
            'participants': np.random.randint(3, 8),
            'accuracy': 0.7 + round_num * 0.02 + np.random.normal(0, 0.01),
            'loss': 0.6 - round_num * 0.03 + np.random.normal(0, 0.02),
            'f1_score': 0.65 + round_num * 0.025 + np.random.normal(0, 0.015),
            'privacy_spent': np.random.uniform(0.3, 0.7),
            'convergence_metric': np.random.uniform(0.8, 1.0),
            'duration_seconds': np.random.uniform(45, 120)
        }
        
        data_manager.record_federated_round(fed_data)
        
        # Mock privacy tracking
        total_budget = 10.0
        budget_spent = round_num * 0.5 + np.random.uniform(0, 0.2)
        privacy_data = {
            'total_budget': total_budget,
            'budget_spent': budget_spent,
            'budget_remaining': total_budget - budget_spent,
            'rounds_completed': round_num,
            'estimated_rounds_remaining': max(0, int((total_budget - budget_spent) / 0.5))
        }
        
        data_manager.record_privacy_metrics(privacy_data)
        
        # Mock client updates
        for client_id in range(3, 8):
            client_data = {
                'client_id': f'client_{client_id}',
                'round_number': round_num,
                'sample_count': np.random.randint(100, 500),
                'local_accuracy': fed_data['accuracy'] + np.random.normal(0, 0.05),
                'local_loss': fed_data['loss'] + np.random.normal(0, 0.03),
                'gradient_norm': np.random.uniform(0.5, 2.0),
                'privacy_spent': np.random.uniform(0.1, 0.3),
                'update_size_mb': np.random.uniform(0.5, 2.5)
            }
            
            data_manager.record_client_update(client_data)
        
        # Mock system health
        health_data = {
            'cpu_usage': np.random.uniform(40, 80),
            'memory_usage': np.random.uniform(50, 75),
            'network_latency': np.random.exponential(15),
            'active_clients': np.random.randint(3, 8),
            'queue_size': np.random.randint(0, 5),
            'error_rate': np.random.exponential(0.01)
        }
        
        data_manager.record_system_health(health_data)


def main():
    """Run the ZKPAS Analytics Dashboard."""
    
    # Initialize data manager
    data_manager = DashboardDataManager()
    
    # Generate some mock data for demonstration
    generate_mock_data(data_manager, 15)
    
    # Initialize and run dashboard app
    dashboard = DashboardApp(data_manager)
    dashboard.run()


if __name__ == "__main__":
    main()
