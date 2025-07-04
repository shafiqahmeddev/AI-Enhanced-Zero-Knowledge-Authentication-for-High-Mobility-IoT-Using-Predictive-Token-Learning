"""
Interactive Research Dashboard for ZKPAS System
Phase 6: GUI & Interactive Research Dashboard

This module implements a powerful, interactive dashboard for deep system analysis
with real-time parameter tuning and sensitivity analysis capabilities.
"""

import asyncio
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import queue

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.events import EventBus, EventType, Event
from app.components.trusted_authority import TrustedAuthority
from app.components.gateway_node import GatewayNode  
from app.components.iot_device import IoTDevice
from app.mobility_predictor import MobilityPredictor
from app.model_trainer import ModelTrainer
from app.dataset_loader import DatasetLoader, get_default_dataset_config
from shared.config import CryptoConfig


@dataclass
class SimulationParameters:
    """Live simulation parameters for sensitivity analysis."""
    network_latency_ms: float = 50.0
    packet_drop_rate_percent: float = 1.0
    ai_prediction_error_meters: float = 100.0
    device_density: int = 10
    authentication_window_seconds: int = 300
    mobility_update_interval_seconds: float = 30.0
    byzantine_fault_ratio: float = 0.1
    privacy_budget_epsilon: float = 1.0


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    authentication_success_rate: float = 0.0
    average_latency_ms: float = 0.0
    throughput_auths_per_second: float = 0.0
    prediction_accuracy_meters: float = 0.0
    network_utilization_percent: float = 0.0
    security_violations: int = 0
    byzantine_detections: int = 0
    privacy_budget_remaining: float = 1.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ScenarioBuilder:
    """Builder for creating complex simulation scenarios."""
    
    def __init__(self):
        self.scenario_commands = []
        self.devices = []
        self.gateways = []
        self.chaos_events = []
    
    def add_device(self, device_id: str, location: tuple = (0.0, 0.0), mobility_pattern: str = "RANDOM"):
        """Add IoT device to scenario."""
        command = {
            "type": "CREATE_DEVICE",
            "device_id": device_id,
            "location": location,
            "mobility_pattern": mobility_pattern,
            "timestamp": 0
        }
        self.scenario_commands.append(command)
        self.devices.append(device_id)
        return self
    
    def add_gateway(self, gateway_id: str, coverage_area: float = 1000.0):
        """Add gateway node to scenario."""
        command = {
            "type": "CREATE_GATEWAY", 
            "gateway_id": gateway_id,
            "coverage_area": coverage_area,
            "timestamp": 0
        }
        self.scenario_commands.append(command)
        self.gateways.append(gateway_id)
        return self
    
    def add_authentication(self, device_id: str, gateway_id: str, timestamp: float):
        """Add authentication event to scenario."""
        command = {
            "type": "AUTHENTICATE",
            "device_id": device_id,
            "gateway_id": gateway_id,
            "timestamp": timestamp
        }
        self.scenario_commands.append(command)
        return self
    
    def add_mobility_event(self, device_id: str, new_location: tuple, timestamp: float):
        """Add device mobility event."""
        command = {
            "type": "MOVE_DEVICE",
            "device_id": device_id,
            "new_location": new_location,
            "timestamp": timestamp
        }
        self.scenario_commands.append(command)
        return self
    
    def add_chaos_event(self, chaos_type: str, target: str, timestamp: float, **kwargs):
        """Add chaos engineering event."""
        command = {
            "type": "CHAOS_EVENT",
            "chaos_type": chaos_type,  # DROP_EVENT, CORRUPT_STATE, INJECT_LATENCY
            "target": target,
            "timestamp": timestamp,
            **kwargs
        }
        self.scenario_commands.append(command)
        self.chaos_events.append(command)
        return self
    
    def build(self) -> List[Dict]:
        """Build and return sorted scenario commands."""
        return sorted(self.scenario_commands, key=lambda x: x["timestamp"])


class VisualTimeline:
    """Visual timeline component for scenario visualization."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, parent_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.timeline_data = []
        self.current_time = 0
        self.metrics_history = []
        
        # Create subplots
        self.ax_events = self.figure.add_subplot(3, 1, 1)
        self.ax_metrics = self.figure.add_subplot(3, 1, 2)
        self.ax_network = self.figure.add_subplot(3, 1, 3)
        
        self.figure.tight_layout()
    
    def add_event(self, timestamp: float, event_type: str, entity: str, details: str = ""):
        """Add event to timeline visualization."""
        self.timeline_data.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "entity": entity,
            "details": details
        })
        self._update_visualization()
    
    def add_metrics(self, metrics: SystemHealthMetrics):
        """Add system metrics to timeline."""
        self.metrics_history.append(metrics)
        self._update_metrics_plot()
    
    def _update_visualization(self):
        """Update the timeline visualization."""
        if not self.timeline_data:
            return
        
        self.ax_events.clear()
        
        # Group events by type
        event_types = {}
        for event in self.timeline_data:
            event_type = event["event_type"]
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event["timestamp"])
        
        # Plot events
        y_pos = 0
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (event_type, timestamps) in enumerate(event_types.items()):
            color = colors[i % len(colors)]
            self.ax_events.scatter(timestamps, [y_pos] * len(timestamps), 
                                 c=color, label=event_type, alpha=0.7)
            y_pos += 1
        
        self.ax_events.set_ylabel("Event Types")
        self.ax_events.set_title("System Event Timeline")
        self.ax_events.legend()
        self.ax_events.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _update_metrics_plot(self):
        """Update metrics visualization."""
        # Implementation for metrics plotting
        pass
    
    def clear(self):
        """Clear timeline data."""
        self.timeline_data = []
        self.ax_events.clear()
        self.ax_metrics.clear()
        self.ax_network.clear()
        self.canvas.draw()


class LiveParametersPanel:
    """Interactive panel for live parameter tuning."""
    
    def __init__(self, parent_frame, on_parameter_change: Callable):
        self.parent_frame = parent_frame
        self.on_parameter_change = on_parameter_change
        self.parameters = SimulationParameters()
        
        self._create_parameter_controls()
    
    def _create_parameter_controls(self):
        """Create parameter control widgets."""
        # Network Latency
        latency_frame = ttk.Frame(self.parent_frame)
        latency_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(latency_frame, text="Network Latency (ms):").pack(side="left")
        self.latency_var = tk.DoubleVar(value=self.parameters.network_latency_ms)
        latency_scale = ttk.Scale(latency_frame, from_=10, to=500, 
                                orient="horizontal", variable=self.latency_var,
                                command=self._on_latency_change)
        latency_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.latency_label = ttk.Label(latency_frame, text=f"{self.parameters.network_latency_ms:.1f}")
        self.latency_label.pack(side="right")
        
        # Packet Drop Rate
        drop_frame = ttk.Frame(self.parent_frame)
        drop_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(drop_frame, text="Packet Drop Rate (%):").pack(side="left")
        self.drop_var = tk.DoubleVar(value=self.parameters.packet_drop_rate_percent)
        drop_scale = ttk.Scale(drop_frame, from_=0, to=20, 
                             orient="horizontal", variable=self.drop_var,
                             command=self._on_drop_rate_change)
        drop_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.drop_label = ttk.Label(drop_frame, text=f"{self.parameters.packet_drop_rate_percent:.1f}")
        self.drop_label.pack(side="right")
        
        # AI Prediction Error
        ai_frame = ttk.Frame(self.parent_frame)
        ai_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(ai_frame, text="AI Prediction Error (m):").pack(side="left")
        self.ai_var = tk.DoubleVar(value=self.parameters.ai_prediction_error_meters)
        ai_scale = ttk.Scale(ai_frame, from_=50, to=1000, 
                           orient="horizontal", variable=self.ai_var,
                           command=self._on_ai_error_change)
        ai_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.ai_label = ttk.Label(ai_frame, text=f"{self.parameters.ai_prediction_error_meters:.0f}")
        self.ai_label.pack(side="right")
        
        # Device Density
        density_frame = ttk.Frame(self.parent_frame)
        density_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(density_frame, text="Device Density:").pack(side="left")
        self.density_var = tk.IntVar(value=self.parameters.device_density)
        density_scale = ttk.Scale(density_frame, from_=5, to=100, 
                                orient="horizontal", variable=self.density_var,
                                command=self._on_density_change)
        density_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.density_label = ttk.Label(density_frame, text=f"{self.parameters.device_density}")
        self.density_label.pack(side="right")
        
        # Byzantine Fault Ratio
        byzantine_frame = ttk.Frame(self.parent_frame)
        byzantine_frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(byzantine_frame, text="Byzantine Fault Ratio:").pack(side="left")
        self.byzantine_var = tk.DoubleVar(value=self.parameters.byzantine_fault_ratio)
        byzantine_scale = ttk.Scale(byzantine_frame, from_=0, to=0.5, 
                                  orient="horizontal", variable=self.byzantine_var,
                                  command=self._on_byzantine_change)
        byzantine_scale.pack(side="left", fill="x", expand=True, padx=5)
        self.byzantine_label = ttk.Label(byzantine_frame, text=f"{self.parameters.byzantine_fault_ratio:.2f}")
        self.byzantine_label.pack(side="right")
    
    def _on_latency_change(self, value):
        """Handle network latency change."""
        self.parameters.network_latency_ms = float(value)
        self.latency_label.config(text=f"{self.parameters.network_latency_ms:.1f}")
        self.on_parameter_change("network_latency_ms", self.parameters.network_latency_ms)
    
    def _on_drop_rate_change(self, value):
        """Handle packet drop rate change."""
        self.parameters.packet_drop_rate_percent = float(value)
        self.drop_label.config(text=f"{self.parameters.packet_drop_rate_percent:.1f}")
        self.on_parameter_change("packet_drop_rate_percent", self.parameters.packet_drop_rate_percent)
    
    def _on_ai_error_change(self, value):
        """Handle AI prediction error change."""
        self.parameters.ai_prediction_error_meters = float(value)
        self.ai_label.config(text=f"{self.parameters.ai_prediction_error_meters:.0f}")
        self.on_parameter_change("ai_prediction_error_meters", self.parameters.ai_prediction_error_meters)
    
    def _on_density_change(self, value):
        """Handle device density change."""
        self.parameters.device_density = int(float(value))
        self.density_label.config(text=f"{self.parameters.device_density}")
        self.on_parameter_change("device_density", self.parameters.device_density)
    
    def _on_byzantine_change(self, value):
        """Handle Byzantine fault ratio change."""
        self.parameters.byzantine_fault_ratio = float(value)
        self.byzantine_label.config(text=f"{self.parameters.byzantine_fault_ratio:.2f}")
        self.on_parameter_change("byzantine_fault_ratio", self.parameters.byzantine_fault_ratio)


class SystemHealthPanel:
    """Real-time system health monitoring panel."""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.metrics = SystemHealthMetrics()
        self.metrics_history = []
        
        self._create_health_widgets()
        
        # Start health monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_health, daemon=True)
        self.monitor_thread.start()
    
    def _create_health_widgets(self):
        """Create health monitoring widgets."""
        # Create metrics display
        metrics_frame = ttk.LabelFrame(self.parent_frame, text="Real-Time Metrics")
        metrics_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Success Rate
        success_frame = ttk.Frame(metrics_frame)
        success_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(success_frame, text="Auth Success Rate:").pack(side="left")
        self.success_label = ttk.Label(success_frame, text="0.0%", foreground="green")
        self.success_label.pack(side="right")
        
        # Average Latency
        latency_frame = ttk.Frame(metrics_frame)
        latency_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(latency_frame, text="Avg Latency:").pack(side="left")
        self.latency_label = ttk.Label(latency_frame, text="0.0 ms")
        self.latency_label.pack(side="right")
        
        # Throughput
        throughput_frame = ttk.Frame(metrics_frame)
        throughput_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(throughput_frame, text="Throughput:").pack(side="left")
        self.throughput_label = ttk.Label(throughput_frame, text="0.0 auth/s")
        self.throughput_label.pack(side="right")
        
        # Prediction Accuracy
        prediction_frame = ttk.Frame(metrics_frame)
        prediction_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(prediction_frame, text="Prediction Accuracy:").pack(side="left")
        self.prediction_label = ttk.Label(prediction_frame, text="0.0 m")
        self.prediction_label.pack(side="right")
        
        # Security Violations
        security_frame = ttk.Frame(metrics_frame)
        security_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(security_frame, text="Security Violations:").pack(side="left")
        self.security_label = ttk.Label(security_frame, text="0", foreground="red")
        self.security_label.pack(side="right")
        
        # Byzantine Detections
        byzantine_frame = ttk.Frame(metrics_frame)
        byzantine_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(byzantine_frame, text="Byzantine Detections:").pack(side="left")
        self.byzantine_label = ttk.Label(byzantine_frame, text="0", foreground="orange")
        self.byzantine_label.pack(side="right")
        
        # Privacy Budget
        privacy_frame = ttk.Frame(metrics_frame)
        privacy_frame.pack(fill="x", padx=5, pady=2)
        ttk.Label(privacy_frame, text="Privacy Budget:").pack(side="left")
        self.privacy_label = ttk.Label(privacy_frame, text="100%", foreground="blue")
        self.privacy_label.pack(side="right")
    
    def update_metrics(self, metrics: SystemHealthMetrics):
        """Update displayed metrics."""
        self.metrics = metrics
        self.metrics_history.append(metrics)
        
        # Update labels
        self.success_label.config(text=f"{metrics.authentication_success_rate:.1f}%")
        self.latency_label.config(text=f"{metrics.average_latency_ms:.1f} ms")
        self.throughput_label.config(text=f"{metrics.throughput_auths_per_second:.1f} auth/s")
        self.prediction_label.config(text=f"{metrics.prediction_accuracy_meters:.1f} m")
        self.security_label.config(text=f"{metrics.security_violations}")
        self.byzantine_label.config(text=f"{metrics.byzantine_detections}")
        self.privacy_label.config(text=f"{metrics.privacy_budget_remaining*100:.1f}%")
        
        # Update colors based on thresholds
        if metrics.authentication_success_rate < 95:
            self.success_label.config(foreground="red")
        else:
            self.success_label.config(foreground="green")
        
        if metrics.security_violations > 0:
            self.security_label.config(foreground="red")
        else:
            self.security_label.config(foreground="green")
    
    def _monitor_health(self):
        """Background health monitoring thread."""
        while self.monitoring:
            # Simulate health metrics collection
            # In real implementation, this would collect from event bus
            time.sleep(1)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False


class ZKPASResearchDashboard:
    """Main ZKPAS Interactive Research Dashboard."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ZKPAS Interactive Research Dashboard v2.0")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.event_bus = EventBus()
        self.scenario_builder = ScenarioBuilder()
        self.parameters = SimulationParameters()
        self.health_metrics = SystemHealthMetrics()
        
        # Simulation state
        self.simulation_running = False
        self.simulation_thread = None
        self.simulation_queue = queue.Queue()
        
        self._create_main_layout()
        self._setup_event_handlers()
        
    def _create_main_layout(self):
        """Create the main window layout."""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Visualization
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        # Create control panels
        self._create_control_panels(left_frame)
        
        # Create visualization panels
        self._create_visualization_panels(right_frame)
    
    def _create_control_panels(self, parent):
        """Create control panels in left frame."""
        # Scenario Builder Panel
        scenario_frame = ttk.LabelFrame(parent, text="Scenario Builder")
        scenario_frame.pack(fill="x", padx=5, pady=5)
        
        # Scenario controls
        scenario_controls = ttk.Frame(scenario_frame)
        scenario_controls.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(scenario_controls, text="Load Scenario", 
                  command=self._load_scenario).pack(side="left", padx=2)
        ttk.Button(scenario_controls, text="Save Scenario", 
                  command=self._save_scenario).pack(side="left", padx=2)
        ttk.Button(scenario_controls, text="Clear", 
                  command=self._clear_scenario).pack(side="left", padx=2)
        
        # Quick scenario buttons
        quick_frame = ttk.Frame(scenario_frame)
        quick_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(quick_frame, text="Basic Auth Test", 
                  command=lambda: self._create_basic_scenario()).pack(fill="x", pady=1)
        ttk.Button(quick_frame, text="High Mobility Test", 
                  command=lambda: self._create_mobility_scenario()).pack(fill="x", pady=1)
        ttk.Button(quick_frame, text="Byzantine Attack Test", 
                  command=lambda: self._create_byzantine_scenario()).pack(fill="x", pady=1)
        ttk.Button(quick_frame, text="Chaos Engineering", 
                  command=lambda: self._create_chaos_scenario()).pack(fill="x", pady=1)
        
        # Live Parameters Panel
        params_frame = ttk.LabelFrame(parent, text="Live Simulation Parameters")
        params_frame.pack(fill="x", padx=5, pady=5)
        
        self.parameters_panel = LiveParametersPanel(params_frame, self._on_parameter_change)
        
        # System Health Panel
        health_frame = ttk.LabelFrame(parent, text="System Health")
        health_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.health_panel = SystemHealthPanel(health_frame)
        
        # Simulation Control
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Simulation", 
                                      command=self._start_simulation)
        self.start_button.pack(side="left", padx=2)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Simulation", 
                                     command=self._stop_simulation, state="disabled")
        self.stop_button.pack(side="left", padx=2)
        
        self.reset_button = ttk.Button(control_frame, text="Reset", 
                                      command=self._reset_simulation)
        self.reset_button.pack(side="left", padx=2)
    
    def _create_visualization_panels(self, parent):
        """Create visualization panels in right frame."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)
        
        # Visual Timeline Tab
        timeline_frame = ttk.Frame(notebook)
        notebook.add(timeline_frame, text="Event Timeline")
        self.visual_timeline = VisualTimeline(timeline_frame)
        
        # System Metrics Tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="System Metrics")
        self._create_metrics_visualization(metrics_frame)
        
        # Network Topology Tab
        topology_frame = ttk.Frame(notebook)
        notebook.add(topology_frame, text="Network Topology")
        self._create_topology_visualization(topology_frame)
        
        # ML Model Performance Tab
        ml_frame = ttk.Frame(notebook)
        notebook.add(ml_frame, text="ML Performance")
        self._create_ml_visualization(ml_frame)
    
    def _create_metrics_visualization(self, parent):
        """Create system metrics visualization."""
        self.metrics_figure = Figure(figsize=(10, 6), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_figure, parent)
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Create subplots for different metrics
        self.ax_auth = self.metrics_figure.add_subplot(2, 2, 1)
        self.ax_latency = self.metrics_figure.add_subplot(2, 2, 2)
        self.ax_throughput = self.metrics_figure.add_subplot(2, 2, 3)
        self.ax_security = self.metrics_figure.add_subplot(2, 2, 4)
        
        self.metrics_figure.tight_layout()
    
    def _create_topology_visualization(self, parent):
        """Create network topology visualization."""
        self.topology_figure = Figure(figsize=(10, 6), dpi=100)
        self.topology_canvas = FigureCanvasTkAgg(self.topology_figure, parent)
        self.topology_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.ax_topology = self.topology_figure.add_subplot(1, 1, 1)
    
    def _create_ml_visualization(self, parent):
        """Create ML model performance visualization."""
        self.ml_figure = Figure(figsize=(10, 6), dpi=100)
        self.ml_canvas = FigureCanvasTkAgg(self.ml_figure, parent)
        self.ml_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.ax_prediction = self.ml_figure.add_subplot(2, 1, 1)
        self.ax_training = self.ml_figure.add_subplot(2, 1, 2)
        
        self.ml_figure.tight_layout()
    
    def _setup_event_handlers(self):
        """Setup event handlers for dashboard."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _on_parameter_change(self, parameter: str, value: Any):
        """Handle live parameter changes."""
        print(f"Parameter changed: {parameter} = {value}")
        
        # Update simulation in real-time
        if self.simulation_running:
            self.simulation_queue.put({
                "type": "UPDATE_PARAMETER",
                "parameter": parameter,
                "value": value
            })
    
    def _create_basic_scenario(self):
        """Create basic authentication test scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Add devices and gateways
        self.scenario_builder.add_gateway("GW001", coverage_area=1000)
        self.scenario_builder.add_device("DEV001", location=(0, 0))
        self.scenario_builder.add_device("DEV002", location=(500, 500))
        
        # Add authentication events
        self.scenario_builder.add_authentication("DEV001", "GW001", 1.0)
        self.scenario_builder.add_authentication("DEV002", "GW001", 2.0)
        
        messagebox.showinfo("Scenario", "Basic Authentication scenario created")
    
    def _create_mobility_scenario(self):
        """Create high mobility test scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Mobile device scenario
        self.scenario_builder.add_gateway("GW001", coverage_area=800)
        self.scenario_builder.add_gateway("GW002", coverage_area=800)
        self.scenario_builder.add_device("MOBILE001", location=(0, 0), mobility_pattern="VEHICLE")
        
        # Mobility events
        for i in range(10):
            timestamp = i * 30.0  # Every 30 seconds
            location = (i * 100, i * 50)  # Moving trajectory
            self.scenario_builder.add_mobility_event("MOBILE001", location, timestamp)
            
            # Authentication when moving
            gateway = "GW001" if i < 5 else "GW002"
            self.scenario_builder.add_authentication("MOBILE001", gateway, timestamp + 1)
        
        messagebox.showinfo("Scenario", "High Mobility scenario created")
    
    def _create_byzantine_scenario(self):
        """Create Byzantine attack test scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Normal setup
        self.scenario_builder.add_gateway("GW001")
        self.scenario_builder.add_device("DEV001", location=(0, 0))
        
        # Add Byzantine fault events
        self.scenario_builder.add_chaos_event("CORRUPT_STATE", "TrustAnchor001", 10.0)
        self.scenario_builder.add_chaos_event("CORRUPT_STATE", "TrustAnchor002", 15.0)
        
        # Normal authentication attempts during attack
        for i in range(5):
            self.scenario_builder.add_authentication("DEV001", "GW001", 5.0 + i * 5.0)
        
        messagebox.showinfo("Scenario", "Byzantine Attack scenario created")
    
    def _create_chaos_scenario(self):
        """Create chaos engineering scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Complex setup
        for i in range(3):
            self.scenario_builder.add_gateway(f"GW{i:03d}")
            self.scenario_builder.add_device(f"DEV{i:03d}", location=(i*300, i*200))
        
        # Chaos events
        chaos_events = [
            ("DROP_EVENT", "AUTHENTICATION_REQUEST", 20.0),
            ("INJECT_LATENCY", "NETWORK", 30.0, {"duration": 5000}),
            ("CORRUPT_STATE", "DEV001", 40.0),
            ("DROP_EVENT", "PREDICTION_UPDATE", 50.0)
        ]
        
        for chaos_type, target, timestamp, *args in chaos_events:
            kwargs = args[0] if args else {}
            self.scenario_builder.add_chaos_event(chaos_type, target, timestamp, **kwargs)
        
        messagebox.showinfo("Scenario", "Chaos Engineering scenario created")
    
    def _load_scenario(self):
        """Load scenario from file."""
        filename = filedialog.askopenfilename(
            title="Load Scenario",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    scenario_data = json.load(f)
                # Reconstruct scenario builder
                messagebox.showinfo("Success", f"Scenario loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load scenario: {e}")
    
    def _save_scenario(self):
        """Save current scenario to file."""
        filename = filedialog.asksaveasfilename(
            title="Save Scenario",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                scenario_data = {
                    "commands": self.scenario_builder.build(),
                    "parameters": asdict(self.parameters),
                    "created": datetime.now().isoformat()
                }
                with open(filename, 'w') as f:
                    json.dump(scenario_data, f, indent=2)
                messagebox.showinfo("Success", f"Scenario saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save scenario: {e}")
    
    def _clear_scenario(self):
        """Clear current scenario."""
        self.scenario_builder = ScenarioBuilder()
        self.visual_timeline.clear()
        messagebox.showinfo("Cleared", "Scenario cleared")
    
    def _start_simulation(self):
        """Start the simulation."""
        if not self.simulation_running:
            self.simulation_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start simulation in separate thread
            self.simulation_thread = threading.Thread(
                target=self._run_simulation, daemon=True
            )
            self.simulation_thread.start()
            
            messagebox.showinfo("Started", "Simulation started")
    
    def _stop_simulation(self):
        """Stop the simulation."""
        if self.simulation_running:
            self.simulation_running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            messagebox.showinfo("Stopped", "Simulation stopped")
    
    def _reset_simulation(self):
        """Reset the simulation."""
        self._stop_simulation()
        self.visual_timeline.clear()
        self.health_metrics = SystemHealthMetrics()
        self.health_panel.update_metrics(self.health_metrics)
        
        messagebox.showinfo("Reset", "Simulation reset")
    
    def _run_simulation(self):
        """Run the simulation (background thread)."""
        scenario_commands = self.scenario_builder.build()
        
        for command in scenario_commands:
            if not self.simulation_running:
                break
            
            # Process scenario command
            self._process_scenario_command(command)
            
            # Check for real-time parameter updates
            try:
                while True:
                    update = self.simulation_queue.get_nowait()
                    self._apply_parameter_update(update)
            except queue.Empty:
                pass
            
            # Simulate time progression
            time.sleep(0.1)  # 100ms simulation tick
            
            # Update metrics
            self._update_simulation_metrics()
    
    def _process_scenario_command(self, command: Dict):
        """Process a scenario command."""
        command_type = command["type"]
        
        if command_type == "CREATE_DEVICE":
            self.visual_timeline.add_event(
                command["timestamp"], "DEVICE_CREATED", 
                command["device_id"], f"Location: {command['location']}"
            )
        elif command_type == "CREATE_GATEWAY":
            self.visual_timeline.add_event(
                command["timestamp"], "GATEWAY_CREATED",
                command["gateway_id"], f"Coverage: {command['coverage_area']}m"
            )
        elif command_type == "AUTHENTICATE":
            self.visual_timeline.add_event(
                command["timestamp"], "AUTHENTICATION",
                command["device_id"], f"Gateway: {command['gateway_id']}"
            )
        elif command_type == "CHAOS_EVENT":
            self.visual_timeline.add_event(
                command["timestamp"], "CHAOS_EVENT",
                command["target"], f"Type: {command['chaos_type']}"
            )
    
    def _apply_parameter_update(self, update: Dict):
        """Apply real-time parameter update."""
        parameter = update["parameter"]
        value = update["value"]
        
        # Update simulation parameters
        setattr(self.parameters, parameter, value)
        
        print(f"Applied parameter update: {parameter} = {value}")
    
    def _update_simulation_metrics(self):
        """Update simulation metrics."""
        # Simulate metrics collection
        # In real implementation, collect from actual system components
        
        self.health_metrics.authentication_success_rate = np.random.uniform(92, 99)
        self.health_metrics.average_latency_ms = self.parameters.network_latency_ms + np.random.uniform(-10, 20)
        self.health_metrics.throughput_auths_per_second = np.random.uniform(5, 15)
        self.health_metrics.prediction_accuracy_meters = self.parameters.ai_prediction_error_meters + np.random.uniform(-20, 30)
        self.health_metrics.network_utilization_percent = np.random.uniform(20, 80)
        self.health_metrics.security_violations = max(0, int(np.random.poisson(self.parameters.byzantine_fault_ratio * 10)))
        self.health_metrics.byzantine_detections = max(0, int(np.random.poisson(self.parameters.byzantine_fault_ratio * 5)))
        self.health_metrics.privacy_budget_remaining = max(0.1, self.health_metrics.privacy_budget_remaining - 0.001)
        
        # Update health panel
        self.health_panel.update_metrics(self.health_metrics)
    
    def _on_close(self):
        """Handle window close event."""
        if self.simulation_running:
            self._stop_simulation()
        
        self.health_panel.stop_monitoring()
        self.root.destroy()
    
    def run(self):
        """Run the dashboard application."""
        self.root.mainloop()


def main():
    """Main entry point for the dashboard."""
    print("🚀 Starting ZKPAS Interactive Research Dashboard...")
    
    try:
        dashboard = ZKPASResearchDashboard()
        dashboard.run()
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()