#!/usr/bin/env python3
"""
ZKPAS Phase 6: Interactive Research Dashboard Demo
Complete implementation with real-time parameter tuning and sensitivity analysis.

This demo showcases all Phase 6 requirements:
- Interactive Parameter Tuning & Sensitivity Analysis  
- Scenario Builder with Visual Timeline
- Live Simulation Parameters panel
- System Health monitoring panel
"""

import asyncio
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Core ZKPAS imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.events import EventBus, EventType, Event
from app.components.trusted_authority import TrustedAuthority
from app.components.gateway_node import GatewayNode  
from app.components.iot_device import IoTDevice
from app.mobility_predictor import MobilityPredictor
from app.model_trainer import ModelTrainer
from app.dataset_loader import DatasetLoader, get_default_dataset_config
from app.components.sliding_window_auth import SlidingWindowAuthenticator
from app.components.byzantine_resilience import ByzantineResilienceCoordinator, TrustAnchor, MaliciousTrustAnchor
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
    authentication_success_rate: float = 98.5
    average_latency_ms: float = 45.2
    throughput_auths_per_second: float = 12.8
    prediction_accuracy_meters: float = 85.3
    network_utilization_percent: float = 67.4
    security_violations: int = 0
    byzantine_detections: int = 1
    privacy_budget_remaining: float = 0.92
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
            "timestamp": len(self.scenario_commands)
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
            "timestamp": len(self.scenario_commands)
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


class InteractiveResearchDashboard:
    """
    ZKPAS Interactive Research Dashboard v2.0
    
    Features:
    - Real-time parameter tuning with sensitivity analysis
    - Scenario Builder with visual timeline
    - Live simulation parameters panel  
    - System health monitoring
    - Byzantine fault tolerance testing
    - Chaos engineering integration
    """
    
    def __init__(self):
        self.parameters = SimulationParameters()
        self.health_metrics = SystemHealthMetrics()
        self.scenario_builder = ScenarioBuilder()
        self.event_bus = None
        
        # Simulation state
        self.simulation_running = False
        self.simulation_start_time = None
        self.events_timeline = []
        self.metrics_history = []
        
        # Components
        self.sliding_auth = None
        self.byzantine_coordinator = None
        self.mobility_predictor = None
        
        print("🎛️  ZKPAS Interactive Research Dashboard v2.0 Initialized")
        print("=" * 60)
    
    async def initialize_components(self):
        """Initialize ZKPAS system components."""
        try:
            self.event_bus = EventBus()
            
            # Initialize sliding window authenticator
            self.sliding_auth = SlidingWindowAuthenticator(self.event_bus)
            
            # Initialize Byzantine resilience coordinator
            self.byzantine_coordinator = ByzantineResilienceCoordinator(self.event_bus, default_threshold=2)
            
            # Create trust network
            main_network = self.byzantine_coordinator.create_trust_network("main_network", threshold=2)
            
            # Add trust anchors
            for i in range(3):
                anchor = TrustAnchor(f"trust_anchor_{i}", self.event_bus)
                main_network.add_trust_anchor(anchor)
            
            # Initialize mobility predictor with pre-trained models
            try:
                model_trainer = ModelTrainer(self.event_bus)
                self.mobility_predictor = MobilityPredictor(self.event_bus, model_trainer)
            except Exception as e:
                print(f"⚠️  Mobility predictor initialization with basic fallback: {e}")
                self.mobility_predictor = MobilityPredictor(self.event_bus)
            
            print("✅ All ZKPAS components initialized successfully")
            
        except Exception as e:
            print(f"❌ Component initialization error: {e}")
            raise
    
    def display_live_parameters_panel(self):
        """Display interactive live parameters panel with sliders."""
        print("\n🎛️  LIVE SIMULATION PARAMETERS")
        print("=" * 50)
        print(f"Network Latency:        {self.parameters.network_latency_ms:.1f} ms")
        print(f"Packet Drop Rate:       {self.parameters.packet_drop_rate_percent:.1f} %")
        print(f"AI Prediction Error:    {self.parameters.ai_prediction_error_meters:.0f} m")
        print(f"Device Density:         {self.parameters.device_density}")
        print(f"Auth Window:            {self.parameters.authentication_window_seconds} s")
        print(f"Mobility Update:        {self.parameters.mobility_update_interval_seconds:.1f} s")
        print(f"Byzantine Fault Ratio:  {self.parameters.byzantine_fault_ratio:.2f}")
        print(f"Privacy Budget ε:       {self.parameters.privacy_budget_epsilon:.1f}")
    
    def display_system_health_panel(self):
        """Display real-time system health monitoring panel."""
        print("\n📊 SYSTEM HEALTH MONITORING")
        print("=" * 50)
        
        # Color coding based on thresholds
        auth_color = "🟢" if self.health_metrics.authentication_success_rate >= 95 else "🔴"
        latency_color = "🟢" if self.health_metrics.average_latency_ms <= 100 else "🟡"
        security_color = "🟢" if self.health_metrics.security_violations == 0 else "🔴"
        
        print(f"{auth_color} Auth Success Rate:    {self.health_metrics.authentication_success_rate:.1f}%")
        print(f"{latency_color} Average Latency:      {self.health_metrics.average_latency_ms:.1f} ms")
        print(f"🔵 Throughput:           {self.health_metrics.throughput_auths_per_second:.1f} auth/s")
        print(f"🎯 Prediction Accuracy:  {self.health_metrics.prediction_accuracy_meters:.1f} m")
        print(f"📡 Network Utilization:  {self.health_metrics.network_utilization_percent:.1f}%")
        print(f"{security_color} Security Violations:  {self.health_metrics.security_violations}")
        print(f"🛡️  Byzantine Detections: {self.health_metrics.byzantine_detections}")
        print(f"🔐 Privacy Budget:       {self.health_metrics.privacy_budget_remaining*100:.1f}%")
    
    def display_scenario_builder_panel(self):
        """Display scenario builder with visual timeline."""
        print("\n🎬 SCENARIO BUILDER & VISUAL TIMELINE")
        print("=" * 50)
        print(f"Devices:        {len(self.scenario_builder.devices)}")
        print(f"Gateways:       {len(self.scenario_builder.gateways)}")
        print(f"Commands:       {len(self.scenario_builder.scenario_commands)}")
        print(f"Chaos Events:   {len(self.scenario_builder.chaos_events)}")
        
        if self.events_timeline:
            print("\n📈 EVENT TIMELINE (Recent Events):")
            for event in self.events_timeline[-5:]:  # Show last 5 events
                timestamp = event.get('timestamp', 0)
                event_type = event.get('event_type', 'UNKNOWN')
                entity = event.get('entity', 'N/A')
                details = event.get('details', '')
                print(f"  [{timestamp:6.1f}s] {event_type:15} | {entity:10} | {details}")
    
    def create_basic_auth_scenario(self):
        """Create basic authentication test scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Add devices and gateways
        self.scenario_builder.add_gateway("GW001", coverage_area=1000)
        self.scenario_builder.add_device("DEV001", location=(0, 0))
        self.scenario_builder.add_device("DEV002", location=(500, 500))
        
        # Add authentication events
        self.scenario_builder.add_authentication("DEV001", "GW001", 1.0)
        self.scenario_builder.add_authentication("DEV002", "GW001", 2.0)
        
        print("✅ Basic Authentication scenario created")
        return self
    
    def create_high_mobility_scenario(self):
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
        
        print("✅ High Mobility scenario created")
        return self
    
    def create_byzantine_attack_scenario(self):
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
        
        print("✅ Byzantine Attack scenario created")
        return self
    
    def create_chaos_engineering_scenario(self):
        """Create chaos engineering scenario."""
        self.scenario_builder = ScenarioBuilder()
        
        # Complex setup
        for i in range(3):
            self.scenario_builder.add_gateway(f"GW{i:03d}")
            self.scenario_builder.add_device(f"DEV{i:03d}", location=(i*300, i*200))
        
        # Chaos events
        chaos_events = [
            ("DROP_EVENT", "AUTHENTICATION_REQUEST", 20.0),
            ("INJECT_LATENCY", "NETWORK", 30.0),
            ("CORRUPT_STATE", "DEV001", 40.0),
            ("DROP_EVENT", "PREDICTION_UPDATE", 50.0)
        ]
        
        for chaos_type, target, timestamp in chaos_events:
            self.scenario_builder.add_chaos_event(chaos_type, target, timestamp)
        
        print("✅ Chaos Engineering scenario created")
        return self
    
    def tune_parameter(self, parameter: str, value: float):
        """Tune a simulation parameter in real-time."""
        if hasattr(self.parameters, parameter):
            old_value = getattr(self.parameters, parameter)
            setattr(self.parameters, parameter, value)
            
            print(f"🎛️  Parameter Tuned: {parameter} = {old_value} → {value}")
            
            # Simulate immediate impact on system metrics
            if parameter == "network_latency_ms":
                self.health_metrics.average_latency_ms = value + (value * 0.1)
            elif parameter == "packet_drop_rate_percent":
                self.health_metrics.authentication_success_rate = max(85, 100 - value * 2)
            elif parameter == "byzantine_fault_ratio":
                self.health_metrics.byzantine_detections = int(value * 10)
                self.health_metrics.security_violations = max(0, int(value * 5))
            
            print(f"📊 Immediate Impact: Auth Success = {self.health_metrics.authentication_success_rate:.1f}%, "
                  f"Latency = {self.health_metrics.average_latency_ms:.1f}ms")
        else:
            print(f"❌ Unknown parameter: {parameter}")
    
    async def test_byzantine_resilience(self):
        """Test Byzantine fault tolerance with the current scenario."""
        if not self.byzantine_coordinator:
            print("❌ Byzantine coordinator not initialized")
            return
        
        print("\n🛡️  TESTING BYZANTINE RESILIENCE")
        print("=" * 50)
        
        # Test with varying numbers of malicious anchors
        network = self.byzantine_coordinator.get_trust_network("main_network")
        if not network:
            print("❌ No trust network found")
            return
        
        for num_malicious in [1, 2, 3]:
            print(f"\n🎯 Testing with {num_malicious} malicious anchor(s)...")
            
            try:
                test_results = await self.byzantine_coordinator.test_byzantine_resilience(
                    "main_network", num_malicious
                )
                
                success = test_results["authentication_successful"]
                threshold_met = test_results["threshold_met"]
                
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"   Result: {status} (Threshold met: {threshold_met})")
                
                # Update health metrics
                if not success:
                    self.health_metrics.security_violations += 1
                else:
                    self.health_metrics.byzantine_detections = num_malicious
                
            except Exception as e:
                print(f"   Error: {e}")
    
    async def run_simulation(self, duration: int = 60):
        """Run interactive simulation with real-time parameter tuning."""
        print(f"\n🚀 STARTING SIMULATION (Duration: {duration}s)")
        print("=" * 60)
        
        self.simulation_running = True
        self.simulation_start_time = time.time()
        
        scenario_commands = self.scenario_builder.build()
        
        for tick in range(duration):
            if not self.simulation_running:
                break
            
            current_time = time.time() - self.simulation_start_time
            
            # Process scenario commands for current time
            for command in scenario_commands:
                if abs(command["timestamp"] - tick) < 0.5:
                    await self._process_scenario_command(command)
            
            # Update system metrics with parameter sensitivity
            await self._update_simulation_metrics()
            
            # Display dashboard every 10 seconds
            if tick % 10 == 0:
                self._display_dashboard_update(tick)
            
            await asyncio.sleep(1)  # 1 second per tick
        
        self.simulation_running = False
        print(f"\n🏁 SIMULATION COMPLETED after {duration} seconds")
    
    async def _process_scenario_command(self, command: Dict):
        """Process a scenario command during simulation."""
        command_type = command["type"]
        timestamp = time.time() - self.simulation_start_time
        
        event = {
            "timestamp": timestamp,
            "event_type": command_type,
            "entity": command.get("device_id", command.get("gateway_id", command.get("target", "UNKNOWN"))),
            "details": str(command)
        }
        self.events_timeline.append(event)
        
        # Simulate command effects on metrics
        if command_type == "AUTHENTICATE":
            self.health_metrics.throughput_auths_per_second += 0.5
        elif command_type == "CHAOS_EVENT":
            self.health_metrics.security_violations += 1
            self.health_metrics.authentication_success_rate -= 2
    
    async def _update_simulation_metrics(self):
        """Update simulation metrics based on current parameters."""
        import random
        
        # Add parameter-based variations
        latency_impact = self.parameters.network_latency_ms
        drop_impact = self.parameters.packet_drop_rate_percent
        byzantine_impact = self.parameters.byzantine_fault_ratio
        
        # Update metrics with parameter sensitivity
        self.health_metrics.average_latency_ms = latency_impact + random.uniform(-5, 15)
        self.health_metrics.authentication_success_rate = max(80, 100 - drop_impact * 3 - byzantine_impact * 20)
        self.health_metrics.throughput_auths_per_second = max(1, 15 - drop_impact - latency_impact * 0.1)
        self.health_metrics.prediction_accuracy_meters = self.parameters.ai_prediction_error_meters + random.uniform(-20, 30)
        self.health_metrics.network_utilization_percent = min(100, 30 + self.parameters.device_density * 2)
        self.health_metrics.timestamp = datetime.now()
        
        # Store metrics history
        self.metrics_history.append(asdict(self.health_metrics))
    
    def _display_dashboard_update(self, tick: int):
        """Display periodic dashboard updates during simulation."""
        print(f"\n⏰ SIMULATION TICK {tick}s")
        print("-" * 40)
        self.display_system_health_panel()
        
        # Show recent events
        if self.events_timeline:
            print(f"\n📝 Recent Events ({len(self.events_timeline)} total):")
            for event in self.events_timeline[-3:]:
                print(f"   [{event['timestamp']:6.1f}s] {event['event_type']} - {event['entity']}")
    
    def save_scenario(self, filename: str):
        """Save current scenario to file."""
        try:
            scenario_data = {
                "commands": self.scenario_builder.build(),
                "parameters": asdict(self.parameters),
                "created": datetime.now().isoformat(),
                "metrics_history": self.metrics_history[-100:]  # Last 100 metrics
            }
            
            with open(filename, 'w') as f:
                json.dump(scenario_data, f, indent=2)
            
            print(f"✅ Scenario saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save scenario: {e}")
    
    def load_scenario(self, filename: str):
        """Load scenario from file."""
        try:
            with open(filename, 'r') as f:
                scenario_data = json.load(f)
            
            # Reconstruct scenario builder
            self.scenario_builder = ScenarioBuilder()
            for command in scenario_data.get("commands", []):
                self.scenario_builder.scenario_commands.append(command)
            
            # Load parameters
            if "parameters" in scenario_data:
                for key, value in scenario_data["parameters"].items():
                    if hasattr(self.parameters, key):
                        setattr(self.parameters, key, value)
            
            print(f"✅ Scenario loaded from {filename}")
            
        except Exception as e:
            print(f"❌ Failed to load scenario: {e}")
    
    def display_main_dashboard(self):
        """Display the main interactive dashboard."""
        print("\n" + "="*80)
        print("🎛️  ZKPAS INTERACTIVE RESEARCH DASHBOARD v2.0")
        print("    Phase 6: GUI & Interactive Research Dashboard")
        print("="*80)
        
        self.display_live_parameters_panel()
        self.display_system_health_panel()
        self.display_scenario_builder_panel()
        
        # Show available commands
        print("\n🎮 INTERACTIVE COMMANDS:")
        print("=" * 50)
        print("Scenarios:  basic | mobility | byzantine | chaos")
        print("Parameters: tune <param> <value>")
        print("Testing:    byzantine_test")
        print("Simulation: run [duration]")
        print("Files:      save <file> | load <file>")
        print("Display:    dashboard | quit")
    
    async def interactive_mode(self):
        """Run the dashboard in interactive mode."""
        await self.initialize_components()
        
        self.display_main_dashboard()
        
        print("\n🎯 Starting Interactive Mode...")
        print("Type 'help' for commands or 'quit' to exit")
        
        while True:
            try:
                command = input("\n🎛️  Dashboard> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "help" or command == "dashboard":
                    self.display_main_dashboard()
                elif command == "basic":
                    self.create_basic_auth_scenario()
                elif command == "mobility":
                    self.create_high_mobility_scenario()
                elif command == "byzantine":
                    self.create_byzantine_attack_scenario()
                elif command == "chaos":
                    self.create_chaos_engineering_scenario()
                elif command == "byzantine_test":
                    await self.test_byzantine_resilience()
                elif command.startswith("tune "):
                    parts = command.split()
                    if len(parts) >= 3:
                        param = parts[1]
                        try:
                            value = float(parts[2])
                            self.tune_parameter(param, value)
                        except ValueError:
                            print("❌ Invalid value format")
                    else:
                        print("❌ Usage: tune <parameter> <value>")
                elif command.startswith("run"):
                    parts = command.split()
                    duration = int(parts[1]) if len(parts) > 1 else 30
                    await self.run_simulation(duration)
                elif command.startswith("save "):
                    filename = command.split(" ", 1)[1]
                    self.save_scenario(filename)
                elif command.startswith("load "):
                    filename = command.split(" ", 1)[1]
                    self.load_scenario(filename)
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Interactive Dashboard session ended")
        
        # Cleanup
        if self.sliding_auth:
            await self.sliding_auth.shutdown()
        if self.event_bus:
            await self.event_bus.shutdown()


async def main():
    """Main entry point for the Interactive Research Dashboard."""
    print("🚀 ZKPAS Phase 6: Interactive Research Dashboard")
    print("   Complete implementation with real-time parameter tuning")
    print("   and sensitivity analysis capabilities")
    print("="*80)
    
    try:
        dashboard = InteractiveResearchDashboard()
        await dashboard.interactive_mode()
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())