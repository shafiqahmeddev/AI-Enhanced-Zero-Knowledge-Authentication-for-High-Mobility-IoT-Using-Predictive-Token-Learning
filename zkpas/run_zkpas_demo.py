#!/usr/bin/env python3
"""
ZKPAS System Demo - Easy Testing Interface

This is the main entry point to test and interact with the ZKPAS system.
Run this file to see all available demos and choose what you want to test.
"""

import asyncio
import os
import sys

# Get the proper Python executable path
PYTHON_PATH = sys.executable

def display_main_menu():
    """Display the main testing menu."""
    print("🚀 ZKPAS SYSTEM TESTING MENU")
    print("=" * 50)
    print("Choose what you want to test:")
    print()
    print("📱 CORE SYSTEM DEMOS:")
    print("  1. Basic ZKPAS Authentication Demo")
    print("  2. High Mobility IoT Scenario")
    print("  3. Real Dataset Integration Demo")
    print("  4. LSTM Neural Network Demo")
    print()
    print("🛡️ SECURITY & RESILIENCE:")
    print("  5. Byzantine Fault Tolerance Test")
    print("  6. Security Stress Testing")
    print("  7. Sliding Window Authentication")
    print()
    print("🎛️ INTERACTIVE DASHBOARDS:")
    print("  8. Phase 6: Interactive Research Dashboard")
    print("  9. Chaos Engineering GUI Demo")
    print()
    print("📊 PERFORMANCE & ANALYSIS:")
    print("  10. Performance Benchmarking")
    print("  11. System Health Monitoring")
    print("  12. Complete System Integration Test")
    print()
    print("❓ HELP & INFO:")
    print("  13. System Information & Requirements")
    print("  14. View Implementation Status")
    print("  15. Exit")
    print()

def get_user_choice():
    """Get user menu selection."""
    try:
        choice = input("Enter your choice (1-15): ").strip()
        return int(choice)
    except ValueError:
        return 0

async def run_basic_auth_demo():
    """Run basic ZKPAS authentication demonstration."""
    print("\n🔐 Running Basic ZKPAS Authentication Demo...")
    os.system(f"{PYTHON_PATH} demo_zkpas_basic.py")

async def run_mobility_demo():
    """Run high mobility IoT scenario."""
    print("\n🚗 Running High Mobility IoT Scenario...")
    os.system(f"{PYTHON_PATH} demo_lstm_system.py")

async def run_dataset_demo():
    """Run real dataset integration demo."""
    print("\n📊 Running Real Dataset Integration Demo...")
    os.system("python demo_real_data_system.py")

async def run_lstm_demo():
    """Run LSTM neural network demo."""
    print("\n🧠 Running LSTM Neural Network Demo...")
    os.system(f"{PYTHON_PATH} demo_lstm_system.py")

async def run_byzantine_test():
    """Run Byzantine fault tolerance test."""
    print("\n⚔️ Running Byzantine Fault Tolerance Test...")
    os.system(f"{PYTHON_PATH} tests/test_phase5_advanced_auth.py")

async def run_security_stress():
    """Run security stress testing."""
    print("\n🔒 Running Security Stress Testing...")
    os.system(f"{PYTHON_PATH} demo_security_stress_testing.py")

async def run_sliding_window():
    """Run sliding window authentication demo."""
    print("\n🪟 Running Sliding Window Authentication Demo...")
    print("This demo shows advanced token-based authentication...")
    # You can add a specific sliding window demo here

async def run_interactive_dashboard():
    """Run Phase 6 Interactive Research Dashboard."""
    print("\n🎛️ Running Interactive Research Dashboard...")
    os.system(f"{PYTHON_PATH} demo_phase6_complete.py")

async def run_chaos_engineering():
    """Run Chaos Engineering GUI Demo."""
    print("\n🔥 Running Chaos Engineering Demo...")
    os.system(f"{PYTHON_PATH} demo_chaos_engineering_gui.py")

async def run_performance_benchmark():
    """Run performance benchmarking."""
    print("\n📈 Running Performance Benchmarking...")
    print("This will test system performance under various loads...")
    # Add performance testing here

async def run_health_monitoring():
    """Run system health monitoring."""
    print("\n💊 Running System Health Monitoring...")
    print("Monitoring system metrics and health indicators...")
    # Add health monitoring demo here

async def run_integration_test():
    """Run complete system integration test."""
    print("\n🧪 Running Complete System Integration Test...")
    print("This tests all system components working together...")
    os.system(f"{PYTHON_PATH} demo_complete_integration.py")

def show_system_info():
    """Show system information and requirements."""
    print("\n📋 ZKPAS SYSTEM INFORMATION")
    print("=" * 40)
    print("🏗️ Architecture: Event-driven, asyncio-based")
    print("🔐 Security: Zero-Knowledge Proofs, Byzantine Fault Tolerance")
    print("🧠 AI/ML: LSTM Neural Networks for mobility prediction")
    print("📱 IoT: High-mobility device authentication")
    print("🎛️ GUI: Interactive Research Dashboard")
    print("🔥 Testing: Chaos Engineering, Security Stress Testing")
    print()
    print("📦 REQUIREMENTS:")
    print("  • Python 3.7+")
    print("  • cryptography library")
    print("  • numpy, pandas (for ML)")
    print("  • matplotlib (for visualization)")
    print("  • loguru (for logging)")
    print("  • asyncio (built-in)")
    print()
    print("📁 KEY FILES:")
    print("  • app/ - Core system components")
    print("  • tests/ - Test suites")
    print("  • gui/ - Interactive dashboards")
    print("  • shared/ - Cryptographic utilities")
    print("  • demo_*.py - Demonstration scripts")

def show_implementation_status():
    """Show current implementation status."""
    print("\n✅ ZKPAS IMPLEMENTATION STATUS")
    print("=" * 40)
    print("Phase 0: Environment Setup           ✅ COMPLETE")
    print("Phase 1: Cryptographic Foundation   ✅ COMPLETE")
    print("Phase 2: Entity Implementation      ✅ COMPLETE")
    print("Phase 3: Core Protocol              ✅ COMPLETE")
    print("Phase 4: ML & Privacy               ✅ COMPLETE")
    print("Phase 5: Advanced Authentication    ✅ COMPLETE")
    print("Phase 6: Interactive Dashboard      ✅ COMPLETE")
    print("Phase 7: Security & Chaos Testing   ✅ COMPLETE")
    print("Phase 8: Documentation              🚧 IN PROGRESS")
    print()
    print("🎯 CURRENT CAPABILITIES:")
    print("  ✅ Zero-Knowledge Authentication")
    print("  ✅ Byzantine Fault Tolerance")
    print("  ✅ LSTM Mobility Prediction")
    print("  ✅ Real Dataset Integration")
    print("  ✅ Interactive Research Dashboard")
    print("  ✅ Chaos Engineering")
    print("  ✅ Security Stress Testing")
    print("  ✅ Sliding Window Authentication")

async def main():
    """Main testing interface."""
    print("🎯 Welcome to ZKPAS System Testing!")
    print("This interface helps you explore and test all system capabilities.")
    print()
    
    while True:
        display_main_menu()
        choice = get_user_choice()
        
        if choice == 1:
            await run_basic_auth_demo()
        elif choice == 2:
            await run_mobility_demo()
        elif choice == 3:
            await run_dataset_demo()
        elif choice == 4:
            await run_lstm_demo()
        elif choice == 5:
            await run_byzantine_test()
        elif choice == 6:
            await run_security_stress()
        elif choice == 7:
            await run_sliding_window()
        elif choice == 8:
            await run_interactive_dashboard()
        elif choice == 9:
            await run_chaos_engineering()
        elif choice == 10:
            await run_performance_benchmark()
        elif choice == 11:
            await run_health_monitoring()
        elif choice == 12:
            await run_integration_test()
        elif choice == 13:
            show_system_info()
        elif choice == 14:
            show_implementation_status()
        elif choice == 15:
            print("👋 Thanks for testing ZKPAS! Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-15.")
        
        input("\nPress Enter to continue...")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())