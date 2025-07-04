#!/usr/bin/env python3
"""
ZKPAS System Health Monitoring Demo

This demo provides comprehensive system health monitoring and diagnostics.
"""

import asyncio
import sys
import time
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the zkpas directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.events import EventBus, EventType
from shared.crypto_utils import generate_ecc_keypair, secure_hash
from loguru import logger


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.start_time = time.time()
        self.health_checks = []
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        print("🖥️  Checking system resources...")
        
        # CPU Information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory Information
        memory = psutil.virtual_memory()
        
        # Disk Information
        disk = psutil.disk_usage('/')
        
        # Network Information
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "status": "healthy" if cpu_percent < 80 else "warning" if cpu_percent < 90 else "critical"
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "usage_percent": memory.percent,
                "status": "healthy" if memory.percent < 80 else "warning" if memory.percent < 90 else "critical"
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "usage_percent": (disk.used / disk.total) * 100,
                "status": "healthy" if (disk.used / disk.total) < 0.8 else "warning"
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        }
    
    async def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment health."""
        print("🐍 Checking Python environment...")
        
        import threading
        
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "active_threads": threading.active_count(),
            "asyncio_running": asyncio.current_task() is not None
        }
    
    async def check_crypto_operations(self) -> Dict[str, Any]:
        """Check cryptographic operations health."""
        print("🔐 Checking cryptographic operations...")
        
        try:
            # Test key generation
            start_time = time.perf_counter()
            private_key, public_key = generate_ecc_keypair()
            keygen_time = time.perf_counter() - start_time
            
            # Test hashing
            start_time = time.perf_counter()
            test_data = b"ZKPAS health check test data"
            hash_result = secure_hash(test_data)
            hash_time = time.perf_counter() - start_time
            
            return {
                "key_generation": {
                    "time_ms": keygen_time * 1000,
                    "status": "healthy" if keygen_time < 0.1 else "warning",
                    "success": True
                },
                "hashing": {
                    "time_ms": hash_time * 1000,
                    "status": "healthy" if hash_time < 0.01 else "warning",
                    "success": True
                },
                "overall_status": "healthy"
            }
        except Exception as e:
            return {
                "key_generation": {"status": "critical", "success": False, "error": str(e)},
                "hashing": {"status": "critical", "success": False, "error": str(e)},
                "overall_status": "critical"
            }
    
    async def check_event_system(self) -> Dict[str, Any]:
        """Check event system health."""
        print("📡 Checking event system...")
        
        try:
            events_published = 0
            events_received = 0
            
            def test_handler(event):
                nonlocal events_received
                events_received += 1
            
            # Subscribe to test events
            self.event_bus.subscribe_sync(EventType.DEVICE_AUTHENTICATED, test_handler)
            
            # Publish test events
            start_time = time.perf_counter()
            for i in range(10):
                await self.event_bus.publish_event(
                    EventType.DEVICE_AUTHENTICATED,
                    correlation_id=f"health_test_{i}",
                    source="health_monitor",
                    target="test_device",
                    data={"test": True}
                )
                events_published += 1
            
            # Allow processing time
            await asyncio.sleep(0.1)
            
            processing_time = time.perf_counter() - start_time
            
            return {
                "events_published": events_published,
                "events_received": events_received,
                "processing_time_ms": processing_time * 1000,
                "delivery_rate": events_received / events_published if events_published > 0 else 0,
                "status": "healthy" if events_received == events_published else "warning"
            }
        except Exception as e:
            return {
                "status": "critical",
                "error": str(e),
                "events_published": 0,
                "events_received": 0
            }
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        print("📦 Checking dependencies...")
        
        dependencies = {}
        
        # Check required modules
        required_modules = [
            'cryptography', 'numpy', 'pandas', 'scikit-learn',
            'matplotlib', 'loguru', 'asyncio', 'psutil'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
                dependencies[module_name] = {
                    "status": "available",
                    "version": getattr(__import__(module_name), '__version__', 'unknown')
                }
            except ImportError as e:
                dependencies[module_name] = {
                    "status": "missing",
                    "error": str(e)
                }
        
        missing_count = sum(1 for dep in dependencies.values() if dep["status"] == "missing")
        overall_status = "healthy" if missing_count == 0 else "critical"
        
        return {
            "dependencies": dependencies,
            "missing_count": missing_count,
            "total_checked": len(required_modules),
            "overall_status": overall_status
        }
    
    async def check_file_system(self) -> Dict[str, Any]:
        """Check file system and permissions."""
        print("📁 Checking file system...")
        
        try:
            zkpas_dir = Path(__file__).parent
            
            # Check directory structure
            required_dirs = [
                'app', 'shared', 'data', 'tests', 'scripts', 'docs'
            ]
            
            dir_status = {}
            for dir_name in required_dirs:
                dir_path = zkpas_dir / dir_name
                dir_status[dir_name] = {
                    "exists": dir_path.exists(),
                    "readable": dir_path.is_dir() and os.access(dir_path, os.R_OK) if dir_path.exists() else False,
                    "writable": dir_path.is_dir() and os.access(dir_path, os.W_OK) if dir_path.exists() else False
                }
            
            # Check write permissions in temp/log areas
            import tempfile
            import os
            
            temp_test = False
            try:
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    tmp.write(b"test")
                    temp_test = True
            except:
                pass
            
            missing_dirs = sum(1 for status in dir_status.values() if not status["exists"])
            overall_status = "healthy" if missing_dirs == 0 and temp_test else "warning"
            
            return {
                "directories": dir_status,
                "temp_write_test": temp_test,
                "missing_directories": missing_dirs,
                "overall_status": overall_status
            }
        except Exception as e:
            return {
                "overall_status": "critical",
                "error": str(e)
            }
    
    async def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check."""
        print("🏥 ZKPAS System Health Check")
        print("=" * 50)
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            # 1. System Resources
            print("\n1️⃣ System Resources")
            health_report["checks"]["system"] = await self.check_system_resources()
            
            cpu_status = health_report["checks"]["system"]["cpu"]["status"]
            memory_status = health_report["checks"]["system"]["memory"]["status"]
            print(f"   CPU Usage: {health_report['checks']['system']['cpu']['usage_percent']:.1f}% ({cpu_status})")
            print(f"   Memory Usage: {health_report['checks']['system']['memory']['usage_percent']:.1f}% ({memory_status})")
            
            # 2. Python Environment
            print("\n2️⃣ Python Environment")
            health_report["checks"]["python"] = await self.check_python_environment()
            
            py_version = health_report["checks"]["python"]["python_version"]
            platform_info = health_report["checks"]["python"]["platform"]
            print(f"   Python Version: {py_version}")
            print(f"   Platform: {platform_info}")
            
            # 3. Dependencies
            print("\n3️⃣ Dependencies")
            health_report["checks"]["dependencies"] = await self.check_dependencies()
            
            dep_status = health_report["checks"]["dependencies"]["overall_status"]
            missing = health_report["checks"]["dependencies"]["missing_count"]
            total = health_report["checks"]["dependencies"]["total_checked"]
            print(f"   Dependencies: {total - missing}/{total} available ({dep_status})")
            
            # 4. Cryptographic Operations
            print("\n4️⃣ Cryptographic Operations")
            health_report["checks"]["crypto"] = await self.check_crypto_operations()
            
            crypto_status = health_report["checks"]["crypto"]["overall_status"]
            keygen_time = health_report["checks"]["crypto"]["key_generation"]["time_ms"]
            print(f"   Crypto Operations: {crypto_status}")
            print(f"   Key Generation: {keygen_time:.2f}ms")
            
            # 5. Event System
            print("\n5️⃣ Event System")
            health_report["checks"]["events"] = await self.check_event_system()
            
            event_status = health_report["checks"]["events"]["status"]
            delivery_rate = health_report["checks"]["events"]["delivery_rate"]
            print(f"   Event System: {event_status}")
            print(f"   Event Delivery: {delivery_rate*100:.1f}%")
            
            # 6. File System
            print("\n6️⃣ File System")
            health_report["checks"]["filesystem"] = await self.check_file_system()
            
            fs_status = health_report["checks"]["filesystem"]["overall_status"]
            missing_dirs = health_report["checks"]["filesystem"]["missing_directories"]
            print(f"   File System: {fs_status}")
            print(f"   Missing Directories: {missing_dirs}")
            
            # Generate Overall Health Score
            print("\n🎯 HEALTH SUMMARY")
            print("=" * 30)
            
            status_scores = {
                "healthy": 100,
                "warning": 70,
                "critical": 0
            }
            
            checks = health_report["checks"]
            scores = []
            
            # Calculate individual scores
            scores.append(status_scores.get(checks["system"]["cpu"]["status"], 0))
            scores.append(status_scores.get(checks["system"]["memory"]["status"], 0))
            scores.append(status_scores.get(checks["dependencies"]["overall_status"], 0))
            scores.append(status_scores.get(checks["crypto"]["overall_status"], 0))
            scores.append(status_scores.get(checks["events"]["status"], 0))
            scores.append(status_scores.get(checks["filesystem"]["overall_status"], 0))
            
            overall_score = sum(scores) / len(scores)
            health_report["overall_score"] = overall_score
            
            # Display health indicators
            print(f"🖥️  System Resources: {self._get_status_emoji(checks['system']['cpu']['status'])}")
            print(f"📦 Dependencies: {self._get_status_emoji(checks['dependencies']['overall_status'])}")
            print(f"🔐 Cryptography: {self._get_status_emoji(checks['crypto']['overall_status'])}")
            print(f"📡 Event System: {self._get_status_emoji(checks['events']['status'])}")
            print(f"📁 File System: {self._get_status_emoji(checks['filesystem']['overall_status'])}")
            
            print(f"\n📊 Overall Health Score: {overall_score:.1f}/100")
            
            # Health Grade
            if overall_score >= 90:
                grade = "🟢 EXCELLENT"
            elif overall_score >= 80:
                grade = "🟡 GOOD"
            elif overall_score >= 60:
                grade = "🟠 FAIR"
            else:
                grade = "🔴 POOR"
            
            print(f"🏆 System Health: {grade}")
            
            # Generate recommendations
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 20)
            
            recommendations = []
            
            if checks["system"]["cpu"]["status"] != "healthy":
                recommendations.append("• Monitor CPU usage - consider reducing load")
            
            if checks["system"]["memory"]["status"] != "healthy":
                recommendations.append("• Monitor memory usage - consider increasing RAM")
            
            if checks["dependencies"]["missing_count"] > 0:
                recommendations.append("• Install missing dependencies")
            
            if checks["crypto"]["overall_status"] != "healthy":
                recommendations.append("• Check cryptographic library installation")
            
            if checks["events"]["status"] != "healthy":
                recommendations.append("• Review event system configuration")
            
            if checks["filesystem"]["overall_status"] != "healthy":
                recommendations.append("• Check file permissions and directory structure")
            
            if not recommendations:
                recommendations.append("• System health is optimal!")
            
            for rec in recommendations:
                print(rec)
            
            health_report["recommendations"] = recommendations
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_report["error"] = str(e)
            print(f"❌ Health check failed: {e}")
        
        return health_report
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        return {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌"
        }.get(status, "❓")


async def main():
    """Main entry point for system health monitoring."""
    print("🏥 Welcome to ZKPAS System Health Monitor!")
    print("This tool provides comprehensive system diagnostics.")
    print()
    
    monitor = SystemHealthMonitor()
    
    try:
        health_report = await monitor.run_comprehensive_health_check()
        
        print(f"\n⏰ Health check completed at: {health_report['timestamp']}")
        print("✨ System health monitoring completed!")
        
        return health_report
        
    except Exception as e:
        print(f"❌ Health monitoring failed: {e}")
        return None


if __name__ == "__main__":
    import os
    asyncio.run(main())
