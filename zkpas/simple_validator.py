#!/usr/bin/env python3
"""
Simple Module Validator for ZKPAS Phase 3
Validates implementation without heavy dependencies
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

class SimpleValidator:
    def __init__(self):
        self.results = {}
        self.project_root = Path(__file__).parent
        
    def validate_file_structure(self):
        """Validate that all required files exist"""
        print("🔍 Validating File Structure...")
        
        required_files = [
            # Core app files
            'app/__init__.py',
            'app/events.py',
            'app/state_machine.py', 
            'app/mobility_predictor.py',
            
            # Component files
            'app/components/__init__.py',
            'app/components/interfaces.py',
            'app/components/trusted_authority.py',
            'app/components/gateway_node.py',
            'app/components/iot_device.py',
            
            # Shared modules
            'shared/__init__.py',
            'shared/config.py',
            'shared/crypto_utils.py',
            
            # Documentation
            'IMPLEMENTATION_TRACKER.md',
            'test_all_modules.py'
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"  ❌ {file_path}")
                
        self.results['file_structure'] = {
            'total_files': len(required_files),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'success_rate': len(existing_files) / len(required_files) * 100
        }
        
        return len(missing_files) == 0
    
    def validate_file_content(self):
        """Validate file content and structure"""
        print("\n📄 Validating File Content...")
        
        content_checks = {
            'app/events.py': [
                'class EventBus',
                'class Event',
                'async def publish',
                'async def subscribe'
            ],
            'app/state_machine.py': [
                'class ZKPASStateMachine',
                'class StateTransition',
                'async def transition_to',
                'def generate_mermaid_diagram'
            ],
            'app/mobility_predictor.py': [
                'class MobilityPredictor',
                'def train_model',
                'def predict_next_location',
                'def update_model'
            ],
            'shared/crypto_utils.py': [
                'def generate_keypair',
                'def create_zk_proof',
                'def verify_proof',
                'def hash_data'
            ]
        }
        
        validation_results = {}
        
        for file_path, required_content in content_checks.items():
            full_path = self.project_root / file_path
            if not full_path.exists():
                validation_results[file_path] = {'status': 'missing', 'found': 0, 'total': len(required_content)}
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_items = []
                missing_items = []
                
                for item in required_content:
                    if item in content:
                        found_items.append(item)
                        print(f"  ✅ {file_path}: {item}")
                    else:
                        missing_items.append(item)
                        print(f"  ❌ {file_path}: {item}")
                
                validation_results[file_path] = {
                    'status': 'checked',
                    'found': len(found_items),
                    'total': len(required_content),
                    'success_rate': len(found_items) / len(required_content) * 100,
                    'missing_items': missing_items
                }
                
            except Exception as e:
                validation_results[file_path] = {'status': 'error', 'error': str(e)}
                print(f"  ❌ {file_path}: Error reading file - {e}")
        
        self.results['content_validation'] = validation_results
        return validation_results
    
    def count_lines_of_code(self):
        """Count lines of code in implementation"""
        print("\n📊 Counting Lines of Code...")
        
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip venv and other non-source directories
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    python_files.append(os.path.join(root, file))
        
        total_lines = 0
        file_stats = {}
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    rel_path = os.path.relpath(file_path, self.project_root)
                    file_stats[rel_path] = lines
                    print(f"  📄 {rel_path}: {lines} lines")
            except Exception as e:
                print(f"  ❌ Error reading {file_path}: {e}")
        
        print(f"\n📈 Total Lines of Code: {total_lines}")
        
        self.results['code_metrics'] = {
            'total_lines': total_lines,
            'file_count': len(python_files),
            'file_stats': file_stats
        }
        
        return total_lines
    
    def validate_documentation(self):
        """Validate documentation exists and is comprehensive"""
        print("\n📚 Validating Documentation...")
        
        doc_files = [
            'IMPLEMENTATION_TRACKER.md',
            'README.md',
            'requirements.in'
        ]
        
        doc_results = {}
        
        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        word_count = len(content.split())
                        line_count = len(content.splitlines())
                        
                    doc_results[doc_file] = {
                        'exists': True,
                        'words': word_count,
                        'lines': line_count
                    }
                    print(f"  ✅ {doc_file}: {word_count} words, {line_count} lines")
                except Exception as e:
                    doc_results[doc_file] = {'exists': True, 'error': str(e)}
                    print(f"  ❌ {doc_file}: Error reading - {e}")
            else:
                doc_results[doc_file] = {'exists': False}
                print(f"  ❌ {doc_file}: Missing")
        
        self.results['documentation'] = doc_results
        return doc_results
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*60)
        print("📋 PHASE 3 VALIDATION SUMMARY")
        print("="*60)
        
        # File structure summary
        fs = self.results.get('file_structure', {})
        print(f"📁 File Structure: {fs.get('existing_files', 0)}/{fs.get('total_files', 0)} files ({fs.get('success_rate', 0):.1f}%)")
        
        # Content validation summary
        cv = self.results.get('content_validation', {})
        total_content_checks = sum(item.get('total', 0) for item in cv.values())
        passed_content_checks = sum(item.get('found', 0) for item in cv.values())
        if total_content_checks > 0:
            content_success_rate = passed_content_checks / total_content_checks * 100
            print(f"📄 Content Validation: {passed_content_checks}/{total_content_checks} checks ({content_success_rate:.1f}%)")
        
        # Code metrics summary
        cm = self.results.get('code_metrics', {})
        print(f"📊 Code Metrics: {cm.get('total_lines', 0)} lines in {cm.get('file_count', 0)} files")
        
        # Documentation summary
        doc = self.results.get('documentation', {})
        doc_exists = sum(1 for item in doc.values() if item.get('exists', False))
        print(f"📚 Documentation: {doc_exists}/{len(doc)} files")
        
        # Overall assessment
        overall_score = (
            fs.get('success_rate', 0) * 0.3 +
            (content_success_rate if total_content_checks > 0 else 100) * 0.4 +
            (100 if cm.get('total_lines', 0) > 1000 else cm.get('total_lines', 0) / 10) * 0.2 +
            (doc_exists / len(doc) * 100) * 0.1
        )
        
        print(f"\n🎯 Overall Phase 3 Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            status = "✅ READY FOR PHASE 4"
            print("🚀 Phase 3 implementation is solid - ready to proceed!")
        elif overall_score >= 60:
            status = "⚠️ MINOR ISSUES"
            print("⚠️ Some minor issues found - review and fix before Phase 4")
        else:
            status = "❌ MAJOR ISSUES"
            print("❌ Major issues found - significant work needed before Phase 4")
        
        # Save detailed report
        self.results['summary'] = {
            'timestamp': timestamp,
            'overall_score': overall_score,
            'status': status,
            'ready_for_phase4': overall_score >= 80
        }
        
        report_file = f"validation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Detailed report saved to: {report_file}")
        return overall_score >= 80

def main():
    print("🔍 ZKPAS Phase 3 Simple Validation")
    print("=" * 50)
    
    validator = SimpleValidator()
    
    # Run validation steps
    validator.validate_file_structure()
    validator.validate_file_content()
    validator.count_lines_of_code()
    validator.validate_documentation()
    
    # Generate final report
    ready_for_phase4 = validator.generate_report()
    
    return 0 if ready_for_phase4 else 1

if __name__ == "__main__":
    sys.exit(main())
