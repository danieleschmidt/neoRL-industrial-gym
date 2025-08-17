#!/usr/bin/env python3
"""Test research capabilities without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_module_imports():
    """Test that all research modules can be imported."""
    
    print("Testing module imports...")
    
    try:
        # Test basic imports
        import neorl_industrial
        print("✅ neorl_industrial imported successfully")
        
        # Test research module import
        from neorl_industrial.research import (
            NovelOfflineRLAlgorithms,
            IndustrialMetaLearning,
            ContinualIndustrialRL,
            AutoMLForIndustrialRL,
            IndustrialFoundationModel,
            ResearchAccelerator,
            DistributedResearchFramework
        )
        print("✅ All research modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_algorithm_listing():
    """Test novel algorithm listing without instantiation."""
    
    print("\nTesting algorithm listing...")
    
    try:
        from neorl_industrial.research import NovelOfflineRLAlgorithms
        
        algorithms = NovelOfflineRLAlgorithms.list_algorithms()
        
        expected_algorithms = ["hierarchical_cql", "distributional_crl", "adaptive_orl"]
        
        for alg in expected_algorithms:
            if alg not in algorithms:
                print(f"❌ Missing algorithm: {alg}")
                return False
            print(f"✅ Algorithm available: {alg} - {algorithms[alg]}")
        
        print(f"✅ All {len(expected_algorithms)} algorithms available")
        return True
        
    except Exception as e:
        print(f"❌ Algorithm listing failed: {e}")
        return False


def test_meta_learning_initialization():
    """Test meta-learning initialization."""
    
    print("\nTesting meta-learning initialization...")
    
    try:
        from neorl_industrial.research import IndustrialMetaLearning
        
        meta_learner = IndustrialMetaLearning(
            base_environments=["ChemicalReactor-v0", "PowerGrid-v0"],
            state_dim=20,
            action_dim=6
        )
        
        print(f"✅ Meta-learner created with {len(meta_learner.base_environments)} environments")
        print(f"✅ State dim: {meta_learner.state_dim}, Action dim: {meta_learner.action_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Meta-learning initialization failed: {e}")
        return False


def test_continual_learning_creation():
    """Test continual learning agent creation."""
    
    print("\nTesting continual learning creation...")
    
    try:
        from neorl_industrial.research import ContinualIndustrialRL
        
        # Test different continual methods
        methods = ["ewc", "progressive", "replay"]
        
        for method in methods:
            try:
                agent = ContinualIndustrialRL(
                    state_dim=10,
                    action_dim=3,
                    continual_method=method
                )
                print(f"✅ {method.upper()} agent created successfully")
                
            except Exception as e:
                print(f"❌ {method.upper()} agent creation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Continual learning test failed: {e}")
        return False


def test_automl_initialization():
    """Test AutoML framework initialization."""
    
    print("\nTesting AutoML initialization...")
    
    try:
        from neorl_industrial.research import AutoMLForIndustrialRL
        
        automl = AutoMLForIndustrialRL(
            state_dim=15,
            action_dim=4,
            search_budget=5,
            parallel_evaluations=1
        )
        
        print(f"✅ AutoML framework created")
        print(f"✅ Search budget: {automl.search_budget}")
        print(f"✅ State dim: {automl.state_dim}, Action dim: {automl.action_dim}")
        
        # Test search space
        search_space = automl.search_space
        architectures = search_space.get_all_architectures()
        print(f"✅ Search space contains {len(architectures)} architectures")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoML initialization failed: {e}")
        return False


def test_foundation_model_creation():
    """Test foundation model creation."""
    
    print("\nTesting foundation model creation...")
    
    try:
        from neorl_industrial.research import IndustrialFoundationModel
        from neorl_industrial.research.foundation_models import FoundationModelConfig
        
        # Test with default config
        foundation_model = IndustrialFoundationModel(
            state_dim=12,
            action_dim=4
        )
        
        print(f"✅ Foundation model created with default config")
        print(f"✅ State dim: {foundation_model.state_dim}")
        print(f"✅ Action dim: {foundation_model.action_dim}")
        print(f"✅ Pre-training objectives: {foundation_model.pre_training_objectives}")
        
        # Test custom config
        custom_config = FoundationModelConfig(
            embed_dim=256,
            num_layers=4,
            num_heads=4
        )
        
        custom_model = IndustrialFoundationModel(
            state_dim=8,
            action_dim=2,
            config=custom_config
        )
        
        print(f"✅ Custom foundation model created")
        print(f"✅ Embed dim: {custom_model.config.embed_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Foundation model test failed: {e}")
        return False


def test_research_accelerator():
    """Test research accelerator creation."""
    
    print("\nTesting research accelerator...")
    
    try:
        from neorl_industrial.research import ResearchAccelerator
        
        accelerator = ResearchAccelerator(
            experiment_dir="test_experiments",
            max_parallel_experiments=2,
            use_quality_gates=False  # Disable for simpler testing
        )
        
        print(f"✅ Research accelerator created")
        print(f"✅ Experiment dir: {accelerator.experiment_dir}")
        print(f"✅ Max parallel experiments: {accelerator.max_parallel_experiments}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research accelerator test failed: {e}")
        return False


def test_distributed_framework():
    """Test distributed framework creation."""
    
    print("\nTesting distributed framework...")
    
    try:
        from neorl_industrial.research import DistributedResearchFramework
        
        distributed_framework = DistributedResearchFramework(
            num_workers=1,  # Single worker for testing
            devices_per_worker=1,
            use_async_updates=False
        )
        
        print(f"✅ Distributed framework created")
        print(f"✅ Num workers: {distributed_framework.num_workers}")
        print(f"✅ Devices per worker: {distributed_framework.devices_per_worker}")
        
        # Cleanup
        distributed_framework.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed framework test failed: {e}")
        return False


def test_package_structure():
    """Test package structure and exports."""
    
    print("\nTesting package structure...")
    
    try:
        import neorl_industrial
        
        # Check main exports
        expected_exports = [
            "DatasetQuality", "SafetyConstraint", "SafetyMetrics",
            "IndustrialEnv", "ChemicalReactorEnv", "PowerGridEnv", "RobotAssemblyEnv",
            "OfflineAgent", "CQLAgent", "IQLAgent", "TD3BCAgent",
            "make", "evaluate_with_safety"
        ]
        
        for export in expected_exports:
            if hasattr(neorl_industrial, export):
                print(f"✅ Export available: {export}")
            else:
                print(f"⚠️ Missing export: {export}")
        
        # Check research exports
        from neorl_industrial import research
        research_exports = [
            "NovelOfflineRLAlgorithms",
            "IndustrialMetaLearning",
            "ContinualIndustrialRL",
            "AutoMLForIndustrialRL",
            "IndustrialFoundationModel",
            "ResearchAccelerator",
            "DistributedResearchFramework"
        ]
        
        for export in research_exports:
            if hasattr(research, export):
                print(f"✅ Research export available: {export}")
            else:
                print(f"❌ Missing research export: {export}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Package structure test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    
    print("🔬 NEORL-INDUSTRIAL RESEARCH CAPABILITIES TESTING")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Algorithm Listing", test_algorithm_listing),
        ("Meta-Learning", test_meta_learning_initialization),
        ("Continual Learning", test_continual_learning_creation),
        ("AutoML Framework", test_automl_initialization),
        ("Foundation Models", test_foundation_model_creation),
        ("Research Accelerator", test_research_accelerator),
        ("Distributed Framework", test_distributed_framework),
        ("Package Structure", test_package_structure),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            if success:
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Final report
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    success_rate = passed_tests / total_tests
    
    print(f"✅ Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Research capabilities are ready.")
        exit_code = 0
    else:
        failed_tests = total_tests - passed_tests
        print(f"❌ Failed: {failed_tests} tests")
        print("🔧 Some capabilities need attention before deployment.")
        exit_code = 1
    
    print("\n🚀 Research framework validation complete!")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)