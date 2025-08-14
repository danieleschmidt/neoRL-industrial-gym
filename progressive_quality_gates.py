#!/usr/bin/env python3
"""
Progressive Quality Gates - Advanced continuous quality monitoring system.

This system provides real-time quality monitoring, adaptive thresholds,
and comprehensive quality dashboards for the neoRL Industrial project.

Features:
- Real-time quality monitoring with file watching
- Adaptive thresholds based on project evolution
- Live quality dashboard with WebSocket updates
- Intelligent alert system with configurable rules
- Progressive quality gate execution by development phase
- Comprehensive quality metrics and trend analysis
- Self-healing quality violations
- Production-ready quality enforcement
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neorl_industrial.quality_gates import (
    ProgressiveQualityMonitor,
    RealTimeQualityMonitor, 
    AdaptiveQualityGates,
    QualityThresholds,
    QualityMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('progressive_quality_gates.log')
    ]
)
logger = logging.getLogger(__name__)


class ProgressiveQualitySystem:
    """
    Main Progressive Quality Gates system orchestrator.
    
    Coordinates all quality monitoring components:
    - Progressive Monitor (file watching + quality checks)
    - Real-time Monitor (dashboard + alerts) 
    - Adaptive Gates (self-tuning thresholds)
    """
    
    def __init__(self, project_root: Path, config: Optional[Dict[str, Any]] = None):
        self.project_root = Path(project_root)
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.progressive_monitor: Optional[ProgressiveQualityMonitor] = None
        self.realtime_monitor: Optional[RealTimeQualityMonitor] = None
        self.adaptive_gates: Optional[AdaptiveQualityGates] = None
        
        # System state
        self.is_running = False
        self.start_time = 0.0
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "progressive_monitor": {
                "check_interval": 5.0,
                "enable_real_time": True
            },
            "realtime_monitor": {
                "dashboard_port": 8080,
                "websocket_port": 8081,
                "enable_dashboard": True,
                "enable_notifications": True
            },
            "adaptive_gates": {
                "adaptation_interval": 3600.0,  # 1 hour
                "history_window": 100,
                "enable_adaptation": True
            },
            "quality_thresholds": {
                "min_code_coverage": 80.0,
                "min_test_pass_rate": 95.0,
                "min_security_score": 85.0,
                "min_performance_score": 70.0,
                "min_overall_score": 75.0
            }
        }
        
    def initialize(self) -> None:
        """Initialize all quality monitoring components."""
        logger.info("Initializing Progressive Quality Gates system...")
        
        # Create quality thresholds
        thresholds_config = self.config.get("quality_thresholds", {})
        thresholds = QualityThresholds(
            min_code_coverage=thresholds_config.get("min_code_coverage", 80.0),
            min_test_pass_rate=thresholds_config.get("min_test_pass_rate", 95.0),
            min_security_score=thresholds_config.get("min_security_score", 85.0),
            min_performance_score=thresholds_config.get("min_performance_score", 70.0),
            min_overall_score=thresholds_config.get("min_overall_score", 75.0)
        )
        
        # Initialize adaptive gates
        adaptive_config = self.config.get("adaptive_gates", {})
        if adaptive_config.get("enable_adaptation", True):
            self.adaptive_gates = AdaptiveQualityGates(
                project_root=self.project_root,
                initial_thresholds=thresholds,
                adaptation_interval=adaptive_config.get("adaptation_interval", 3600.0),
                history_window=adaptive_config.get("history_window", 100)
            )
            
            # Try to load previously adapted thresholds
            if self.adaptive_gates.load_thresholds():
                thresholds = self.adaptive_gates.current_thresholds
                logger.info("Loaded previously adapted quality thresholds")
        
        # Initialize progressive monitor
        prog_config = self.config.get("progressive_monitor", {})
        self.progressive_monitor = ProgressiveQualityMonitor(
            project_root=self.project_root,
            thresholds=thresholds,
            check_interval=prog_config.get("check_interval", 5.0),
            enable_real_time=prog_config.get("enable_real_time", True)
        )
        
        # Initialize real-time monitor
        rt_config = self.config.get("realtime_monitor", {})
        self.realtime_monitor = RealTimeQualityMonitor(
            project_root=self.project_root,
            dashboard_port=rt_config.get("dashboard_port", 8080),
            websocket_port=rt_config.get("websocket_port", 8081),
            enable_dashboard=rt_config.get("enable_dashboard", True),
            enable_notifications=rt_config.get("enable_notifications", True)
        )
        
        # Connect components
        self._connect_components()
        
        logger.info("Progressive Quality Gates system initialized")
        
    def _connect_components(self) -> None:
        """Connect monitoring components together."""
        if not (self.progressive_monitor and self.realtime_monitor):
            return
            
        # Connect progressive monitor to real-time monitor
        self.progressive_monitor.add_event_handler(self.realtime_monitor.handle_quality_event)
        
        # Connect adaptive gates if enabled
        if self.adaptive_gates:
            def handle_metrics_for_adaptation(event):
                if event.metrics:
                    self.adaptive_gates.add_metrics(event.metrics)
                    
            self.progressive_monitor.add_event_handler(handle_metrics_for_adaptation)
            
        # Add custom notification handlers
        self.realtime_monitor.add_notification_handler(self._handle_critical_alert)
        
        logger.info("Quality monitoring components connected")
        
    def _handle_critical_alert(self, alert) -> None:
        """Handle critical quality alerts."""
        if alert.severity == "critical":
            logger.critical(f"CRITICAL QUALITY ALERT: {alert.message}")
            
            # Could integrate with external systems:
            # - Send Slack notification
            # - Create JIRA ticket
            # - Send email to team
            # - Trigger CI/CD pipeline halt
            
    def start(self) -> None:
        """Start the Progressive Quality Gates system."""
        if self.is_running:
            logger.warning("System is already running")
            return
            
        logger.info("Starting Progressive Quality Gates system...")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            # Start all components
            if self.progressive_monitor:
                self.progressive_monitor.start_monitoring()
                
            if self.realtime_monitor:
                self.realtime_monitor.start()
                
            logger.info("🚀 Progressive Quality Gates system is running!")
            logger.info(f"📊 Dashboard available at: http://localhost:{self.config['realtime_monitor']['dashboard_port']}")
            logger.info(f"⚡ WebSocket server on port: {self.config['realtime_monitor']['websocket_port']}")
            
            if self.adaptive_gates:
                logger.info("🧠 Adaptive thresholds enabled")
                
            # Print initial status
            self._print_status()
            
        except Exception as e:
            logger.error(f"Failed to start Progressive Quality Gates system: {e}")
            self.stop()
            raise
            
    def stop(self) -> None:
        """Stop the Progressive Quality Gates system."""
        if not self.is_running:
            return
            
        logger.info("Stopping Progressive Quality Gates system...")
        
        # Stop all components
        if self.progressive_monitor:
            self.progressive_monitor.stop_monitoring()
            
        if self.realtime_monitor:
            self.realtime_monitor.stop()
            
        # Save final state
        if self.adaptive_gates:
            self.adaptive_gates._save_thresholds()
            
        self.is_running = False
        
        runtime = time.time() - self.start_time
        logger.info(f"Progressive Quality Gates system stopped (runtime: {runtime:.1f}s)")
        
    def _print_status(self) -> None:
        """Print current system status."""
        print("\n" + "="*80)
        print("🏭 PROGRESSIVE QUALITY GATES - SYSTEM STATUS")
        print("="*80)
        
        if self.progressive_monitor:
            status = self.progressive_monitor.get_quality_status()
            print(f"📁 Files Tracked: {status['files_tracked']}")
            print(f"📊 Current Quality Score: {status['current_metrics'].overall_score:.1f}")
            print(f"🔄 Project Phase: {status['project_phase']}")
            print(f"⚠️  Active Violations: {len(status['violations'])}")
            
        if self.realtime_monitor:
            dashboard_status = self.realtime_monitor.get_dashboard_status()
            print(f"🌐 Dashboard Port: {dashboard_status['dashboard_port']}")
            print(f"🔗 Connected Clients: {dashboard_status['connected_clients']}")
            print(f"🚨 Active Alerts: {dashboard_status['active_alerts']}")
            
        if self.adaptive_gates:
            adaptation_report = self.adaptive_gates.get_adaptation_report()
            print(f"🧠 Adaptive Rules: {len(adaptation_report['rules_status'])} active")
            print(f"📈 Total Adaptations: {adaptation_report['adaptation_statistics']['total_adaptations']}")
            
        print("="*80)
        print("System is actively monitoring quality metrics...")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")
        
    def run_quality_check(self) -> Dict[str, Any]:
        """Run an immediate quality check."""
        if not self.progressive_monitor:
            raise RuntimeError("System not initialized")
            
        logger.info("Running immediate quality check...")
        self.progressive_monitor._run_quality_check(trigger="manual")
        
        return self.progressive_monitor.get_quality_status()
        
    def export_report(self, output_path: Path) -> None:
        """Export comprehensive quality report."""
        if not self.progressive_monitor:
            raise RuntimeError("System not initialized")
            
        logger.info(f"Exporting quality report to {output_path}")
        self.progressive_monitor.export_quality_report(output_path)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "system": {
                "is_running": self.is_running,
                "start_time": self.start_time,
                "runtime": time.time() - self.start_time if self.is_running else 0,
                "project_root": str(self.project_root)
            },
            "components": {}
        }
        
        if self.progressive_monitor:
            status["components"]["progressive_monitor"] = self.progressive_monitor.get_quality_status()
            
        if self.realtime_monitor:
            status["components"]["realtime_monitor"] = self.realtime_monitor.get_dashboard_status()
            
        if self.adaptive_gates:
            status["components"]["adaptive_gates"] = self.adaptive_gates.get_adaptation_report()
            
        return status


def signal_handler(signum, frame, quality_system):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    quality_system.stop()
    sys.exit(0)


def main():
    """Main entry point for Progressive Quality Gates."""
    parser = argparse.ArgumentParser(
        description="Progressive Quality Gates - Advanced continuous quality monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python progressive_quality_gates.py                    # Start with defaults
  python progressive_quality_gates.py --dashboard-port 8000  # Custom dashboard port
  python progressive_quality_gates.py --check-only       # Run single quality check
  python progressive_quality_gates.py --export-report quality_report.json
        """
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path (JSON format)"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8080,
        help="Dashboard HTTP port (default: 8080)"
    )
    
    parser.add_argument(
        "--websocket-port", 
        type=int,
        default=8081,
        help="WebSocket server port (default: 8081)"
    )
    
    parser.add_argument(
        "--check-interval",
        type=float,
        default=5.0,
        help="File check interval in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable web dashboard"
    )
    
    parser.add_argument(
        "--no-adaptation",
        action="store_true", 
        help="Disable adaptive thresholds"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run single quality check and exit"
    )
    
    parser.add_argument(
        "--export-report",
        type=Path,
        help="Export quality report to file and exit"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Load configuration
    config = None
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = {}
        
    # Override config with command line arguments
    if not config.get("realtime_monitor"):
        config["realtime_monitor"] = {}
    if not config.get("progressive_monitor"):
        config["progressive_monitor"] = {}
    if not config.get("adaptive_gates"):
        config["adaptive_gates"] = {}
        
    config["realtime_monitor"]["dashboard_port"] = args.dashboard_port
    config["realtime_monitor"]["websocket_port"] = args.websocket_port
    config["realtime_monitor"]["enable_dashboard"] = not args.no_dashboard
    config["progressive_monitor"]["check_interval"] = args.check_interval
    config["adaptive_gates"]["enable_adaptation"] = not args.no_adaptation
    
    # Initialize system
    quality_system = ProgressiveQualitySystem(args.project_root, config)
    
    try:
        quality_system.initialize()
        
        if args.check_only:
            # Run single quality check
            status = quality_system.run_quality_check()
            print(json.dumps(status, indent=2, default=str))
            return
            
        if args.export_report:
            # Export report and exit
            quality_system.export_report(args.export_report)
            print(f"Quality report exported to {args.export_report}")
            return
            
        # Start continuous monitoring
        quality_system.start()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, quality_system))
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler(s, f, quality_system))
        
        # Keep running until interrupted
        try:
            while quality_system.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
            
    except Exception as e:
        logger.error(f"Progressive Quality Gates failed: {e}")
        return 1
    finally:
        quality_system.stop()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())