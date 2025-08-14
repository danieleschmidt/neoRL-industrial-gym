"""Real-time quality monitoring with live dashboard and notifications."""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import websockets
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor

from .quality_metrics import QualityMetrics, QualityThresholds
from .progressive_monitor import QualityEvent

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Rule for triggering quality alerts."""
    name: str
    condition: str  # Python expression to evaluate
    severity: str  # "low", "medium", "high", "critical"
    message: str
    cooldown_seconds: int = 300  # 5 minutes default
    last_triggered: float = 0.0
    enabled: bool = True


@dataclass
class QualityAlert:
    """Quality alert notification."""
    id: str
    timestamp: float
    rule_name: str
    severity: str
    message: str
    metrics: Optional[QualityMetrics] = None
    context: Dict[str, Any] = None


class RealTimeQualityMonitor:
    """
    Real-time quality monitoring with live dashboard and alert system.
    
    Features:
    - Live quality metrics dashboard
    - WebSocket-based real-time updates  
    - Configurable alert rules
    - Notification system (email, Slack, etc.)
    - Quality trend analysis
    - Performance bottleneck detection
    """
    
    def __init__(
        self,
        project_root: Path,
        dashboard_port: int = 8080,
        websocket_port: int = 8081,
        enable_dashboard: bool = True,
        enable_notifications: bool = True
    ):
        self.project_root = Path(project_root)
        self.dashboard_port = dashboard_port
        self.websocket_port = websocket_port
        self.enable_dashboard = enable_dashboard
        self.enable_notifications = enable_notifications
        
        # State
        self.is_running = False
        self.current_metrics = QualityMetrics()
        self.metrics_history: List[QualityMetrics] = []
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.event_queue = queue.Queue()
        
        # Alert system
        self.alert_rules: List[AlertRule] = self._create_default_alert_rules()
        self.notification_handlers: List[Callable[[QualityAlert], None]] = []
        
        # Async components
        self.websocket_clients: set = set()
        self.dashboard_server: Optional[http.server.HTTPServer] = None
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="rt-monitor")
        
    def start(self) -> None:
        """Start real-time monitoring."""
        if self.is_running:
            logger.warning("Real-time monitor is already running")
            return
            
        logger.info("Starting real-time quality monitor...")
        self.is_running = True
        
        # Start dashboard server
        if self.enable_dashboard:
            self.executor.submit(self._start_dashboard_server)
            
        # Start WebSocket server  
        self.executor.submit(self._start_websocket_server)
        
        # Start alert processing
        self.executor.submit(self._process_alerts)
        
        logger.info(f"Real-time monitor started (dashboard: http://localhost:{self.dashboard_port})")
        
    def stop(self) -> None:
        """Stop real-time monitoring."""
        if not self.is_running:
            return
            
        logger.info("Stopping real-time quality monitor...")
        self.is_running = False
        
        # Close WebSocket connections
        for client in self.websocket_clients:
            try:
                asyncio.create_task(client.close())
            except:
                pass
                
        # Stop servers
        if self.dashboard_server:
            self.dashboard_server.shutdown()
            
        self.executor.shutdown(wait=True)
        logger.info("Real-time monitor stopped")
        
    def update_metrics(self, metrics: QualityMetrics) -> None:
        """Update current quality metrics."""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-800:]
            
        # Update trend
        if len(self.metrics_history) > 1:
            self.current_metrics.update_trend(self.metrics_history[-2])
            
        # Check alert rules
        self._check_alert_rules(metrics)
        
        # Broadcast update
        self._broadcast_metrics_update()
        
    def handle_quality_event(self, event: QualityEvent) -> None:
        """Handle quality event from progressive monitor."""
        self.event_queue.put(event)
        
        # Update metrics if available
        if event.metrics:
            self.update_metrics(event.metrics)
            
        # Check for special events
        if event.event_type == "threshold_violation":
            self._handle_threshold_violation(event)
            
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
        
    def add_notification_handler(self, handler: Callable[[QualityAlert], None]) -> None:
        """Add notification handler."""
        self.notification_handlers.append(handler)
        
    def _create_default_alert_rules(self) -> List[AlertRule]:
        """Create default alert rules."""
        return [
            AlertRule(
                name="Critical Security Vulnerability",
                condition="metrics.high_severity_vulnerabilities > 0",
                severity="critical",
                message="Critical security vulnerability detected: "
                       "{metrics.high_severity_vulnerabilities} high-severity issues",
                cooldown_seconds=60
            ),
            AlertRule(
                name="Low Test Coverage",
                condition="metrics.code_coverage < 70.0",
                severity="high", 
                message="Test coverage dropped below 70%: {metrics.code_coverage:.1f}%",
                cooldown_seconds=300
            ),
            AlertRule(
                name="Build Failure",
                condition="metrics.build_success_rate < 95.0",
                severity="high",
                message="Build success rate below 95%: {metrics.build_success_rate:.1f}%",
                cooldown_seconds=180
            ),
            AlertRule(
                name="Performance Degradation", 
                condition="metrics.performance_score < 60.0",
                severity="medium",
                message="Performance score dropped below 60: {metrics.performance_score:.1f}",
                cooldown_seconds=600
            ),
            AlertRule(
                name="Quality Score Decline",
                condition="metrics.overall_score < 70.0",
                severity="medium",
                message="Overall quality score dropped below 70: {metrics.overall_score:.1f}",
                cooldown_seconds=900
            ),
            AlertRule(
                name="Technical Debt Accumulation",
                condition="metrics.technical_debt_hours > 80.0",
                severity="low",
                message="Technical debt exceeded 80 hours: {metrics.technical_debt_hours:.1f}",
                cooldown_seconds=1800
            )
        ]
        
    def _check_alert_rules(self, metrics: QualityMetrics) -> None:
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            # Check cooldown
            if current_time - rule.last_triggered < rule.cooldown_seconds:
                continue
                
            try:
                # Evaluate condition
                if eval(rule.condition, {"metrics": metrics}):
                    # Create alert
                    alert = QualityAlert(
                        id=f"{rule.name}_{int(current_time)}",
                        timestamp=current_time,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.message.format(metrics=metrics),
                        metrics=metrics,
                        context={"rule": rule.name, "condition": rule.condition}
                    )
                    
                    self._trigger_alert(alert)
                    rule.last_triggered = current_time
                    
            except Exception as e:
                logger.warning(f"Alert rule '{rule.name}' evaluation failed: {e}")
                
    def _trigger_alert(self, alert: QualityAlert) -> None:
        """Trigger a quality alert."""
        self.active_alerts[alert.id] = alert
        
        logger.warning(f"QUALITY ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Send notifications
        if self.enable_notifications:
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Notification handler failed: {e}")
                    
        # Broadcast to WebSocket clients
        self._broadcast_alert(alert)
        
    def _handle_threshold_violation(self, event: QualityEvent) -> None:
        """Handle threshold violation event."""
        violation = event.details
        severity = violation.get("severity", "medium")
        
        alert = QualityAlert(
            id=f"threshold_{int(time.time())}",
            timestamp=time.time(),
            rule_name="Threshold Violation",
            severity=severity,
            message=f"Quality threshold violated: {violation['type']} "
                   f"({violation['current']:.1f} vs {violation['threshold']:.1f})",
            context=violation
        )
        
        self._trigger_alert(alert)
        
    def _broadcast_metrics_update(self) -> None:
        """Broadcast metrics update to WebSocket clients."""
        if not self.websocket_clients:
            return
            
        message = {
            "type": "metrics_update",
            "data": {
                "metrics": asdict(self.current_metrics),
                "timestamp": time.time(),
                "trend_data": [asdict(m) for m in self.metrics_history[-50:]]  # Last 50 points
            }
        }
        
        self._broadcast_websocket_message(message)
        
    def _broadcast_alert(self, alert: QualityAlert) -> None:
        """Broadcast alert to WebSocket clients."""
        message = {
            "type": "alert",
            "data": asdict(alert)
        }
        
        self._broadcast_websocket_message(message)
        
    def _broadcast_websocket_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all WebSocket clients."""
        if not self.websocket_clients:
            return
            
        message_str = json.dumps(message, default=str)
        
        # Remove closed connections
        closed_clients = set()
        for client in self.websocket_clients:
            try:
                if client.closed:
                    closed_clients.add(client)
                else:
                    asyncio.create_task(client.send(message_str))
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                closed_clients.add(client)
                
        self.websocket_clients -= closed_clients
        
    def _start_dashboard_server(self) -> None:
        """Start the dashboard HTTP server."""
        try:
            class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, monitor=None, **kwargs):
                    self.monitor = monitor
                    super().__init__(*args, **kwargs)
                    
                def do_GET(self):
                    if self.path == "/" or self.path == "/dashboard":
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(self._generate_dashboard_html().encode())
                    elif self.path == "/api/metrics":
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        data = {
                            "metrics": asdict(self.monitor.current_metrics),
                            "history": [asdict(m) for m in self.monitor.metrics_history[-100:]],
                            "alerts": [asdict(a) for a in self.monitor.active_alerts.values()]
                        }
                        self.wfile.write(json.dumps(data, default=str).encode())
                    else:
                        super().do_GET()
                        
                def _generate_dashboard_html(self) -> str:
                    """Generate dashboard HTML."""
                    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>neoRL Industrial - Quality Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-value {{ font-weight: bold; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert-critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .alert-high {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
        .alert-medium {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
        .status-good {{ color: #4caf50; }}
        .status-warning {{ color: #ff9800; }}
        .status-error {{ color: #f44336; }}
        h1 {{ color: #333; text-align: center; }}
        .chart-container {{ position: relative; height: 300px; }}
    </style>
</head>
<body>
    <h1>🏭 neoRL Industrial - Quality Dashboard</h1>
    <div class="dashboard">
        <div class="card">
            <h3>📊 Current Metrics</h3>
            <div id="metrics-display">
                <div class="metric">
                    <span>Overall Score:</span>
                    <span class="metric-value" id="overall-score">--</span>
                </div>
                <div class="metric">
                    <span>Code Coverage:</span>
                    <span class="metric-value" id="coverage">--</span>
                </div>
                <div class="metric">
                    <span>Security Score:</span>
                    <span class="metric-value" id="security">--</span>
                </div>
                <div class="metric">
                    <span>Performance Score:</span>
                    <span class="metric-value" id="performance">--</span>
                </div>
                <div class="metric">
                    <span>Test Pass Rate:</span>
                    <span class="metric-value" id="test-rate">--</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Quality Trend</h3>
            <div class="chart-container">
                <canvas id="trendChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h3>🚨 Active Alerts</h3>
            <div id="alerts-display">
                <p>No active alerts</p>
            </div>
        </div>
        
        <div class="card">
            <h3>⚡ Real-time Status</h3>
            <div id="status-display">
                <div class="metric">
                    <span>Connection:</span>
                    <span class="metric-value" id="connection-status">Connecting...</span>
                </div>
                <div class="metric">
                    <span>Last Update:</span>
                    <span class="metric-value" id="last-update">--</span>
                </div>
                <div class="metric">
                    <span>Quality Trend:</span>
                    <span class="metric-value" id="quality-trend">--</span>
                </div>
                <div class="metric">
                    <span>Risk Level:</span>
                    <span class="metric-value" id="risk-level">--</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket('ws://localhost:{self.websocket_port}');
        let trendChart;
        
        ws.onopen = function() {{
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('connection-status').className = 'metric-value status-good';
        }};
        
        ws.onclose = function() {{
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('connection-status').className = 'metric-value status-error';
        }};
        
        ws.onmessage = function(event) {{
            const message = JSON.parse(event.data);
            
            if (message.type === 'metrics_update') {{
                updateMetrics(message.data.metrics);
                updateTrendChart(message.data.trend_data);
            }} else if (message.type === 'alert') {{
                addAlert(message.data);
            }}
        }};
        
        function updateMetrics(metrics) {{
            document.getElementById('overall-score').textContent = metrics.overall_score.toFixed(1);
            document.getElementById('coverage').textContent = metrics.code_coverage.toFixed(1) + '%';
            document.getElementById('security').textContent = metrics.security_score.toFixed(1);
            document.getElementById('performance').textContent = metrics.performance_score.toFixed(1);
            document.getElementById('test-rate').textContent = metrics.test_pass_rate.toFixed(1) + '%';
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            document.getElementById('quality-trend').textContent = metrics.quality_trend;
            document.getElementById('risk-level').textContent = metrics.risk_level;
            
            // Update colors based on values
            updateMetricColor('overall-score', metrics.overall_score, 75, 50);
            updateMetricColor('coverage', metrics.code_coverage, 80, 60);
            updateMetricColor('security', metrics.security_score, 85, 70);
        }}
        
        function updateMetricColor(elementId, value, goodThreshold, warningThreshold) {{
            const element = document.getElementById(elementId);
            if (value >= goodThreshold) {{
                element.className = 'metric-value status-good';
            }} else if (value >= warningThreshold) {{
                element.className = 'metric-value status-warning';
            }} else {{
                element.className = 'metric-value status-error';
            }}
        }}
        
        function updateTrendChart(trendData) {{
            if (!trendChart) {{
                const ctx = document.getElementById('trendChart').getContext('2d');
                trendChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [{{
                            label: 'Overall Quality Score',
                            data: [],
                            borderColor: '#4caf50',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{ min: 0, max: 100 }}
                        }}
                    }}
                }});
            }}
            
            // Update chart data
            const labels = trendData.map((_, i) => i);
            const scores = trendData.map(d => d.overall_score);
            
            trendChart.data.labels = labels;
            trendChart.data.datasets[0].data = scores;
            trendChart.update();
        }}
        
        function addAlert(alert) {{
            const alertsDiv = document.getElementById('alerts-display');
            const alertElement = document.createElement('div');
            alertElement.className = `alert alert-${{alert.severity}}`;
            alertElement.innerHTML = `
                <strong>${{alert.rule_name}}</strong><br>
                ${{alert.message}}<br>
                <small>${{new Date(alert.timestamp * 1000).toLocaleString()}}</small>
            `;
            
            if (alertsDiv.children.length === 1 && alertsDiv.children[0].tagName === 'P') {{
                alertsDiv.innerHTML = '';
            }}
            
            alertsDiv.appendChild(alertElement);
        }}
        
        // Initial data load
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {{
                updateMetrics(data.metrics);
                updateTrendChart(data.history);
                
                // Load existing alerts
                data.alerts.forEach(alert => addAlert(alert));
            }});
    </script>
</body>
</html>"""
            
            handler = lambda *args: DashboardHandler(*args, monitor=self)
            
            with socketserver.TCPServer(("", self.dashboard_port), handler) as httpd:
                self.dashboard_server = httpd
                logger.info(f"Dashboard server started on port {self.dashboard_port}")
                
                while self.is_running:
                    httpd.handle_request()
                    
        except Exception as e:
            logger.error(f"Dashboard server failed: {e}")
            
    def _start_websocket_server(self) -> None:
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected from {websocket.remote_address}")
            
            try:
                # Send initial data
                initial_data = {
                    "type": "initial_data",
                    "data": {
                        "metrics": asdict(self.current_metrics),
                        "history": [asdict(m) for m in self.metrics_history[-50:]],
                        "alerts": [asdict(a) for a in self.active_alerts.values()]
                    }
                }
                await websocket.send(json.dumps(initial_data, default=str))
                
                # Keep connection alive
                while self.is_running:
                    await asyncio.sleep(1)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.warning(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
                logger.info("WebSocket client disconnected")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(
                handle_client, 
                "localhost", 
                self.websocket_port
            )
            
            logger.info(f"WebSocket server started on port {self.websocket_port}")
            loop.run_until_complete(start_server)
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket server failed: {e}")
            
    def _process_alerts(self) -> None:
        """Process alert queue."""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                # Process different event types
                if event.event_type == "file_changed":
                    logger.debug(f"File changed: {event.file_path}")
                elif event.event_type == "quality_check":
                    logger.info(f"Quality check completed (score: {event.metrics.overall_score:.1f})")
                elif event.event_type == "threshold_violation":
                    logger.warning(f"Threshold violation: {event.details}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get current dashboard status."""
        return {
            "is_running": self.is_running,
            "dashboard_port": self.dashboard_port,
            "websocket_port": self.websocket_port,
            "connected_clients": len(self.websocket_clients),
            "active_alerts": len(self.active_alerts),
            "metrics_history_size": len(self.metrics_history),
            "current_metrics": asdict(self.current_metrics)
        }