#!/usr/bin/env python3
"""Automation orchestrator for neoRL-industrial-gym.

This script orchestrates various automation tasks based on schedules and triggers:
- Daily operational tasks
- Weekly analysis and reporting
- Monthly strategic reviews
- Quarterly compliance audits
- Event-driven automations
"""

import argparse
import asyncio
import json
import logging
import os
import schedule
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import subprocess
import yaml
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation_orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AutomationTask:
    """Represents an automation task."""
    name: str
    command: List[str]
    schedule: str  # cron-like or keyword
    timeout: int = 300  # seconds
    retry_count: int = 2
    retry_delay: int = 60  # seconds
    enabled: bool = True
    dependencies: List[str] = None
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.notification_channels is None:
            self.notification_channels = []


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_name: str
    start_time: datetime
    end_time: datetime
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    error_message: Optional[str] = None


class NotificationManager:
    """Manages notifications for automation events."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.notification_config = config.get("automation", {}).get("notification_channels", {})
    
    async def send_notification(self, channel: str, message: str, severity: str = "info"):
        """Send notification to specified channel."""
        try:
            if channel.startswith("slack:"):
                await self._send_slack_notification(channel[6:], message, severity)
            elif channel.startswith("email:"):
                await self._send_email_notification(channel[6:], message, severity)
            elif channel == "console":
                logger.info(f"NOTIFICATION: {message}")
            else:
                logger.warning(f"Unknown notification channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to send notification to {channel}: {e}")
    
    async def _send_slack_notification(self, channel: str, message: str, severity: str):
        """Send Slack notification."""
        slack_config = self.notification_config.get("slack", {})
        webhook_url = slack_config.get(channel)
        
        if not webhook_url:
            logger.warning(f"No Slack webhook configured for channel: {channel}")
            return
        
        # Color coding based on severity
        color_map = {
            "info": "good",
            "warning": "warning", 
            "error": "danger",
            "critical": "danger"
        }
        
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨"
        }
        
        payload = {
            "channel": f"#{channel}",
            "attachments": [{
                "color": color_map.get(severity, "good"),
                "title": f"{emoji_map.get(severity, '🔔')} Automation Notification",
                "text": message,
                "ts": int(time.time())
            }]
        }
        
        # In a real implementation, you'd use aiohttp to send this
        logger.info(f"SLACK({channel}): {message}")
    
    async def _send_email_notification(self, recipient_group: str, message: str, severity: str):
        """Send email notification."""
        email_config = self.notification_config.get("email", {})
        recipients = email_config.get(recipient_group, [])
        
        if not recipients:
            logger.warning(f"No email recipients configured for group: {recipient_group}")
            return
        
        # In a real implementation, you'd send actual emails
        logger.info(f"EMAIL({recipient_group}): {message}")


class AutomationOrchestrator:
    """Orchestrates automation tasks based on schedules and events."""
    
    def __init__(self, repo_path: Path, config_path: Optional[Path] = None):
        self.repo_path = repo_path
        self.config_path = config_path or repo_path / ".github" / "project-metrics.json"
        self.config = self._load_config()
        self.tasks = self._load_automation_tasks()
        self.notification_manager = NotificationManager(self.config)
        self.task_history = []
        self.running = False
    
    def _load_config(self) -> Dict[str, Any]:
        """Load automation configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "automation": {
                "enabled_automations": ["metrics_collection", "security_scanning"],
                "schedules": {
                    "daily": ["collect_development_metrics"],
                    "weekly": ["comprehensive_metrics_report"],
                    "monthly": ["strategic_metrics_review"]
                }
            }
        }
    
    def _load_automation_tasks(self) -> Dict[str, AutomationTask]:
        """Load automation tasks from configuration."""
        tasks = {}
        
        # Core automation tasks
        task_definitions = {
            "collect_development_metrics": AutomationTask(
                name="collect_development_metrics",
                command=["python", "scripts/collect_metrics.py", "--output", "daily-metrics.json"],
                schedule="daily",
                timeout=600,
                notification_channels=["slack:general"]
            ),
            
            "security_vulnerability_scan": AutomationTask(
                name="security_vulnerability_scan",
                command=["python", "scripts/security_scan.py", "--comprehensive", "--output", "security-scan.json"],
                schedule="daily",
                timeout=900,
                notification_channels=["slack:security"]
            ),
            
            "performance_monitoring": AutomationTask(
                name="performance_monitoring",
                command=["python", "scripts/benchmark_suite.py", "--quick", "--output", "performance-metrics.json"],
                schedule="daily",
                timeout=1200,
                notification_channels=["slack:performance"]
            ),
            
            "repository_maintenance": AutomationTask(
                name="repository_maintenance",
                command=["python", "scripts/repository_maintenance.py", "--tasks", "cleanup_repository"],
                schedule="daily",
                timeout=300,
                notification_channels=["console"]
            ),
            
            "comprehensive_metrics_report": AutomationTask(
                name="comprehensive_metrics_report",
                command=["python", "scripts/generate_reports.py", "--type", "weekly", "--output", "weekly-report.json"],
                schedule="weekly",
                timeout=1800,
                dependencies=["collect_development_metrics", "security_vulnerability_scan"],
                notification_channels=["slack:general", "email:development_team"]
            ),
            
            "dependency_update_check": AutomationTask(
                name="dependency_update_check",
                command=["python", "scripts/repository_maintenance.py", "--tasks", "check_dependencies"],
                schedule="weekly",
                timeout=600,
                notification_channels=["slack:general"]
            ),
            
            "code_quality_analysis": AutomationTask(
                name="code_quality_analysis",
                command=["python", "scripts/repository_maintenance.py", "--tasks", "analyze_code_quality"],
                schedule="weekly",
                timeout=900,
                notification_channels=["slack:general"]
            ),
            
            "compliance_audit": AutomationTask(
                name="compliance_audit",
                command=["python", "scripts/run_quality_gates_comprehensive.py", "--compliance-check"],
                schedule="weekly",
                timeout=2400,
                notification_channels=["slack:compliance"]
            ),
            
            "strategic_metrics_review": AutomationTask(
                name="strategic_metrics_review",
                command=["python", "scripts/generate_reports.py", "--type", "monthly", "--output", "monthly-strategic-report.json"],
                schedule="monthly",
                timeout=3600,
                dependencies=["comprehensive_metrics_report", "compliance_audit"],
                notification_channels=["email:stakeholders"]
            ),
            
            "technical_debt_assessment": AutomationTask(
                name="technical_debt_assessment",
                command=["python", "scripts/repository_maintenance.py", "--tasks", "analyze_code_quality", "check_documentation"],
                schedule="monthly",
                timeout=1800,
                notification_channels=["email:development_team"]
            ),
            
            "capacity_planning": AutomationTask(
                name="capacity_planning",
                command=["python", "scripts/generate_reports.py", "--type", "capacity", "--output", "capacity-report.json"],
                schedule="monthly",
                timeout=1200,
                notification_channels=["email:stakeholders"]
            )
        }
        
        # Filter tasks based on enabled automations
        enabled_automations = self.config.get("automation", {}).get("enabled_automations", [])
        for task_name, task in task_definitions.items():
            if any(automation in task_name for automation in enabled_automations):
                tasks[task_name] = task
        
        return tasks
    
    async def execute_task(self, task: AutomationTask) -> TaskResult:
        """Execute a single automation task."""
        logger.info(f"🚀 Starting task: {task.name}")
        start_time = datetime.now()
        
        for attempt in range(task.retry_count + 1):
            try:
                process = await asyncio.create_subprocess_exec(
                    *task.command,
                    cwd=self.repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=task.timeout
                )
                
                end_time = datetime.now()
                success = process.returncode == 0
                
                result = TaskResult(
                    task_name=task.name,
                    start_time=start_time,
                    end_time=end_time,
                    success=success,
                    exit_code=process.returncode,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else ""
                )
                
                if success:
                    logger.info(f"✅ Task completed successfully: {task.name} ({(end_time - start_time).total_seconds():.1f}s)")
                    await self._notify_task_completion(task, result)
                    return result
                else:
                    logger.warning(f"⚠️ Task failed (attempt {attempt + 1}/{task.retry_count + 1}): {task.name}")
                    if attempt < task.retry_count:
                        await asyncio.sleep(task.retry_delay)
                        continue
                    else:
                        result.error_message = f"Task failed after {task.retry_count + 1} attempts"
                        await self._notify_task_failure(task, result)
                        return result
                        
            except asyncio.TimeoutError:
                end_time = datetime.now()
                logger.error(f"⏰ Task timed out: {task.name}")
                result = TaskResult(
                    task_name=task.name,
                    start_time=start_time,
                    end_time=end_time,
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr="Task timed out",
                    error_message=f"Task timed out after {task.timeout} seconds"
                )
                await self._notify_task_failure(task, result)
                return result
                
            except Exception as e:
                end_time = datetime.now()
                logger.error(f"💥 Task execution failed: {task.name} - {e}")
                result = TaskResult(
                    task_name=task.name,
                    start_time=start_time,
                    end_time=end_time,
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    error_message=str(e)
                )
                await self._notify_task_failure(task, result)
                return result
    
    async def _notify_task_completion(self, task: AutomationTask, result: TaskResult):
        """Notify about successful task completion."""
        duration = (result.end_time - result.start_time).total_seconds()
        message = f"✅ Task '{task.name}' completed successfully in {duration:.1f} seconds"
        
        for channel in task.notification_channels:
            await self.notification_manager.send_notification(channel, message, "info")
    
    async def _notify_task_failure(self, task: AutomationTask, result: TaskResult):
        """Notify about task failure."""
        duration = (result.end_time - result.start_time).total_seconds()
        message = f"❌ Task '{task.name}' failed after {duration:.1f} seconds\\nError: {result.error_message or result.stderr}"
        
        for channel in task.notification_channels:
            await self.notification_manager.send_notification(channel, message, "error")
    
    def _resolve_dependencies(self, tasks_to_run: List[str]) -> List[str]:
        """Resolve task dependencies and return execution order."""
        resolved_order = []
        remaining_tasks = tasks_to_run.copy()
        
        while remaining_tasks:
            progress_made = False
            
            for task_name in remaining_tasks[:]:
                task = self.tasks[task_name]
                dependencies_met = all(
                    dep in resolved_order or dep not in tasks_to_run
                    for dep in task.dependencies
                )
                
                if dependencies_met:
                    resolved_order.append(task_name)
                    remaining_tasks.remove(task_name)
                    progress_made = True
            
            if not progress_made:
                # Circular dependency or missing dependency
                logger.error(f"Cannot resolve dependencies for tasks: {remaining_tasks}")
                resolved_order.extend(remaining_tasks)
                break
        
        return resolved_order
    
    async def run_scheduled_tasks(self, schedule_type: str) -> List[TaskResult]:
        """Run tasks for a specific schedule type."""
        logger.info(f"📅 Running {schedule_type} scheduled tasks...")
        
        scheduled_tasks = self.config.get("automation", {}).get("schedules", {}).get(schedule_type, [])
        
        # Filter for enabled tasks
        available_tasks = [task for task in scheduled_tasks if task in self.tasks and self.tasks[task].enabled]
        
        if not available_tasks:
            logger.info(f"No {schedule_type} tasks to run")
            return []
        
        # Resolve dependencies
        execution_order = self._resolve_dependencies(available_tasks)
        
        logger.info(f"Task execution order: {' -> '.join(execution_order)}")
        
        results = []
        for task_name in execution_order:
            task = self.tasks[task_name]
            result = await self.execute_task(task)
            results.append(result)
            self.task_history.append(result)
            
            # If a critical task fails, consider stopping
            if not result.success and "critical" in task_name:
                logger.error(f"Critical task failed: {task_name}. Stopping execution.")
                break
        
        # Generate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        summary_message = f"📊 {schedule_type.title()} automation summary: {successful} successful, {failed} failed"
        logger.info(summary_message)
        
        # Send summary notification
        if results:
            await self.notification_manager.send_notification(
                "slack:general", 
                summary_message,
                "warning" if failed > 0 else "info"
            )
        
        return results
    
    def setup_scheduler(self):
        """Setup the task scheduler."""
        logger.info("⏰ Setting up task scheduler...")
        
        # Schedule daily tasks
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.run_scheduled_tasks("daily"))
        )
        
        # Schedule weekly tasks (Monday at 3 AM)
        schedule.every().monday.at("03:00").do(
            lambda: asyncio.create_task(self.run_scheduled_tasks("weekly"))
        )
        
        # Schedule monthly tasks (1st of month at 4 AM)
        schedule.every().month.do(
            lambda: asyncio.create_task(self.run_scheduled_tasks("monthly"))
        )
        
        logger.info("✅ Scheduler configured")
    
    async def run_daemon(self):
        """Run the automation orchestrator as a daemon."""
        logger.info("🤖 Starting automation orchestrator daemon...")
        self.running = True
        
        self.setup_scheduler()
        
        # Send startup notification
        await self.notification_manager.send_notification(
            "slack:general",
            "🤖 Automation orchestrator started successfully",
            "info"
        )
        
        while self.running:
            try:
                # Check for scheduled tasks
                schedule.run_pending()
                
                # Sleep for a minute before checking again
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                self.running = False
                break
            except Exception as e:
                logger.error(f"💥 Daemon error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        # Send shutdown notification
        await self.notification_manager.send_notification(
            "slack:general",
            "🛑 Automation orchestrator shutting down",
            "warning"
        )
        
        logger.info("👋 Automation orchestrator daemon stopped")
    
    async def run_one_time_tasks(self, task_names: List[str]) -> List[TaskResult]:
        """Run specific tasks one time."""
        logger.info(f"🎯 Running one-time tasks: {', '.join(task_names)}")
        
        # Validate task names
        invalid_tasks = [name for name in task_names if name not in self.tasks]
        if invalid_tasks:
            logger.error(f"Invalid task names: {invalid_tasks}")
            return []
        
        # Resolve dependencies
        execution_order = self._resolve_dependencies(task_names)
        
        results = []
        for task_name in execution_order:
            task = self.tasks[task_name]
            result = await self.execute_task(task)
            results.append(result)
            self.task_history.append(result)
        
        return results
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task status and history."""
        return {
            "configured_tasks": {name: asdict(task) for name, task in self.tasks.items()},
            "recent_history": [asdict(result) for result in self.task_history[-20:]],  # Last 20 tasks
            "running": self.running,
            "next_scheduled": {
                "daily": schedule.next_run(),
                "total_jobs": len(schedule.jobs)
            }
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automation orchestrator for neoRL-industrial-gym")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd(),
                      help="Path to repository (default: current directory)")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--run-tasks", nargs="+", help="Run specific tasks once")
    parser.add_argument("--schedule", choices=["daily", "weekly", "monthly"], 
                      help="Run tasks for specific schedule")
    parser.add_argument("--status", action="store_true", help="Show task status")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    orchestrator = AutomationOrchestrator(args.repo_path, args.config)
    
    try:
        if args.daemon:
            await orchestrator.run_daemon()
            
        elif args.run_tasks:
            results = await orchestrator.run_one_time_tasks(args.run_tasks)
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            print(f"Completed: {successful} successful, {failed} failed")
            if failed > 0:
                sys.exit(1)
                
        elif args.schedule:
            results = await orchestrator.run_scheduled_tasks(args.schedule)
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            print(f"Completed: {successful} successful, {failed} failed")
            if failed > 0:
                sys.exit(1)
                
        elif args.status:
            status = orchestrator.get_task_status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.list_tasks:
            print("Available automation tasks:")
            for name, task in orchestrator.tasks.items():
                enabled_status = "✅" if task.enabled else "❌"
                print(f"  {enabled_status} {name} ({task.schedule})")
            
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"💥 Orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())