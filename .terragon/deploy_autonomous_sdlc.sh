#!/bin/bash
# Terragon Autonomous SDLC Deployment Script
# Deploys the autonomous value discovery and execution system

set -e

echo "🚀 Deploying Terragon Autonomous SDLC System"
echo "============================================"

# Configuration
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
TERRAGON_DIR="$REPO_ROOT/.terragon"

# Ensure we're in the right directory
cd "$REPO_ROOT"

echo "📍 Repository: $(basename "$REPO_ROOT")"
echo "📍 Terragon Directory: $TERRAGON_DIR"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p "$TERRAGON_DIR"
mkdir -p "$REPO_ROOT/.github/workflows" 2>/dev/null || true

# Validate Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 available: $(python3 --version)"

# Run initial value discovery
echo "🔍 Running initial value discovery..."
if [ -f "$TERRAGON_DIR/value_discovery_simple.py" ]; then
    python3 "$TERRAGON_DIR/value_discovery_simple.py"
    echo "✅ Value discovery completed"
else
    echo "❌ Value discovery script not found"
    exit 1
fi

# Create systemd service (optional, for continuous execution)
create_systemd_service() {
    local service_file="/etc/systemd/system/terragon-autonomous-sdlc.service"
    
    if [ "$EUID" -eq 0 ]; then
        echo "🔧 Creating systemd service..."
        cat > "$service_file" << EOF
[Unit]
Description=Terragon Autonomous SDLC
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$REPO_ROOT
ExecStart=$REPO_ROOT/.terragon/autonomous_executor.py --continuous
Restart=always
RestartSec=300

[Install]
WantedBy=multi-user.target
EOF
        
        systemctl daemon-reload
        systemctl enable terragon-autonomous-sdlc
        echo "✅ Systemd service created and enabled"
    else
        echo "ℹ️  Skipping systemd service (requires root)"
    fi
}

# Create cron jobs for periodic execution
setup_cron_jobs() {
    echo "⏰ Setting up cron jobs..."
    
    # Create cron script
    cat > "$TERRAGON_DIR/cron_runner.sh" << 'EOF'
#!/bin/bash
REPO_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$REPO_ROOT"

case "$1" in
    "discovery")
        python3 .terragon/value_discovery_simple.py > /tmp/terragon-discovery.log 2>&1
        ;;
    "execution")
        python3 .terragon/autonomous_executor.py > /tmp/terragon-execution.log 2>&1
        ;;
    *)
        echo "Usage: $0 {discovery|execution}"
        exit 1
        ;;
esac
EOF
    
    chmod +x "$TERRAGON_DIR/cron_runner.sh"
    
    # Add cron jobs (user decides whether to install)
    echo "📝 Cron job script created at: $TERRAGON_DIR/cron_runner.sh"
    echo ""
    echo "To enable automatic execution, add these cron jobs:"
    echo "# Hourly value discovery"
    echo "0 * * * * $TERRAGON_DIR/cron_runner.sh discovery"
    echo "# Daily autonomous execution"
    echo "0 2 * * * $TERRAGON_DIR/cron_runner.sh execution"
    echo ""
}

# Test execution
test_execution() {
    echo "🧪 Testing autonomous execution (dry-run)..."
    
    if python3 "$TERRAGON_DIR/autonomous_executor.py" --dry-run; then
        echo "✅ Autonomous execution test passed"
    else
        echo "❌ Autonomous execution test failed"
        return 1
    fi
}

# Create monitoring script
create_monitoring_script() {
    echo "📊 Creating monitoring script..."
    
    cat > "$TERRAGON_DIR/monitor.py" << 'EOF'
#!/usr/bin/env python3
"""Simple monitoring script for Terragon Autonomous SDLC."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

def main():
    repo_root = Path(__file__).parent.parent
    execution_log = repo_root / ".terragon" / "execution-log.json"
    backlog_file = repo_root / "AUTONOMOUS_BACKLOG.md"
    
    print("📊 Terragon Autonomous SDLC Status")
    print("=" * 40)
    
    # Check if system is set up
    if not execution_log.parent.exists():
        print("❌ System not initialized")
        sys.exit(1)
    
    # Execution history
    if execution_log.exists():
        try:
            with open(execution_log) as f:
                log_data = json.load(f)
            
            print(f"📈 Total Executions: {len(log_data)}")
            
            if log_data:
                recent = [entry for entry in log_data 
                         if datetime.fromisoformat(entry['timestamp']) > datetime.now() - timedelta(days=7)]
                print(f"📅 This Week: {len(recent)}")
                
                successful = [entry for entry in log_data if entry['result']['success']]
                print(f"✅ Success Rate: {len(successful)}/{len(log_data)} ({100*len(successful)/len(log_data):.1f}%)")
        except (json.JSONDecodeError, KeyError):
            print("⚠️  Execution log format error")
    else:
        print("📈 No executions yet")
    
    # Backlog status
    if backlog_file.exists():
        with open(backlog_file) as f:
            content = f.read()
        
        lines = content.split('\n')
        items_line = [line for line in lines if "Items Discovered" in line]
        if items_line:
            print(f"📋 {items_line[0].split('**')[1]}")
    
    print(f"📝 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$TERRAGON_DIR/monitor.py"
    echo "✅ Monitoring script created"
}

# Main deployment steps
main() {
    echo "1️⃣  Validating environment..."
    
    echo "2️⃣  Setting up cron jobs..."
    setup_cron_jobs
    
    echo "3️⃣  Creating monitoring..."
    create_monitoring_script
    
    echo "4️⃣  Testing execution..."
    if ! test_execution; then
        echo "❌ Deployment failed during testing"
        exit 1
    fi
    
    echo "5️⃣  Optional: systemd service..."
    if [ "${INSTALL_SERVICE:-}" = "true" ]; then
        create_systemd_service
    else
        echo "ℹ️  Skipping systemd service (set INSTALL_SERVICE=true to enable)"
    fi
    
    echo ""
    echo "🎉 Terragon Autonomous SDLC Successfully Deployed!"
    echo "================================================="
    echo ""
    echo "📁 Files created:"
    echo "  - .terragon/config.yaml (configuration)"
    echo "  - .terragon/value_discovery_simple.py (discovery engine)"
    echo "  - .terragon/autonomous_executor.py (execution engine)"
    echo "  - .terragon/cron_runner.sh (scheduled execution)"
    echo "  - .terragon/monitor.py (monitoring script)"
    echo "  - AUTONOMOUS_BACKLOG.md (value backlog)"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Review AUTONOMOUS_BACKLOG.md for discovered value items"
    echo "  2. Run: python3 .terragon/autonomous_executor.py --dry-run"
    echo "  3. Set up cron jobs for automatic execution (see above)"
    echo "  4. Monitor with: python3 .terragon/monitor.py"
    echo ""
    echo "🔄 Continuous execution:"
    echo "  python3 .terragon/autonomous_executor.py --continuous"
    echo ""
    echo "✨ The system will now continuously discover and execute the highest-value work!"
}

# Run main function
main "$@"