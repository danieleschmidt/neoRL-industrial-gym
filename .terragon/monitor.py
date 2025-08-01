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
