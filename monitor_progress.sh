#!/bin/bash
# Monitor experiment progress in real-time

echo "================================================================================"
echo "THESIS EXPERIMENT PROGRESS MONITOR"
echo "================================================================================"
echo ""

# Check if checkpoint exists
if [ -f "checkpoint_progress.csv" ]; then
    COMPLETED=$(tail -n +2 checkpoint_progress.csv | wc -l)
    TOTAL=30
    REMAINING=$((TOTAL - COMPLETED))
    PERCENT=$((COMPLETED * 100 / TOTAL))
    
    echo "📊 Progress: $COMPLETED/$TOTAL networks completed ($PERCENT%)"
    echo "⏳ Remaining: $REMAINING networks"
    echo ""
    
    echo "✅ Completed networks:"
    tail -n +2 checkpoint_progress.csv | sort
    echo ""
    
    # Estimate time remaining if log exists
    if [ -f "experiment.log" ]; then
        LAST_ETA=$(grep "ETA:" experiment.log | tail -1 | awk '{print $NF}')
        if [ ! -z "$LAST_ETA" ]; then
            echo "⏱️  Last ETA: $LAST_ETA"
            echo ""
        fi
    fi
    
    # Check for errors
    if [ -f "experiment.log" ]; then
        ERROR_COUNT=$(grep -c "✗ Error:" experiment.log 2>/dev/null || echo "0")
        if [ "$ERROR_COUNT" -gt 0 ]; then
            echo "⚠️  Errors encountered: $ERROR_COUNT"
            echo "   Last error:"
            grep "✗ Error:" experiment.log | tail -1
            echo ""
        fi
    fi
    
    # Show current network being processed
    if [ -f "experiment.log" ]; then
        CURRENT=$(grep "Network [0-9]*/30:" experiment.log | tail -1)
        if [ ! -z "$CURRENT" ]; then
            echo "🔄 Current: $CURRENT"
            echo ""
        fi
    fi
    
else
    echo "❌ No checkpoint found"
    echo "   Either:"
    echo "   - Experiment hasn't started yet"
    echo "   - Experiment completed (checkpoint cleaned up)"
    echo "   - Checkpoint file was deleted"
    echo ""
fi

# Check if process is running
if pgrep -f "thesis_experiments_final_script.py" > /dev/null; then
    PID=$(pgrep -f "thesis_experiments_final_script.py")
    echo "✅ Script is RUNNING (PID: $PID)"
    
    # Show CPU and memory usage
    if command -v ps &> /dev/null; then
        echo "   CPU/Memory usage:"
        ps -p $PID -o pid,pcpu,pmem,etime,cmd | tail -1
    fi
else
    echo "⏸️  Script is NOT running"
    echo "   To start/resume: python thesis_experiments_final_script.py"
fi

echo ""

# Check disk space
echo "💾 Disk space:"
df -h . | tail -1 | awk '{print "   Total: "$2"  Used: "$3"  Free: "$4"  ("$5" used)"}'

echo ""

# Check file sizes
if [ -f "test_thesis_4rewards_30networks_metrics.csv" ]; then
    echo "📁 Output files:"
    ls -lh test_thesis_4rewards_30networks_*.csv 2>/dev/null | awk '{print "   "$9": "$5}'
    ls -lh thesis_4rewards_30networks_reward_landscape.csv 2>/dev/null | awk '{print "   "$9": "$5}'
fi

echo ""
echo "================================================================================"
echo "Commands:"
echo "  tail -f experiment.log          # Watch live output"
echo "  python test_checkpoint_system.py # Detailed checkpoint info"
echo "  ./monitor_progress.sh           # Refresh this view"
echo "================================================================================"
