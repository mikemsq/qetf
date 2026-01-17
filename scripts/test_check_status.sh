#!/bin/bash
# Test check_status.sh script

echo "Testing check_status.sh script..."
echo ""
echo "Script location: $(pwd)/check_status.sh"
echo "Script size: $(wc -l < check_status.sh) lines"
echo ""

echo "Running: ./check_status.sh --all"
echo "=================================================="
./check_status.sh --all
echo "=================================================="
echo ""
echo "Test complete!"
