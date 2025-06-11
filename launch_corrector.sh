#!/bin/bash

# SmartDartCorrector Launch Script
# This script launches the corrector first, then the SmartDarts game

echo "========================================"
echo "SmartDartCorrector Launch Script"
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Define paths
GAME_PATH="../smartDarts/smartdarts"
CORRECTOR_SCRIPT="run_corrector.py"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if the corrector script exists
if [ ! -f "$CORRECTOR_SCRIPT" ]; then
    echo "Error: $CORRECTOR_SCRIPT not found in current directory"
    echo "Please make sure you're in the SmartDartCorrector directory"
    exit 1
fi

# Check if Godot is available
if ! command -v godot &> /dev/null; then
    echo "Error: Godot is not installed or not in PATH"
    echo "Please install Godot and make sure it's accessible from command line"
    exit 1
fi

# Check if the game project exists
if [ ! -d "$GAME_PATH" ]; then
    echo "Error: Game directory not found at $GAME_PATH"
    exit 1
fi

if [ ! -f "$GAME_PATH/project.godot" ]; then
    echo "Error: project.godot not found in $GAME_PATH"
    exit 1
fi

echo "Starting corrector..."
python3 "$CORRECTOR_SCRIPT" &
CORRECTOR_PID=$!
echo "Corrector launched with PID: $CORRECTOR_PID"

# Wait a moment for the corrector to initialize
echo "Waiting for corrector to initialize..."
sleep 3

echo "========================================"
echo "Starting SmartDarts game..."
echo "Game path: $GAME_PATH"

# Launch Godot game
cd "$GAME_PATH"
godot &
GAME_PID=$!
echo "Game launched with PID: $GAME_PID"

# Return to corrector directory
cd - > /dev/null

echo "========================================"
echo "Both processes are running:"
echo "- Corrector PID: $CORRECTOR_PID"
echo "- Game PID: $GAME_PID"
echo "Press Ctrl+C to stop both processes"

# Function to cleanup on exit
cleanup() {
    echo "\n========================================"
    echo "Stopping processes..."
    
    if kill -0 $CORRECTOR_PID 2>/dev/null; then
        kill $CORRECTOR_PID
        echo "Corrector process terminated"
    fi
    
    if kill -0 $GAME_PID 2>/dev/null; then
        kill $GAME_PID
        echo "Game process terminated"
    fi
    
    echo "Script terminated"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $CORRECTOR_PID
CORRECTOR_EXIT_CODE=$?

echo "========================================"
echo "Corrector finished (exit code: $CORRECTOR_EXIT_CODE)"

# Stop the game when corrector finishes
if kill -0 $GAME_PID 2>/dev/null; then
    kill $GAME_PID
    echo "Game process terminated"
fi

if [ $CORRECTOR_EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully"
else
    echo "Script exited with corrector error code: $CORRECTOR_EXIT_CODE"
    exit $CORRECTOR_EXIT_CODE
fi

