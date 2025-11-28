#!/bin/bash
# BEV-RiskViz 실행 스크립트

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PYTHONPATH 설정
export PYTHONPATH=".:$PYTHONPATH"

echo "========================================="
echo "  BEV Risk Map Generator (BEV-RiskViz)"
echo "========================================="
echo ""

# 명령어 파싱
case "${1:-help}" in
    demo)
        echo "Running demo mode..."
        python tools/bev_risk_viz/cli.py \
            --mode demo \
            --demo-scenario "${2:-Multi-Vehicle Intersection}" \
            --export png,pdf
        ;;

    example)
        echo "Running example scripts..."
        python tools/bev_risk_viz/example_usage.py
        ;;

    gui)
        echo "Launching GUI (Streamlit)..."
        streamlit run tools/bev_risk_viz/gui_app.py
        ;;

    cli)
        echo "Running CLI with custom arguments..."
        shift
        python tools/bev_risk_viz/cli.py "$@"
        ;;

    nuscenes)
        echo "Running nuScenes mode..."
        scene="${2:-scene-0001}"
        python tools/bev_risk_viz/cli.py \
            --mode nuscenes \
            --scene "$scene" \
            --export png
        ;;

    help|*)
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  demo [scenario]    - Run demo with scenario"
        echo "                       Scenarios: 'Simple Occlusion', 'Multi-Vehicle Intersection',"
        echo "                                  'Parking Lot Exit', 'Highway Merge', 'Pedestrian Crossing'"
        echo "  example            - Run example scripts (5 examples)"
        echo "  gui                - Launch interactive GUI (Streamlit)"
        echo "  cli [options]      - Run CLI with custom options"
        echo "  nuscenes [scene]   - Process nuScenes scene (e.g., scene-0001)"
        echo "  help               - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 demo"
        echo "  $0 demo 'Parking Lot Exit'"
        echo "  $0 example"
        echo "  $0 gui"
        echo "  $0 nuscenes scene-0001"
        echo "  $0 cli --mode demo --export png,pdf,npy"
        echo ""
        ;;
esac
