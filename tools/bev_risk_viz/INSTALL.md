# BEV Risk Map Generator - Installation Guide

## Quick Installation

### 1. Install Dependencies

```bash
# Navigate to BEVFormer directory
cd /path/to/BEVFormer

# Install required packages
pip install -r tools/bev_risk_viz/requirements.txt
```

### 2. Verify Installation

```bash
# Run example script to verify everything works
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/example_usage.py
```

This will generate several example visualizations:
- `example_1_simple.png` - Basic risk map
- `example_2_breakdown.png` - Risk factor analysis
- `example_3_comparison.png` - Parameter comparison
- `example_4_*` - Multiple export formats
- `example_5_velocity_impact.png` - Velocity analysis

### 3. Test Command-Line Interface

```bash
# Run demo scenario
PYTHONPATH=.:$PYTHONPATH python tools/bev_risk_viz/cli.py \
    --mode demo \
    --demo-scenario "Multi-Vehicle Intersection" \
    --export png,pdf
```

## Optional: Install for Interactive GUI

```bash
# Install Streamlit for web-based GUI
pip install streamlit

# Launch GUI
streamlit run tools/bev_risk_viz/gui_app.py
```

Access the GUI at: http://localhost:8501

## Optional: Install nuScenes Support

For nuScenes dataset integration:

```bash
# Install nuScenes devkit
pip install nuscenes-devkit

# Verify nuScenes data is available
ls data/nuscenes/
# Should contain: v1.0-mini or v1.0-trainval
```

## System Requirements

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 100MB for tool + dataset size
- **OS**: Linux, macOS, or Windows with WSL

## Package Dependencies

### Core (Required)
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- pyyaml >= 6.0
- opencv-python >= 4.5.0
- pillow >= 9.0.0

### Optional
- streamlit >= 1.20.0 (for GUI)
- nuscenes-devkit >= 1.1.9 (for dataset support)

## Troubleshooting

### Issue: Import errors

**Solution**: Make sure to set PYTHONPATH correctly

```bash
# From BEVFormer root directory
export PYTHONPATH=.:$PYTHONPATH
python tools/bev_risk_viz/cli.py --help
```

Or use absolute imports in Python:

```python
import sys
sys.path.insert(0, '/path/to/BEVFormer')
from tools.bev_risk_viz import RiskCalculationEngine
```

### Issue: Qt/Display errors

**Solution**: Use non-interactive backend

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

This is already set in CLI and example scripts.

### Issue: nuScenes not found

**Solution**: Check data path in config.yaml

```yaml
data_source:
  nuscenes:
    data_root: "data/nuscenes"  # Update this path
    version: "v1.0-mini"
```

### Issue: Memory error with large grids

**Solution**: Reduce grid resolution

```yaml
bev_grid:
  resolution: 1.0  # Increase from default 0.5
```

## Verification Checklist

After installation, verify:

- [ ] Example script runs without errors
- [ ] PNG visualizations are generated
- [ ] PDF reports are created
- [ ] CLI help displays correctly
- [ ] Configuration loads successfully

## Next Steps

1. **Read Documentation**: [README.md](README.md)
2. **Try Examples**: Run `example_usage.py`
3. **Explore CLI**: `python tools/bev_risk_viz/cli.py --help`
4. **Launch GUI**: `streamlit run tools/bev_risk_viz/gui_app.py`
5. **Customize Config**: Edit `config.yaml` for your needs

## Support

For issues or questions:
- Check [README.md](README.md) for usage examples
- Review [config.yaml](config.yaml) for all parameters
- Run examples to see working code

## Update

To get the latest version:

```bash
cd /path/to/BEVFormer
git pull origin master
pip install -r tools/bev_risk_viz/requirements.txt --upgrade
```
