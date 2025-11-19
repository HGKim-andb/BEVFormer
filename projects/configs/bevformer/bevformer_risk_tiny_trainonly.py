"""
BEVFormer with Risk Prediction - Tiny Version (Training Only)
For quick testing without validation
"""

_base_ = ['./bevformer_risk_tiny.py']

# Disable validation during training
data = dict(
    val=dict(
        type='NuScenesRiskDataset',
        use_risk=False,  # Disable risk for validation
    ),
    test=dict(
        type='NuScenesRiskDataset',
        use_risk=False,  # Disable risk for test
    ),
)

# Disable evaluation
evaluation = None

# Shorter training for testing
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
