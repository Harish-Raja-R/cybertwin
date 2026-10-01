"""
Digital Twin State Utilities
=============================
Thin wrapper around the existing src/digital_twin.py module for use by the
Streamlit application.  Does NOT duplicate state logic — imports the canonical
DigitalTwinNode and NodeState from src/.
"""
import sys, os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.digital_twin import DigitalTwinNode, NodeState  # noqa: F401


def state_color(state_value: str) -> str:
    """Return a hex color for the given Digital Twin state string."""
    return {
        'NORMAL':       '#4CAF50',
        'SUSPICIOUS':   '#FFC107',
        'UNDER_ATTACK': '#F44336',
        'RECOVERING':   '#2196F3',
    }.get(state_value, '#9E9E9E')


def state_emoji(state_value: str) -> str:
    return {
        'NORMAL':       '🟢',
        'SUSPICIOUS':   '🟡',
        'UNDER_ATTACK': '🔴',
        'RECOVERING':   '🔵',
    }.get(state_value, '⚪')


def risk_band(risk_0_100: float) -> str:
    """Map a 0–100 risk score to a human-readable band."""
    if risk_0_100 <= 30:
        return 'Low'
    elif risk_0_100 <= 60:
        return 'Moderate'
    elif risk_0_100 <= 80:
        return 'High'
    else:
        return 'Critical'


def risk_band_color(risk_0_100: float) -> str:
    if risk_0_100 <= 30:
        return '#4CAF50'
    elif risk_0_100 <= 60:
        return '#FFC107'
    elif risk_0_100 <= 80:
        return '#FF9800'
    else:
        return '#F44336'
