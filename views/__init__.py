"""
Views package untuk DETR Isolated Expression Detector
Berisi UI components untuk Streamlit
"""
from .detection_view import DetectionView
from .training_view import TrainingView

__all__ = ['DetectionView', 'TrainingView']
