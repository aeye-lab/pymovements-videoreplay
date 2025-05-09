"""
antiocr: A package for generating stimulus images from eye-tracking annotations.

This package provides utilities for creating stimulus images using word labels
and pixel coordinates, effectively reversing the OCR process.

Main Components:
- `AntiOCR`: Class to generate PNG stimuli from CSV annotations.

Example Usage:
    from antiocr import AntiOCR

    generator = AntiOCR(frame_width=1920, frame_height=1080)
    generator.generate_from_csv("annotations.csv", output_path="stimulus.png")
"""
from .anti_ocr import AntiOCR

__all__ = ['AntiOCR']
