"""
SCLAM Models Package

This package contains the three main models in the SCLAM pipeline:
- snow17: SNOW17 snowmelt model
- hydro_model: CREST distributed hydrological model
- landslide: Landslide probability prediction models (RF and Infinite Slope)
"""

__version__ = "1.0.0"
__all__ = ["run_snow17", "run_crest_model", "run_landslide"]

def __getattr__(name):
    if name == "run_snow17":
        from .snow17 import main as run_snow17

        return run_snow17
    if name == "run_crest_model":
        from .hydro_model import run_crest_model

        return run_crest_model
    if name == "run_landslide":
        from .landslide import main as run_landslide

        return run_landslide
    raise AttributeError(name)
