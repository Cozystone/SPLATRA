# -*- coding: utf-8 -*-
"""SPLATRA narrative — thought (concepts+relations) -> Gaussian scene spec with physics motion,
structural silhouettes and a LoD budget. Mirrored from the ATANOR monorepo (presentation layer)."""
from .scene_compiler import compile_scene
from .cloud import form_for, shape_spec, lod_budget, decimate_importance, scene_point_estimate

__all__ = ["compile_scene", "form_for", "shape_spec", "lod_budget",
           "decimate_importance", "scene_point_estimate"]
