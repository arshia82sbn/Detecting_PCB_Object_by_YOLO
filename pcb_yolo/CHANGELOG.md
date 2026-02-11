# Changelog

## [0.1.0] - 2026-02-02

### Added
- Complete YOLOv8-based PCB defect detection pipeline.
- Factory pattern for models and datasets.
- Pipeline pattern for training and deployment.
- Singleton Logger for structured run tracking.
- Hardened `prepare_dataset.py` with stratified splitting.
- Support for Roboflow dataset downloads.
- ONNX and TorchScript export functionality.
- Comprehensive integration tests and unit tests for metrics.
- CI/CD workflow with GitHub Actions.
- Academic experiment protocol documentation.

### Fixed
- Resolved `ModuleNotFoundError` during test execution.
- Fixed absolute path issues for experiment artifact registration.
- Corrected JSON metadata schema to match deployment requirements.
- Updated metrics calculation for NumPy 2.0 compatibility.
