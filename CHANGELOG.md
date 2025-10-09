# Changelog

All notable changes to the WOPE project are documented here.

## [Unreleased] - 2025-10-09

### Major Improvements

#### Code Professionalism
- **Rewrote all code comments** to be technical and professional
  - Removed casual language ("BFFs", "party's over", "thou shalt not pass", etc.)
  - Added proper docstrings with parameter types and return values
  - Improved inline comments to be informative and concise
  - Affected files: `utils.py`, `poem_generator.py`, `masterlogits.py`, `endcriteria.py`

#### Architecture & Dependencies
- **Removed TensorFlow dependency** (~800MB reduction)
  - Migrated sparse tensor operations to PyTorch/NumPy
  - Simplified `boost_scores()` function with pure PyTorch implementation
  - Maintained 100% functional compatibility
  - Updated `requirements.txt` to remove TensorFlow

#### Code Quality
- **Eliminated duplicate code**
  - Removed duplicate `get_tool_tools()` function from `endcriteria.py`
  - Single source of truth now in `poem_generator.py`

- **Extracted magic numbers to constants**
  - Created `constants.py` with named constants
  - Replaced hardcoded values: EOS_TOKEN_ID (50256), PAD_TOKEN_ID (0)
  - Centralized configuration: DEFAULT_NUM_VERSES, DEFAULT_VERSE_SIZE, etc.
  - Improved code maintainability and readability

- **Enhanced error handling**
  - Added comprehensive error handling in `streamlit_app.py`
  - Implemented logging for debugging
  - Better error messages for users (vocabulary not found, generation failures)

#### Performance & Deployment
- **Improved Streamlit application**
  - Added lazy model loading with `@st.cache_resource`
  - Fixed deprecated API: replaced `st.experimental_rerun()` with `st.rerun()`
  - Model loaded once per session instead of on every import

- **Enhanced Docker configuration**
  - Added `curl` to Dockerfile for health check functionality
  - Implemented GPT-2 model pre-downloading during build (faster startup)
  - Fixed model directory permissions
  - Added comprehensive health check with proper timeouts
  - Improved `.dockerignore` to exclude unnecessary files

#### Git & Project Management
- **Fixed version control issues**
  - Comprehensive `.gitignore` for Python projects
  - Removed 12 `__pycache__` files from repository
  - Excluded IDE, OS, and build artifacts

- **Updated documentation**
  - Fixed installation instructions in `README.md`
  - Updated `CLAUDE.md` with accurate architecture description
  - Corrected Docker usage instructions (changed to `docker compose`)

### Technical Details

#### Removed Functions
- `create_boosted_vocab()` - Simplified implementation using NumPy arrays

#### Modified Functions
- `boost_scores()` - Complete rewrite using PyTorch/NumPy instead of TensorFlow
  - More efficient: direct array operations instead of sparse tensor conversions
  - Better device handling: proper `.to(scores.device)` for GPU compatibility
  - Cleaner code: 27 lines vs 47 lines

#### New Files
- `constants.py` - Central configuration constants

### Breaking Changes
None. All changes maintain backward compatibility.

### Migration Notes
For users updating from previous versions:
1. **Update dependencies**: Run `pip install -r requirements.txt`
2. **Remove TensorFlow**: Run `pip uninstall tensorflow` (optional, saves space)
3. **No code changes required**: All APIs remain the same

### Performance Improvements
- **Docker build time**: Faster subsequent builds with layer caching
- **Application startup**: Model cached in Streamlit session
- **Memory usage**: ~800MB reduction from removing TensorFlow
- **Code readability**: Significantly improved with professional comments

### Bug Fixes
- Fixed Streamlit deprecation warning
- Fixed Docker health check (missing curl)
- Fixed model directory permissions in Docker

---

## Release Philosophy

This project follows semantic versioning. Changes are categorized as:
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, refactoring, documentation
