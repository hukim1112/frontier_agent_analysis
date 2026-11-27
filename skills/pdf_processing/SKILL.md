---
name: pdf-processing
description: Extract text and metadata (title, author, page count) from PDF files, or render PDF pages to PNG images, by running the bundled Python scripts with bash_command.
---

# PDF Processing Skill

This skill provides scripts to extract text, metadata, and convert PDF pages to PNG images.

## Available Scripts
All scripts are located in `skills/pdf_processing/scripts/` and should be executed with `bash_command` using Python 3 relative to the workspace root.

---

### 📄 1. pdf_extractor.py
- **Purpose**: Extracts text and metadata (author, title, page count) from a PDF.
- **Path**: `skills/pdf_processing/scripts/pdf_extractor.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
- **Usage**:
  ```bash
  python skills/pdf_processing/scripts/pdf_extractor.py --path [PDF_PATH]
  ```

---

### 📝 2. pdf2text.py
- **Purpose**: Extracts raw text contents from all pages of a PDF.
- **Path**: `skills/pdf_processing/scripts/pdf2text.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
- **Usage**:
  ```bash
  python skills/pdf_processing/scripts/pdf2text.py --path [PDF_PATH]
  ```

---

### 🖼️ 3. pdf2image.py
- **Purpose**: Converts PDF pages into PNG images saved in an output directory.
- **Path**: `skills/pdf_processing/scripts/pdf2image.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
  - `--out [OUT_DIR]`: Directory to save the output images (Optional, defaults to `./sandbox/`)
- **Usage**:
  ```bash
  python skills/pdf_processing/scripts/pdf2image.py --path [PDF_PATH] --out [OUT_DIR]
  ```
