---
name: pdf-processing
description: Extract text and metadata (title, author, page count) from PDF files, or render PDF pages to PNG images, by running the bundled Python scripts with bash_command.
---

# PDF Processing Skill

This skill provides tools to extract text, metadata, and convert PDF pages to images.

## Available Scripts
All scripts are written in Python and expect the PDF file path as an argument.
Run them with `bash_command` from the workspace root.

---

### 📄 1. pdf_extractor.py
- **Purpose**: Extracts text and metadata (author, title, page count) from a PDF.
- **Path**: `app/agents/main_agent/skills/pdf_processing/scripts/pdf_extractor.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
- **Usage**:
  ```bash
  python app/agents/main_agent/skills/pdf_processing/scripts/pdf_extractor.py --path [PDF_PATH]
  ```

---

### 📝 2. pdf2text.py
- **Purpose**: Extracts raw text contents from all pages of a PDF.
- **Path**: `app/agents/main_agent/skills/pdf_processing/scripts/pdf2text.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
- **Usage**:
  ```bash
  python app/agents/main_agent/skills/pdf_processing/scripts/pdf2text.py --path [PDF_PATH]
  ```

---

### 🖼️ 3. pdf2image.py
- **Purpose**: Converts PDF pages into PNG images saved in an output directory.
- **Path**: `app/agents/main_agent/skills/pdf_processing/scripts/pdf2image.py`
- **Arguments**:
  - `--path [PDF_PATH]`: Absolute or relative path to the PDF file (Required)
  - `--out [OUT_DIR]`: Directory to save the output images (Optional, defaults to `./sandbox/`)
- **Usage**:
  ```bash
  python app/agents/main_agent/skills/pdf_processing/scripts/pdf2image.py --path [PDF_PATH] --out [OUT_DIR]
  ```
