import argparse
import json
import sys

def extract_pdf_info(pdf_path):
    metadata = {"path": pdf_path}
    text = ""
    
    # 1. 가상 PDF 컨테이너 (텍스트 파일 형태) 감지 및 모의 파싱
    try:
        with open(pdf_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line == "[Virtual PDF Container]":
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip().lower()] = v.strip()
                text = metadata.get("content", "")
                metadata["pages"] = int(metadata.get("pages", 1))
                metadata["library"] = "VirtualParser"
                return metadata, text
    except Exception:
        pass

    # 2. 진짜 바이너리 PDF 파싱 (fitz - PyMuPDF)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        metadata.update({
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "pages": len(doc),
            "library": "PyMuPDF"
        })
        for page in doc:
            text += page.get_text()
        return metadata, text
    except ImportError:
        pass

    # 3. pypdf 시도 (폴백)
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        info = reader.metadata
        metadata.update({
            "title": info.title or "",
            "author": info.author or "",
            "subject": info.subject or "",
            "pages": len(reader.pages),
            "library": "pypdf"
        })
        for page in reader.pages:
            text += page.extract_text() or ""
        return metadata, text
    except ImportError:
        pass

    raise RuntimeError("Please install 'pypdf' or 'pymupdf' to parse PDF files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to PDF file")
    args = parser.parse_args()

    try:
        meta, text = extract_pdf_info(args.path)
        result = {
            "status": "SUCCESS",
            "metadata": meta,
            "text_preview": text[:1000] + ("..." if len(text) > 1000 else "")
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"status": "FAILED", "error": str(e)}, ensure_ascii=False))
        sys.exit(1)
