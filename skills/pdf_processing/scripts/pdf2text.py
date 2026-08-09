import argparse
import sys

def pdf_to_text(pdf_path):
    # 1. 가상 PDF 컨테이너 (텍스트 파일 형태) 감지 및 본문 모의 파싱
    try:
        with open(pdf_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line == "[Virtual PDF Container]":
                for line in f:
                    if line.startswith("Content:"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass

    text = ""
    # 2. 진짜 바이너리 PDF 파싱 (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except ImportError:
        pass

    # 3. pypdf 시도
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except ImportError:
        pass

    raise RuntimeError("Please install 'pypdf' or 'pymupdf' to parse PDF files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to PDF file")
    args = parser.parse_args()

    try:
        text = pdf_to_text(args.path)
        print(text)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
