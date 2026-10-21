import argparse
import os
import sys

def pdf_to_images(pdf_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. PyMuPDF (fitz) 사용 (가장 안전하고 시스템 의존성 없음)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap()
            out_path = os.path.join(out_dir, f"page_{i+1}.png")
            pix.save(out_path)
            print(f"Saved: {out_path}")
        return len(doc)
    except ImportError:
        pass

    # 2. pdf2image 사용 (poppler 엔진이 os상에 필요함)
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            out_path = os.path.join(out_dir, f"page_{i+1}.png")
            image.save(out_path, "PNG")
            print(f"Saved: {out_path}")
        return len(images)
    except ImportError:
        pass

    raise RuntimeError("Please install 'pymupdf' or 'pdf2image' to convert PDF to images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to PDF file")
    parser.add_argument("--out", default="./sandbox", help="Output directory")
    args = parser.parse_args()

    try:
        count = pdf_to_images(args.path, args.out)
        print(f"Successfully converted {count} pages to images under {args.out}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
