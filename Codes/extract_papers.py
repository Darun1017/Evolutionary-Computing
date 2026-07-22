import os, sys
import PyPDF2

sys.stdout.reconfigure(encoding='utf-8')

papers_dir = r"C:\Users\Pradeep S\Desktop\Semester VI\Project Phase - 1\Review 2 Papers\Review 2 Papers"
fnames = sorted([f for f in os.listdir(papers_dir) if f.lower().endswith('.pdf')])

# First 6 papers only
for fname in fnames[:6]:
    fpath = os.path.join(papers_dir, fname)
    print("=" * 70)
    print(f"FILE: {fname}")
    print("=" * 70)
    try:
        reader = PyPDF2.PdfReader(fpath)
        text = ""
        for page_num in range(min(2, len(reader.pages))):
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                text += page_text
        print(text[:1200])
    except Exception as e:
        print(f"ERROR: {e}")
    print("\n")
