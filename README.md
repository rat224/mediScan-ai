README (Simplified Explanation)
MediScan.AI – Lab & Hospital Report Analyzer

MediScan.AI is a smart healthcare tool that reads, understands, and translates medical reports.
It supports both Lab Reports (like CBC, glucose, cholesterol, etc.) and Hospital Reports (like diagnosis, prescription, discharge summary).

It uses OCR (Optical Character Recognition) to extract text from PDF and image reports.
After extracting the text, it analyzes it, shows results clearly, and can translate the full report into 9 Indian languages.

What the Project Does
1. Upload Report

You can upload:

PDF files

JPG / PNG images

The system automatically converts PDF pages into images and prepares them for extraction.

2. OCR Extraction

The app uses Tesseract OCR to read the text from the report.
It removes noise, improves quality using OpenCV, and extracts clear text.

3. Lab Report Analysis

If the report is a Lab Report, the app:

Finds common medical values like Hemoglobin, WBC, Platelets, Glucose, etc.

Compares them with normal reference ranges

Shows status:

Normal

High

Low

Displays results in a clean table

Draws a colored bar chart for better understanding

Saves all results automatically as a CSV file

4. Hospital Report Summary

If the report is a Hospital Report, the app:

Detects sections like Diagnosis, Prescription, Doctor Advice, Discharge Summary

Shows the full text clearly

Saves the summary and extracted text as a TXT file

5. Multi-Language Translation

The app can translate the entire report into 9 Indian languages:

Kannada

Hindi

Marathi

Tamil

Telugu

Malayalam

Gujarati

Bengali

Punjabi

You can choose one language or translate to all languages at once.

Each translated file is saved in the output_reports folder.

6. PPT Generation

The app can automatically create a simple PowerPoint file containing:

Original text

Downloadable slides

Useful for Unstop submission.

Tools Used

Streamlit (Web UI)

Python

Tesseract OCR (for text extraction)

OpenCV (image preprocessing)

pdf2image + Poppler (PDF conversion)

Matplotlib (visual charts)

Pandas (data handling)

googletrans / deep-translator (translations)

Why This Project Is Useful

Helps patients easily understand medical reports

Supports Indian languages (local language medical interpretation)

Saves time for doctors & hospitals

Works offline except translation

Useful for telemedicine & digital healthcare platforms

How to Run
pip install -r requirements.txt
streamlit run app.py

Folder Structure
app.py
requirements.txt
assets/
output_reports/
README.md

Conclusion

MediScan.AI is a complete, user-friendly solution for analyzing medical reports, showing insights, and providing translations in local languages.
It is perfect for hackathons, academic projects, and real-world healthcare use.
