# DocuMaster Pro: Exam Pipeline ETL & Data Validator

## Overview
This repository contains a robust data extraction and reconciliation tool developed to manage large-scale university examination pipelines. It is designed to bridge the gap between unstructured physical assets (PDFs) and structured SQL databases by extracting, cleaning, and validating multi-lingual exam data.


## Technical Architecture & Core Logic
This tool acts as an automated ETL (Extract, Transform, Load) pipeline with a heavy focus on Data Quality and Observability.

* **Hybrid Data Extraction (OCR):** Utilizes `pdfplumber` and `PyMuPDF` for structural grid mapping, combined with a dual-brain OCR system (`RapidOCR` for English/Math and `EasyOCR` for Bengali script) to extract text from heavily formatted tables.
* **Advanced Deduplication (4-Pass Logic):** Implements `rapidfuzz` for strict string matching and space-erasure logic to identify data entry errors and option-level conflicts.
* **Semantic AI Scanning:** Integrates HuggingFace's `SentenceTransformers` (`paraphrase-multilingual-MiniLM-L12-v2` and `l3cube-pune/bengali-sentence-bert-nli`) to perform deep semantic conflict resolution on isolated "orphan" questions.
* **Automated Audit Reporting:** Generates comprehensive Excel reports (`xlsxwriter`) detailing processing logs, assigned UIDs, and True ID Conflicts to ensure 100% downstream data integrity.

## Key Features
* **Dynamic Grid Enforcement:** Bypasses merged-cell errors in PDFs by projecting virtual X-Y grids.
* **Smart Stitching:** Automatically identifies and merges fragmented data rows across multiple pages.
* **Conflict Resolution Engine:** Tracks the exact original ID and page location of duplicate questions to warn database administrators of "True Conflicts."
* **GUI Control Panel:** Built with `tkinter` to allow non-technical operators to configure batch processing and manage metadata stamping.

## Tech Stack
* **Language:** Python
* **Data Processing:** NumPy, RapidFuzz
* **Machine Learning / NLP:** PyTorch, SentenceTransformers, EasyOCR, RapidOCR
* **Document Parsing:** PyMuPDF (fitz), pdfplumber
