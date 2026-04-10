import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import threading
import re
import pdfplumber
import fitz  # PyMuPDF
import xlsxwriter
from rapidfuzz import fuzz
from datetime import datetime

# --- EASYOCR LIBRARIES ---
import easyocr

#-----RAPIDOCR LIBRARIES ----
from rapidocr_onnxruntime import RapidOCR
import numpy as np

#--- NLP AI Libraries-----
from sentence_transformers import SentenceTransformer, util
import torch

# ==========================================
# FIX FOR PyInstaller '--windowed' isatty CRASH
# ==========================================
class DummyOutput:
    def write(self, x): pass
    def flush(self): pass
    def isatty(self): return False

if sys.stdout is None:
    sys.stdout = DummyOutput()
if sys.stderr is None:
    sys.stderr = DummyOutput()
# ==========================================

# ==========================================
#      EXE RESOURCE HELPER
# ==========================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
#      CONFIGURATION
# ==========================================
OFFSET_Y = 15                 
PN_MARGIN_RIGHT = 40          
PN_MARGIN_BOTTOM = 22  
PN_FONT_SIZE = 11
PN_COLOR = (0, 0, 0)          

# ==========================================
#      HELPER FUNCTIONS
# ==========================================

def normalize(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    # Allow Bengali Range \u0980-\u09FF
    text = re.sub(r'[^\w\s\+\-\*\/\=\^\∫\∑\.\%\(\)\[\]\{\}\u0980-\u09FF]', '', text)
    return text.strip()

def perform_ocr_on_cell(rapid_reader, easy_reader, page_obj, cell_bbox, lang_mode="EN"):
    try:
        if not cell_bbox: return ""
        cropped = page_obj.crop(cell_bbox)

        # 1. Extract raw PIL image & convert to NumPy (300 DPI)
        pil_img = cropped.to_image(resolution=300).original
        img_np = np.array(pil_img)

        # 🚫 OPENCV COMPLETELY BYPASSED 🚫
        # We are handing the raw, full-color, untouched 300 DPI image straight to the brains.
        final_img = img_np

        # ==========================================
        # 4. THE TWO-BRAIN ROUTER
        # ==========================================
        if lang_mode == "EN":
            # 🧠 BRAIN 1: RapidOCR for English / Science / Math
            result, _ = rapid_reader(final_img)
            if result:
                texts = [line[1] for line in result]
                return " ".join(texts)
            return ""
            
        elif lang_mode == "BN":
            # 🧠 BRAIN 2: EasyOCR for Bengali
            result_list = easy_reader.readtext(
                final_img, 
                detail=0, 
                paragraph=True,
                mag_ratio=1.0,         
                contrast_ths=0.0,      
                adjust_contrast=0.0    
            )
            return " ".join(result_list)
            
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
    

def find_duplicates(questions, threshold=83, fallback_min=65):
    """
    3-PASS LOGIC (RESTORED):
    1. Strict Grouping (No Options logic) -> Creates clean groups.
    2. Orphan -> Group (WIPO Logic) -> Adds tricky duplicates to existing groups.
    3. Orphan -> Orphan (WIPO Logic) -> Groups remaining tricky pairs.
    """
    if not questions: return []
    
    n = len(questions)
    visited = [False] * n
    groups = []

    # --- PRE-CALCULATE STRINGS ---
    for q in questions:
        q["norm_q"] = normalize(q.get("Question", ""))
        q["sorted_norm"] = " ".join(sorted(q["norm_q"].split()))
        q["spaceless_q"] = q["norm_q"].replace(" ", "")
        
        # Combined Context (Q + Options)
        combined = f"{q.get('Question','')} {q.get('A','')} {q.get('B','')} {q.get('C','')} {q.get('D','')}"
        q["norm_combined"] = normalize(combined)
        q["sorted_combined"] = " ".join(sorted(q["norm_combined"].split()))
        
        # Options Only (For WIPO Logic)
        options_only = f"{q.get('A','')} {q.get('B','')} {q.get('C','')} {q.get('D','')}"
        q["norm_options"] = normalize(options_only)
        q["sorted_options"] = " ".join(sorted(q["norm_options"].split()))
        q["spaceless_options"] = q["norm_options"].replace(" ", "")

    # =========================================================
    # PASS 1: STRICT GROUPING (Questions & Combined Only)
    # =========================================================
    for i in range(n):
        if visited[i]: continue
        q1 = questions[i]
        if len(q1["norm_q"]) < 3: continue

        current_group = []
        
        for j in range(i + 1, n):
            if visited[j]: continue
            q2 = questions[j]
            if len(q2["norm_q"]) < 3: continue
            
            # SCORING (Strict Only)
            q_score = fuzz.ratio(q1["sorted_norm"], q2["sorted_norm"])
            opts_score = fuzz.ratio(q1["sorted_options"], q2["sorted_options"]) 
            
            is_match = False
            final_score = q_score

            # Rule A: Strict Question Match
            if q_score >= threshold:
                if opts_score >= 40:
                    is_match = True

            # Rule A.2: The Space Eraser (Glued Word Fallback) 
            elif fuzz.ratio(q1["spaceless_q"], q2["spaceless_q"]) >= threshold:
                is_match = True
                final_score = fuzz.ratio(q1["spaceless_q"], q2["spaceless_q"])

            # Rule B: Fallback (Combined Context)
            elif q_score >= fallback_min:
                combined_score = fuzz.ratio(q1["sorted_combined"], q2["sorted_combined"])
                if combined_score >= threshold:
                    is_match = True
                    final_score = combined_score

            if is_match:
                if not current_group:
                    current_group.append((q1, "BASE"))
                    visited[i] = True
                current_group.append((q2, f"{int(final_score)}%"))
                visited[j] = True
        
        if current_group:
            groups.append(current_group)

    # =========================================================
    # PASS 2: ORPHAN -> EXISTING GROUP CHECK
    # =========================================================
    for i in range(n):
        if visited[i]: continue 
        orphan = questions[i]
        if len(orphan["norm_q"]) < 3: continue

        for group in groups:
            base_q = group[0][0] 
            
            q_score = fuzz.ratio(orphan["sorted_norm"], base_q["sorted_norm"])
            opts_score = fuzz.ratio(orphan["sorted_options"], base_q["sorted_options"])

            token_opts = fuzz.token_set_ratio(orphan["norm_options"], base_q["norm_options"])

            # THE SPACE ERASERS (Bypasses pdfplumber glued-text bugs)
            spaceless_q_score = fuzz.ratio(orphan["spaceless_q"], base_q["spaceless_q"])
            spaceless_opts_score = fuzz.ratio(orphan["spaceless_options"], base_q["spaceless_options"])

            # ==========================================
            # THE CLEAN DATA DUPLICATE RULES
            # ==========================================
            is_match = False
            match_type = ""
            
            # RULE 1: The Standard Match
            # Both the Question and Options are highly similar (The most common true duplicate)
            if q_score >= 80 and opts_score >= 80:
                is_match = True
                match_type = "Standard Match"

            # RULE 2: The WIPO/Long Option Match
            # The OCR slightly messed up the Question, but the Options are long and identical.
            elif (opts_score >= 85 or token_opts >= 85 or spaceless_opts_score >= 85) and len(orphan["norm_options"]) > 40 and (q_score >= 60 or spaceless_q_score >= 60):
                is_match = True
                match_type = "Long Options Match"

            # RULE 3: The Short/Date Match (Years, "1918 1918", short names)
            # Options are very short, so we demand a flawless option match and a decent question match.
            elif token_opts >= 80 and len(orphan["norm_options"]) <= 40 and q_score >= 60: 
                is_match = True
                match_type = "Short Exact Match"

            if is_match:
                group.append((orphan, f"Matched via: {match_type}"))
                visited[i] = True
                break

    # =========================================================
    # PASS 3: ORPHAN -> ORPHAN CHECK
    # =========================================================
    for i in range(n):
        if visited[i]: continue
        q1 = questions[i]
        if len(q1["norm_q"]) < 3: continue

        current_group = []
        
        for j in range(i + 1, n):
            if visited[j]: continue
            q2 = questions[j]
            if len(q2["norm_q"]) < 3: continue
            
            # We only calculate what we need for the Clean Data Era!
            # (Notice token_comb is completely gone, saving processing power)
            q_score = fuzz.ratio(q1["sorted_norm"], q2["sorted_norm"])
            opts_score = fuzz.ratio(q1["sorted_options"], q2["sorted_options"])
            token_opts = fuzz.token_set_ratio(q1["norm_options"], q2["norm_options"])

            # THE SPACE ERASERS
            spaceless_q_score = fuzz.ratio(q1["spaceless_q"], q2["spaceless_q"])
            spaceless_opts_score = fuzz.ratio(q1["spaceless_options"], q2["spaceless_options"])
                            

            # ==========================================
            # THE CLEAN DATA DUPLICATE RULES
            # ==========================================
            is_match = False
            match_type = ""

            # RULE 1: The Standard Match
            if q_score >= 80 and opts_score >= 80:
                is_match = True
                match_type = "Standard Match"

            # RULE 2: The Long Options Match
            elif (opts_score >= 85 or token_opts >= 85 or spaceless_opts_score >= 85) and len(q1["norm_options"]) > 40 and (q_score >= 60 or spaceless_q_score >= 60):
                is_match = True
                match_type = "Long Options Match"

            # RULE 3: The Short Exact Match
            elif token_opts >= 80 and len(q1["norm_options"]) <= 40 and q_score >= 60: 
                is_match = True
                match_type = "Short Exact Match"

            # ==========================================
            # PASS 3 SPECIFIC GROUPING LOGIC
            # ==========================================
            if is_match:
                # If this is the first match, make q1 the BASE of a new group
                if not current_group:
                    current_group.append((q1, "BASE"))
                    visited[i] = True
                
                # Add q2 to the group
                current_group.append((q2, f"Matched via: {match_type}"))
                visited[j] = True
        
        # If we built a new group, save it to the master list
        if current_group:
            groups.append(current_group)

    return groups


def apply_nlp_hybrid_pass(duplicate_groups, all_questions, nlp_model, log_callback, threshold=0.80):
    """
    Pass 4: The Deep Semantic Scan (All-to-All Matrix).
    Vectorizes EVERY question. If any two questions are semantically identical, 
    it forcefully merges their respective groups.
    """
    if nlp_model is None:
        return duplicate_groups 

    log_callback("\n🧠 Starting Deep Semantic Scan (All-to-All)...")
    
    try:
        # 1. Create a map of where every question currently lives
        group_map = {}
        for idx, group in enumerate(duplicate_groups):
            for q_tuple in group:
                group_map[q_tuple[0].get("OrigID")] = idx

        # 2. Vectorize ALL questions at once
        texts = [q.get("Question", "") for q in all_questions]
        log_callback(f"⚙️ Vectorizing {len(texts)} questions for full matrix matching...")
        embeddings = nlp_model.encode(texts, convert_to_tensor=True)

        # 3. Calculate Cosine Similarity for every combination
        cosine_scores = util.cos_sim(embeddings, embeddings)

        merges_needed = []
        n = len(all_questions)
        
        # 4. Find matches and determine action
        for i in range(n):
            for j in range(i + 1, n):
                score = cosine_scores[i][j].item()
                
                if score >= threshold:
                    qA = all_questions[i]
                    qB = all_questions[j]
                    idA = qA.get("OrigID")
                    idB = qB.get("OrigID")
                    
                    gA = group_map.get(idA)
                    gB = group_map.get(idB)
                    
                    # Case A: They are in different groups -> Merge the groups!
                    if gA is not None and gB is not None and gA != gB:
                        merges_needed.append((gA, gB, score))
                        
                    # Case B: One is in a group, one is an Orphan -> Add to group
                    elif gA is not None and gB is None:
                        duplicate_groups[gA].append((qB, f"AI Orphan: {int(score*100)}%"))
                        group_map[idB] = gA 
                        
                    elif gB is not None and gA is None:
                        duplicate_groups[gB].append((qA, f"AI Orphan: {int(score*100)}%"))
                        group_map[idA] = gB
                        
                    # Case C: Both are Orphans -> Create a new group
                    elif gA is None and gB is None:
                        new_group = [
                            (qA, "BASE"),
                            (qB, f"AI: {int(score*100)}%")
                        ]
                        duplicate_groups.append(new_group)
                        new_idx = len(duplicate_groups) - 1
                        group_map[idA] = new_idx
                        group_map[idB] = new_idx

        # 5. Execute Group Merges (If Q27 and Q138 bridged two groups!)
        if merges_needed:
            for gA, gB, score in merges_needed:
                # Ensure they haven't already been merged in a previous loop step
                if not duplicate_groups[gA] or not duplicate_groups[gB]: 
                    continue
                
                target = min(gA, gB)
                source = max(gA, gB)
                
                for q_tuple in duplicate_groups[source]:
                    duplicate_groups[target].append((q_tuple[0], f"AI Group Merge: {int(score*100)}%"))
                
                duplicate_groups[source] = [] # Empty the old group

        # Clean up empty groups
        final_groups = [g for g in duplicate_groups if len(g) > 0]
        log_callback("✅ Semantic Scan Complete!")
        return final_groups

    except Exception as e:
        log_callback(f"⚠️ Semantic Scan encountered an error: {e}. Proceeding with previous state.")
        return duplicate_groups

def save_audit_log(audit_data, duplicate_groups, conflicts, pdf_filename, log_callback, settings, extracted_qs, original_files):
    report_filename = os.path.splitext(pdf_filename)[0] + "_Report.xlsx"
    
    try:
        workbook = xlsxwriter.Workbook(report_filename)

        # Define the header style once
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})

        # # ---> NEW: SHEET 0: ALL EXTRACTED DATA <---
        # if extracted_qs:
        #     ws_all = workbook.add_worksheet("All Extracted Data")
        #     all_headers = ["Page", "Original ID", "Assigned UID", "Question", "A", "B", "C", "D", "Ans"]
        #     for col, h in enumerate(all_headers): 
        #         ws_all.write(0, col, h, header_fmt)
            
        #     for r, q in enumerate(extracted_qs, start=1):
        #         ws_all.write(r, 0, q.get("Page", ""))
        #         ws_all.write(r, 1, q.get("OrigID", ""))
        #         ws_all.write(r, 2, q.get("QNo", ""))
        #         ws_all.write(r, 3, q.get("Question", ""))
        #         ws_all.write(r, 4, q.get("A", ""))
        #         ws_all.write(r, 5, q.get("B", ""))
        #         ws_all.write(r, 6, q.get("C", ""))
        #         ws_all.write(r, 7, q.get("D", ""))
        #         ws_all.write(r, 8, q.get("Ans", ""))
        # # ------------------------------------------
        
        # SHEET 1: PROCESSING LOG
        ws_proc = workbook.add_worksheet("Processing Log")
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        headers = ["Page", "Status", "Original ID", "Assigned UID", "Notes"]
        for col, h in enumerate(headers): ws_proc.write(0, col, h, header_fmt)
        for r, entry in enumerate(audit_data, start=1):
            ws_proc.write(r, 0, entry['Page'])
            ws_proc.write(r, 1, entry['Status'])
            ws_proc.write(r, 2, entry['Original ID'])
            ws_proc.write(r, 3, entry['Assigned UID'])
            ws_proc.write(r, 4, entry['Notes'])

        # SHEET 2: ANALYSIS
        if settings['find_dups']:
            ws_dup = workbook.add_worksheet("Analysis")
            head_fmt = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1, 'align':'center'})
            warn_fmt = workbook.add_format({'bg_color': '#FFFFFF', 'font_color': '#9C0006', 'border': 1})
            title_fmt = workbook.add_format({'bold': True, 'font_size': 12})
            sep_fmt = workbook.add_format({'bottom': 1, 'bottom_color': '#808080'})
            center_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

            current_time = datetime.now().strftime("%d-%m-%Y %H:%M")
            total_dups = sum(len(g) for g in duplicate_groups)

            # Count how many Original IDs appear more than once
            qid_counts = {}
            for q in extracted_qs:
                o_id = str(q.get("OrigID", "")).strip()
                if o_id and o_id.lower() != "none":
                    qid_counts[o_id] = qid_counts.get(o_id, 0) + 1
            duplicate_qid_count = sum(1 for count in qid_counts.values() if count > 1)

            # SUMMARY HEADER
            ws_dup.write(0, 0, f"Report: {os.path.basename(pdf_filename)}", title_fmt)
            ws_dup.write(0, 3, f"Date: {current_time}", title_fmt)
            # Print the merged file names
            file_names = ", ".join([os.path.basename(f) for f in original_files])
            ws_dup.write(1, 0, f"Merged Files: {file_names}")

            ws_dup.write(2, 0, f"Total Pages Processed: {len(audit_data)}")
            ws_dup.write(3, 0, f"Duplicates Found: {total_dups} in {len(duplicate_groups)} groups")
            ws_dup.write(4, 0, f"Duplicate QID found : {duplicate_qid_count}")
            
            # Write Conflicts
            row = 6
            if conflicts:
                ws_dup.write(row, 0, "ID CONFLICTS DETECTED:", warn_fmt)
                row += 1
                for conf in conflicts:
                    ws_dup.write(row, 0, conf)
                    row += 1
                row += 1 

            headers_dup = ["Group", "Page", "QID","New QID", "Score"]
            if settings['print_q']: headers_dup.append("Question")
            if settings['print_opts']: headers_dup.extend(["A", "B", "C", "D", "Ans"])
            
            for c, h in enumerate(headers_dup): ws_dup.write(row, c, h, head_fmt)
            row += 1
            
            for idx, group in enumerate(duplicate_groups):
                for q_tuple in group:
                    q, score = q_tuple
                    col = 0
                    ws_dup.write(row, col, f"Group {idx+1}"); col+=1
                    ws_dup.write(row, col, q.get("Page"), center_fmt); col+=1
                    ws_dup.write(row, col, q.get("OrigID"), center_fmt); col+=1
                    ws_dup.write(row, col, q.get("QNo"), center_fmt); col+=1
                    ws_dup.write(row, col, score); col+=1
                    if settings['print_q']: ws_dup.write(row, col, q.get("Question")); col+=1
                    if settings['print_opts']:
                        ws_dup.write(row, col, q.get("A")); col+=1
                        ws_dup.write(row, col, q.get("B")); col+=1
                        ws_dup.write(row, col, q.get("C")); col+=1
                        ws_dup.write(row, col, q.get("D")); col+=1
                        ws_dup.write(row, col, q.get("Ans")); col+=1
                    row += 1
                row += 1
            
            ws_dup.write(row, 0, "___________END Report___________")

        workbook.close()
        log_callback(f"📊 Report saved: {os.path.basename(report_filename)}")
        return True
    except Exception as e:
        log_callback(f"⚠️ Log Error: {e}")
        return False

# ==========================================
#      WORKER LOGIC (FINAL v7)
# ==========================================

def get_output_filename(first_filepath):
    base_name = os.path.basename(first_filepath)
    prefix = base_name.split('_')[0] if "_" in base_name else os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_ALL_{timestamp}.pdf"

def run_processing(files, final_output_path, start_id, settings, log_callback, progress_callback, finish_callback, is_cancelled):
    # --- 1. INITIAL SETUP ---
    temp_filename = "temp_gui_worker.pdf"
    audit_data = []
    extracted_qs = []
    current_q_data = None
    

    # Unpack Settings
    do_stamp = settings['stamp_ids']
    do_pagenum = settings['stamp_pages']
    do_dups = settings['find_dups']
    is_bengali = settings['is_bengali']
    pg_start = settings['pg_start']
    pg_end = settings['pg_end']

    try:
        log_callback("🧠 Initializing EasyEYE (Multilanguage Brain)...")
        ocr_reader_bn = easyocr.Reader(['en', 'bn'], gpu=False) 


        log_callback("⚡ Initializing RapidEYE (Science Brain)...")
        ocr_reader_en = RapidOCR()

    except Exception as e:
        log_callback(f"❌ EYE failed to OPEN:  Init Failed: {e}")
        finish_callback(False)
        return
    
    # ---> 1. Safely load the AI Semantic Brain based on document language
    nlp_model = None
    if do_dups:
        # Determine which brain to load!
        if is_bengali:
            model_name = 'l3cube-pune/bengali-sentence-bert-nli'
            model_folder = "models--l3cube-pune--bengali-sentence-bert-nli"
            log_callback("🧠 Loading Bengali Specialist Brain...")
        else:
            model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
            model_folder = "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
            log_callback("🧠 Loading Multilingual Generalist Brain...")
        
        # Setup Cache Paths
        user_home = os.path.expanduser("~")
        if getattr(sys, 'frozen', False):
            cache_path = os.path.join(user_home, ".EasyOCR", "nlp_brain")
            os.makedirs(cache_path, exist_ok=True)
        else:
            # FOR VS CODE TESTING
            cache_path = os.path.join(user_home, ".EasyOCR", "nlp_brain")

        def is_online():
            import urllib.request
            try:
                urllib.request.urlopen('https://huggingface.co', timeout=2)
                return True
            except:
                return False

        try:
            if not is_online():
                log_callback("⚠️ No internet. Locating physical files for Direct Local Mode...")
                import glob
                
                # Dynamically search for whichever model we selected above
                search_pattern = os.path.join(cache_path, model_folder, "snapshots", "*")
                snapshots = glob.glob(search_pattern)
                
                if snapshots:
                    exact_local_path = snapshots[0] 
                    nlp_model = SentenceTransformer(exact_local_path, local_files_only=True)
                    log_callback("✅ AI Brain Loaded (Direct Local Mode)!")
                else:
                    raise Exception(f"Local files for {model_name} not found in the nlp_brain folder.")
            else:
                nlp_model = SentenceTransformer(model_name, cache_folder=cache_path)
                log_callback("✅ AI Brain Loaded Successfully!")
                
        except Exception as e:
            log_callback(f"⚠️ AI Load Failed: {e}")
            nlp_model = None
    
    # --- 2. CUSTOM FONT SETUP ---
    custom_font_path = resource_path("ShonarBangla.ttf") 
    has_custom_font = os.path.exists(custom_font_path)
    if is_bengali and not has_custom_font:
        log_callback(f"⚠️ Warning: '{custom_font_path}' not found. Using fallback.")

    # --- 3. MERGE FILES ---
    try:
        log_callback("⏳ Merging files...")
        merged_doc = fitz.open()
        for f in files:
            if is_cancelled(): # <--- Kill Switch
                finish_callback(False, None, cancelled=True)
                return
            try:
                doc = fitz.open(f)
                merged_doc.insert_pdf(doc)
                doc.close()
            except Exception as e:
                log_callback(f"❌ Failed to merge {os.path.basename(f)}")
                finish_callback(False)
                return
        merged_doc.save(temp_filename)
        merged_doc.close()
    except Exception as e:
        log_callback(f"❌ Merge Critical Error: {e}")
        finish_callback(False)
        return

    # --- 4. PROCESS PAGES ---
    try:
        doc_write = fitz.open(temp_filename)
        total_pages = len(doc_write)
        
        if pg_end == -1 or pg_end > total_pages: pg_end = total_pages
        if pg_start < 1: pg_start = 1
        
        log_callback(f"🚀 Processing Pages {pg_start} to {pg_end}...")
        log_callback(f"⚠️ MODE: {'PURE Bengali, Slow Extract' if is_bengali else 'FAST TEXT EXTRACT'}")

        previous_old_qid = None
        current_new_id = start_id
        is_first_id_assigned = False 

        with pdfplumber.open(temp_filename) as doc_read:
            for i, page_read in enumerate(doc_read.pages):

                if is_cancelled(): # <--- Kill Switch
                    finish_callback(False, None, cancelled=True)
                    return

                current_page_num = i + 1
                percent = int(((i + 1) / total_pages) * 100)
                progress_callback(percent)

                if current_page_num < pg_start or current_page_num > pg_end:
                    continue

                page_write = doc_write[i]
                
                # --- A. CLEAN PAGE ---
                try: page_write.clean_contents()
                except: pass

                # --- B. REGISTER FONT ---
                if is_bengali and has_custom_font:
                    try: page_write.insert_font(fontname="sb", fontfile=custom_font_path)
                    except: pass

                # --- C. STAMP PAGE NUMBERS (Hybrid) ---
                if do_pagenum:
                    try:
                        page_num_str = f"Page {current_page_num}"
                        try:
                            # Try High Quality
                            pn_font = "tibo"
                            text_len = fitz.get_text_length(page_num_str, fontname=pn_font, fontsize=PN_FONT_SIZE)
                            x = page_write.rect.width - PN_MARGIN_RIGHT - text_len
                            y = page_write.rect.height - PN_MARGIN_BOTTOM
                            page_write.insert_text((x, y), page_num_str, fontsize=PN_FONT_SIZE, fontname=pn_font, color=PN_COLOR)
                        except:
                            # Fallback Safe
                            pn_font = "helv"
                            text_len = len(page_num_str) * (PN_FONT_SIZE * 0.55)
                            x = page_write.rect.width - PN_MARGIN_RIGHT - text_len
                            y = page_write.rect.height - PN_MARGIN_BOTTOM
                            page_write.insert_text((x, y), page_num_str, fontsize=PN_FONT_SIZE, fontname=pn_font, color=PN_COLOR)
                    except: pass 

                # --- D. FIND TABLE & ID ---
                log_entry = {'Page': current_page_num, 'Status': '', 'Original ID': '', 'Assigned UID': '', 'Notes': ''}
                
                try: found_tables = page_read.find_tables()
                except: found_tables = []
                
                if not found_tables:
                    log_entry['Status'] = "Skipped"
                    log_entry['Notes'] = "No table found"
                    audit_data.append(log_entry)
                    continue 
                
                target_table = found_tables[0]
                table_content = target_table.extract()
                extracted_qid = None
                
                if table_content and len(table_content) >= 1:
                    try: extracted_qid = str(table_content[0][1] or "").strip()
                    except: pass
                
                log_entry['Original ID'] = extracted_qid if extracted_qid else "None"
                
                # --- ID ASSIGNMENT ---
                active_id_str = None
                if do_stamp:
                    if extracted_qid:
                        if extracted_qid == previous_old_qid:
                            log_entry['Status'] = "Continuation"
                            active_id_str = str(current_new_id)
                        else:
                            if not is_first_id_assigned: is_first_id_assigned = True
                            else: current_new_id += 1
                            previous_old_qid = extracted_qid
                            log_entry['Status'] = "New ID"
                            active_id_str = str(current_new_id)
                else:
                    log_entry['Status'] = "Existing ID Used"
                    active_id_str = extracted_qid

                # --- E. STAMP QID (Hybrid) ---
                if active_id_str:
                    log_entry['Assigned UID'] = active_id_str
                    if do_stamp:
                        try:
                            x0, top, x1, bottom = target_table.bbox
                            text_to_print = f"New QID: {active_id_str}"
                            fs_id = 18
                            
                            font_to_use = "tibo"
                            if is_bengali and has_custom_font: font_to_use = "sb"
                            
                            try:
                                id_text_len = fitz.get_text_length(text_to_print, fontname=font_to_use, fontsize=fs_id)
                                nx = x1 - id_text_len
                                ny = top - OFFSET_Y
                                if ny < 20: ny = 20
                                page_write.insert_text((nx, ny), text_to_print, fontsize=fs_id, fontname=font_to_use, color=(0,0,0))
                            except:
                                font_to_use = "helv"
                                id_text_len = len(text_to_print) * (fs_id * 0.55)
                                nx = x1 - id_text_len
                                ny = top - OFFSET_Y
                                if ny < 20: ny = 20
                                page_write.insert_text((nx, ny), text_to_print, fontsize=fs_id, fontname=font_to_use, color=(0,0,0))
                        except: pass

                    # --- F. TWO-PASS GRID ENFORCEMENT EXTRACTION ---
                    try:
                        raw_q = ""
                        opts = ["", "", "", ""] # [A, B, C, D]
                        ocr_announced = False
                        
                        
                        # ==========================================
                        # PHASE 1: THE CARTOGRAPHER (Build the Virtual Grid)
                        # ==========================================
                        grid_map = {}
                        
                        # [BENGALI MODE ONLY] - Step 1: Find Permanent X-Walls
                        global_v_x0 = None
                        global_v_x1 = None
                        if is_bengali and target_table.rows:
                            # Left Wall: Right edge of the header column
                            global_v_x0 = target_table.rows[0].cells[0][2] 
                            # Right Wall: Absolute right edge of the entire table
                            global_v_x1 = target_table.bbox[2]             
                        
                        for row_idx, row in enumerate(target_table.rows):
                            cells = row.cells
                            # RELAXED GUARD: Allows processing even if right cell is missing due to severe merging
                            if not cells: continue 
                            
                            header_bbox = cells[0]
                            if not header_bbox: continue
                            
                            # Read the header text
                            header_text = page_read.crop(header_bbox).extract_text() or ""
                            clean_header = header_text.upper().replace("_", "").replace(" ", "").replace("\n", "")
                            
                            # Identify the row type
                            valid_keys = ["QEN", "QBN", "OPTAEN", "OPTABN", "OPTBEN", "OPTBBN", "OPTCEN", "OPTCBN", "OPTDEN", "OPTDBN"]
                            matched_key = None
                            for k in valid_keys:
                                if k in clean_header or clean_header == k.replace("OPT", ""):
                                    matched_key = k
                                    break
                            
                            if matched_key:
                                _, h_top, _, h_bottom = header_bbox
                                
                                # 💡 THE "LOOK-AHEAD" FIX: Seal the bottom boundary using the next header
                                if row_idx + 1 < len(target_table.rows):
                                    next_header = target_table.rows[row_idx + 1].cells[0]
                                    if next_header:
                                        h_bottom = next_header[1] # Set bottom to the top of the next row
                                
                                # 🛡️ FATAL MERGE INTERCEPTOR: Runs for BOTH modes to ensure perfect vertical slices
                                if "EN" in clean_header and "BN" in clean_header:
                                    for w in page_read.crop(header_bbox).extract_words():
                                        if "BN" in w['text'].upper():
                                            h_bottom = w['top'] - 2
                                            break

                                # ==========================================
                                # THE BRANCHING PATH (English vs Bengali)
                                # ==========================================
                                if is_bengali:
                                    # 🟢 BENGALI: THE LEFT-BOX PROJECTION (Immune to Merged Cells)
                                    if global_v_x0 is not None and global_v_x1 is not None:
                                        grid_map[matched_key] = (global_v_x0, max(0, h_top - 2), global_v_x1, h_bottom)
                                else:
                                    # 🔵 ENGLISH: THE STANDARD CELL READER
                                    value_cells = [c for c in cells[1:] if c]
                                    if value_cells:
                                        v_x0 = value_cells[0][0]
                                        v_x1 = value_cells[-1][2]
                                        grid_map[matched_key] = (v_x0, max(0, h_top - 2), v_x1, h_bottom)
                        # ==========================================
                        # PHASE 2: THE READER (Extract using the Grid)
                        # ==========================================
                        
                        def safe_glue(bucket_text, new_text):
                            if not new_text: return bucket_text
                            # Prevents exact duplicate options (e.g., 1918 + 1918)
                            if new_text.strip().lower() in bucket_text.strip().lower():
                                return bucket_text
                            if bucket_text: return bucket_text + " " + new_text
                            return new_text

                        # Define exactly what we want to read based on the toggle
                        keys_to_read = ["QEN", "OPTAEN", "OPTBEN", "OPTCEN", "OPTDEN"]
                        if is_bengali:
                            keys_to_read.extend(["QBN", "OPTABN", "OPTBBN", "OPTCBN", "OPTDBN"])

                        for key in keys_to_read:
                            if key not in grid_map: continue
                            bbox = grid_map[key]
                            row_text = ""

                            if is_bengali and "BN" in key:
                                # 🧠 SLOW MODE: THE TWO-BRAIN OCR
                                log_entry['Notes'] += " [OCR BN]"
                                row_text = perform_ocr_on_cell(ocr_reader_en, ocr_reader_bn, page_read, bbox, "BN")
                                
                            else:
                                # ⚡ FAST MODE: PURE TEXT + VAPORIZER
                                raw_extracted = page_read.crop(bbox).extract_text() or ""
                                raw_extracted = raw_extracted.replace('\n', ' ').strip()
                                
                                # Instantly vaporize non-English characters and garbage
                                clean_text = re.sub(r'[^\x00-\x7F]+', ' ', raw_extracted)
                                row_text = re.sub(r'\s+', ' ', clean_text).strip()
                                
                                # 🛡️ ECHO CANCELER (Fixes PDF "Fake Bold" repeating text)
                                if row_text:
                                    words = row_text.split()
                                    row_text = " ".join([words[i] for i in range(len(words)) if i == 0 or words[i].lower() != words[i-1].lower()])
                                    
                                # Safety Net: If Fast Mode fails completely, try OCR for English
                                if not row_text or len(normalize(row_text)) < 3:
                                    if not ocr_announced:
                                        log_callback(f"➔ Eye Open on Page {current_page_num}...")
                                        ocr_announced = True
                                    ocr_text = perform_ocr_on_cell(ocr_reader_en, ocr_reader_bn, page_read, bbox, "EN")
                                    if ocr_text:
                                        ocr_clean = re.sub(r'[^\x00-\x7F]+', ' ', ocr_text)
                                        row_text = re.sub(r'\s+', ' ', ocr_clean).strip()

                            # === ROUTE THE TEXT TO THE CORRECT BUCKET ===
                            if row_text:
                                if "QEN" in key or "QBN" in key: raw_q = safe_glue(raw_q, row_text)
                                elif "OPTA" in key: opts[0] = safe_glue(opts[0], row_text)
                                elif "OPTB" in key: opts[1] = safe_glue(opts[1], row_text)
                                elif "OPTC" in key: opts[2] = safe_glue(opts[2], row_text)
                                elif "OPTD" in key: opts[3] = safe_glue(opts[3], row_text)

                        raw = raw_q.strip()
                        opts = [o.strip() for o in opts]

                        is_continuation = False
                        if current_q_data and extracted_qid == current_q_data.get("OrigID"):
                            has_all_opts = all(current_q_data.get(k) for k in ["A", "B", "C", "D"])
                            if not has_all_opts:
                                is_continuation = True

                        # 2. SMART STITCHING & DATA STRUCTURING
                        
                        if is_continuation:
                            if raw and raw not in current_q_data["Question"]: 
                                current_q_data["Question"] += " " + raw
                            # Fill in any missing options
                            if opts[0] and not current_q_data["A"]: current_q_data["A"] = opts[0]
                            if opts[1] and not current_q_data["B"]: current_q_data["B"] = opts[1]
                            if opts[2] and not current_q_data["C"]: current_q_data["C"] = opts[2]
                            if opts[3] and not current_q_data["D"]: current_q_data["D"] = opts[3]

                            log_entry['Notes'] += " [Stitched to Prev Page]"

                        else:
                            if current_q_data: extracted_qs.append(current_q_data)
                            
                            ans_key = ""
                            try:
                                if len(table_content) > 0 and len(table_content[0]) > 3:
                                    ans_key = str(table_content[0][3] or "").strip()
                            except: pass

                            current_q_data = {
                                "Page": current_page_num, "QNo": active_id_str, "OrigID": extracted_qid,
                                "Question": raw, "A":opts[0], "B":opts[1], "C":opts[2], "D":opts[3], "Ans": ans_key
                            }

                            log_entry['Notes'] += " [New Q Captured]"

                    except Exception as ex:
                        log_entry['Notes'] = f"Ext Error: {ex}"

                audit_data.append(log_entry)

        if current_q_data: extracted_qs.append(current_q_data)

        # 3. SAVE FINAL PDF & REPORT
        try:
            doc_write.save(final_output_path)
            log_callback(f"✅ Saved PDF: {os.path.basename(final_output_path)}")

            # # ==========================================
            # # 🛠️ DEBUG DUMP: SEE TEXT BEFORE DUP CHECK
            # # ==========================================
            # try:
            #     debug_path = os.path.join(os.path.dirname(final_output_path), "DEBUG_Extracted_Text.txt")
            #     with open(debug_path, "w", encoding="utf-8") as f:
            #         for idx, q in enumerate(extracted_qs):
            #             f.write(f"--- QID: {q.get('OrigID')} | Assigned: {q.get('QNo')} ---\n")
            #             f.write(f"Question: {q.get('Question')}\n")
            #             f.write(f"A: {q.get('A')}\n")
            #             f.write(f"B: {q.get('B')}\n")
            #             f.write(f"C: {q.get('C')}\n")
            #             f.write(f"D: {q.get('D')}\n\n")
            #     log_callback("🛠️ Debug file saved: DEBUG_Extracted_Text.txt")
            # except Exception as e:
            #     log_callback(f"⚠️ Debug dump failed: {e}")
            # # ==========================================
            
            if do_dups:
                log_callback("🔍 Analyzing duplicates (4-Pass Logic)...")
                dups = find_duplicates(extracted_qs)
                # --->Run the NLP Semantic logic right after!
                if nlp_model is not None:
                    dups = apply_nlp_hybrid_pass(dups, extracted_qs, nlp_model, log_callback, threshold=0.85)
                else:
                    log_callback("⏩ Skipping AI Semantic Scan (Relying purely on structural math).")
                
                # ==========================================
                # THE "TRUE CONFLICT" GUARD (With Page Numbers)
                # ==========================================
                true_conflicts = []
                id_tracker = {}
                
                # 1. Tag grouped questions with a temporary Group ID
                for group_idx, group in enumerate(dups):
                    for q_tuple in group:
                        q_dict = q_tuple[0]  
                        q_dict['temp_group_id'] = f"Group {group_idx + 1}"
                        
                # 2. Track where every Original ID ends up AND its Page Number
                for idx, q in enumerate(extracted_qs):
                    orig_id = str(q.get("OrigID", "")).strip()  
                    page_num = str(q.get("Page", "")).strip()
                    
                    if orig_id and orig_id.lower() != "none":
                        # We use 'idx' internally to keep the conflict math perfectly accurate
                        assigned_group = q.get('temp_group_id', f"Ungrouped_{idx}")
                        
                        if orig_id not in id_tracker:
                            id_tracker[orig_id] = []
                            
                        # Store both the group and the page number together
                        id_tracker[orig_id].append({
                            "group": assigned_group,
                            "page": page_num
                        })
                        
                # 3. Apply the "Final Verdict" Logic
                for orig_id, instances in id_tracker.items():
                    if len(instances) > 1: 
                        # Did they end up in different groups?
                        unique_groups = set([inst["group"] for inst in instances])
                        
                        if len(unique_groups) > 1:
                            # 1. Gather and sort the page numbers
                            pages = sorted(list(set([inst["page"] for inst in instances])), key=lambda x: int(x) if x.isdigit() else x)
                            page_str = " & ".join(pages)
                            
                            # 2. Clean up the group names for the final Excel printout
                            clean_locations = set()
                            for loc in unique_groups:
                                if loc.startswith("Ungrouped_"):
                                    clean_locations.add("Isolated Question")
                                else:
                                    clean_locations.add(loc)
                            group_str = ", ".join(sorted(clean_locations))
                            
                            # 3. Build the final formatted message
                            msg = f"⚠ TRUE CONFLICT: Original ID '{orig_id}' has different questions! (Found on Page: {page_str} | In: {group_str})"
                            true_conflicts.append(msg)
                # ==========================================
                

                # Send the true_conflicts to the Excel writer
                save_audit_log(audit_data, dups, true_conflicts, final_output_path, log_callback, settings, extracted_qs, files)
            else:
                log_callback("⏩ Skipping Duplicate Check.")
                save_audit_log(audit_data, [], [], final_output_path, log_callback, settings, extracted_qs, files)
            
            finish_callback(True, current_new_id)
        except PermissionError:
            log_callback("❌ Error: The file is currently OPEN.")
            finish_callback(False)
        
        doc_write.close()

    except Exception as e:
        log_callback(f"❌ Critical Error: {e}")
        finish_callback(False)
    finally:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except: pass

# ==========================================
#              GUI APPLICATION
# ==========================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DocuMaster Pro")
        self.geometry("650x700")
        
        self.selected_files = []

        self.lbl_credits = tk.Label(self, text="Built by Siddhartha Debnath (D05)", 
                                    font=("Arial", 9, "italic"), fg="#666666", pady=5)
        self.lbl_credits.pack(side="bottom", fill="x")
        
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.page_setup = tk.Frame(self.container)
        self.page_process = tk.Frame(self.container)
        self.page_setup.grid(row=0, column=0, sticky="nsew")
        self.page_process.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.build_setup_page()
        self.build_process_page()
        self.show_setup()

    def show_setup(self): self.page_setup.tkraise()
    def show_process(self): self.page_process.tkraise()

    def build_setup_page(self):
        tk.Label(self.page_setup, text="Step 1: Select Files", font=("Arial", 14, "bold"), pady=10).pack()
        
        frame_list = tk.Frame(self.page_setup, padx=20)
        frame_list.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(frame_list, selectmode=tk.EXTENDED, font=("Arial", 10), height=6)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(frame_list, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        frame_btns = tk.Frame(self.page_setup, pady=5)
        frame_btns.pack(fill="x")
        tk.Button(frame_btns, text="+ Add Files", command=self.add_files, width=15).pack(side="left", padx=20)
        tk.Button(frame_btns, text="Clear List", command=self.clear_files, width=15).pack(side="left", padx=5)

        frame_set = tk.LabelFrame(self.page_setup, text="Control Panel", padx=10, pady=10)
        frame_set.pack(fill="x", padx=20, pady=10)

        frame_toggles = tk.Frame(frame_set)
        frame_toggles.pack(fill="x", pady=5)
        
        self.var_ben = tk.BooleanVar(value=False)
        self.var_stamp = tk.BooleanVar(value=True)
        self.var_dups = tk.BooleanVar(value=True)
        self.var_pagenum = tk.BooleanVar(value=True)

        self.btn_ben = tk.Checkbutton(frame_toggles, text="Is Bengali PDF?", variable=self.var_ben, 
                                      indicatoron=False, selectcolor="#90EE90", width=15, height=2)
        self.btn_stamp = tk.Checkbutton(frame_toggles, text="Stamp Unique IDs", variable=self.var_stamp, 
                                        indicatoron=False, selectcolor="#90EE90", width=15, height=2)
        self.btn_dups = tk.Checkbutton(frame_toggles, text="Find Duplicates", variable=self.var_dups, 
                                       indicatoron=False, selectcolor="#90EE90", width=15, height=2)
        self.btn_pagenum = tk.Checkbutton(frame_toggles, text="Stamp Page No.", variable=self.var_pagenum, 
                                       indicatoron=False, selectcolor="#90EE90", width=15, height=2)
        
        self.btn_ben.pack(side="left", padx=5)
        self.btn_pagenum.pack(side = "left", padx=5)
        self.btn_stamp.pack(side="left", padx=5)
        self.btn_dups.pack(side="left", padx=5)

        frame_qid = tk.Frame(frame_set)
        frame_qid.pack(fill="x", pady=(10, 5))
        
        tk.Label(frame_qid, text="Last QID:", font=("Arial",9)).pack(side="left")
        self.entry_last_qid = tk.Entry(frame_qid, width=8)
        self.entry_last_qid.pack(side="left", padx=(2, 10))

        tk.Label(frame_qid, text="Round off:", font=("Arial", 9)).pack(side="left")
        self.entry_round = tk.Entry(frame_qid, width=4)
        self.entry_round.pack(side="left", padx=(2, 10))
        self.entry_round.insert(0, "10") # Default

        # The Generate Button
        tk.Button(frame_qid, text="Generate ➔", font=("Arial", 8, "bold"), bg="#E0E0E0", 
                  command=self.generate_next_qid).pack(side="left", padx=5)

        tk.Label(frame_qid, text="Start QID:", font=("Arial", 9, "bold")).pack(side="left", padx=(10, 2))
        self.entry_id = tk.Entry(frame_qid, width=8, font=("Arial", 9, "bold"))
        self.entry_id.pack(side="left")
        self.entry_id.insert(0, "100001")

        # ==========================================
        # 2. PAGE SELECTION FRAME
        # ==========================================
        frame_pages = tk.Frame(frame_set)
        frame_pages.pack(fill="x", pady=(0, 10))

        tk.Label(frame_pages, text="Process Pages:", font=("Arial", 10)).pack(side="left")
        self.entry_pg_start = tk.Entry(frame_pages, width=5)
        self.entry_pg_start.pack(side="left", padx=5)
        self.entry_pg_start.insert(0, "1")
        
        tk.Label(frame_pages, text="to").pack(side="left")
        self.entry_pg_end = tk.Entry(frame_pages, width=5)
        self.entry_pg_end.pack(side="left", padx=5)
        self.entry_pg_end.insert(0, "End")

        frame_excel = tk.LabelFrame(frame_set, text="Excel Report Columns", padx=5, pady=5)
        frame_excel.pack(fill="x", pady=5)
        
        self.var_print_q = tk.BooleanVar(value=False)
        self.var_print_opts = tk.BooleanVar(value=False)
        
        tk.Checkbutton(frame_excel, text="Print Question Text", variable=self.var_print_q).pack(side="left", padx=10)
        tk.Checkbutton(frame_excel, text="Print Options & Answer Key", variable=self.var_print_opts).pack(side="left", padx=10)

        self.btn_go = tk.Button(self.page_setup, text="▶ START PROCESSING", font=("Arial", 12, "bold"), 
                                bg="#4CAF50", fg="white", height=2, command=self.start_job)
        self.btn_go.pack(fill="x", padx=20, pady=5)

    def build_process_page(self):
        tk.Label(self.page_process, text="Step 2: Processing Data", font=("Arial", 14, "bold"), pady=10).pack()
        self.progress = ttk.Progressbar(self.page_process, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(fill="x", padx=40, pady=20)
        self.text_log = tk.Text(self.page_process, height=20, font=("Consolas", 9), state="disabled", bg="#f4f4f4")
        self.text_log.pack(fill="both", expand=True, padx=20, pady=5)
        self.btn_new_batch = tk.Button(self.page_process, text="Processing Next Batch ↺", 
                                       font=("Arial", 11, "bold"), bg="#2196F3", fg="white", command=self.reset_ui)
        
        # ---> The Cancel Button
        self.btn_cancel = tk.Button(self.page_process, text="🛑 Cancel Process", 
                                    font=("Arial", 11, "bold"), bg="#f44336", fg="white", command=self.cancel_job)
        
        self.btn_new_batch = tk.Button(self.page_process, text="Processing Next Batch ↺", 
                                       font=("Arial", 11, "bold"), bg="#2196F3", fg="white", command=self.reset_ui)
        
    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if files:
            for f in files:
                if f not in self.selected_files: self.selected_files.append(f)
            self.sort_files_suffix()
    def cancel_job(self):
        self.cancel_flag = True
        self.btn_cancel.config(text="Stopping safely... Please wait", state="disabled")
        self.log_msg("\n🛑 Cancel requested! Finishing current page and stopping safely...")

    def generate_next_qid(self):
        # 1. Safely read the Last QID
        last_val = self.entry_last_qid.get().strip()
        if not last_val:
            messagebox.showwarning("Missing Data", "Please enter a Last QID to generate the next one.")
            return
        
        try:
            last_id_int = int(last_val)
        except ValueError:
            messagebox.showwarning("Invalid Input", "Last QID must be a number.")
            return

        # 2. Safely read the Round Off value (Crash Proofing)
        round_str = self.entry_round.get().strip()
        if not round_str:
            round_val = 10  # Fallback to 10 if they leave it blank
            self.entry_round.insert(0, "10")
        else:
            try:
                round_val = int(round_str)
                if round_val < 1: round_val = 1
            except ValueError:
                # If they typed letters by accident, force it back to 10
                round_val = 10 
                self.entry_round.delete(0, tk.END)
                self.entry_round.insert(0, "10")

        # 3. The Math (Find the next clean multiple)
        remainder = last_id_int % round_val
        if remainder == 0:
            next_id = last_id_int + round_val+1
        else:
            next_id = last_id_int + (round_val - remainder)+1

        # 4. Safely update the Start QID box
        self.entry_id.delete(0, tk.END)
        self.entry_id.insert(0, str(next_id))

    def clear_files(self):
        self.selected_files = []
        self.listbox.delete(0, tk.END)

    def sort_files_suffix(self):
        self.selected_files.sort(key=lambda x: os.path.basename(x).split('_', 1)[-1])
        self.listbox.delete(0, tk.END)
        for f in self.selected_files: self.listbox.insert(tk.END, os.path.basename(f))

    def start_job(self):
        if not self.selected_files: return messagebox.showwarning("No Files", "Select PDFs")
        
        try: start_id = int(self.entry_id.get())
        except: return messagebox.showerror("Error", "UID must be a number")
        
        try: pg_start = int(self.entry_pg_start.get())
        except: pg_start = 1
        
        try: pg_end = int(self.entry_pg_end.get())
        except: pg_end = -1 

        settings = {
            'is_bengali': self.var_ben.get(),
            'stamp_ids': self.var_stamp.get(),
            'stamp_pages': self.var_pagenum.get(),
            'find_dups': self.var_dups.get(),
            'print_q': self.var_print_q.get(),
            'print_opts': self.var_print_opts.get(),
            'pg_start': pg_start,
            'pg_end': pg_end
        }

        out_dir = os.path.dirname(self.selected_files[0])
        final_name = get_output_filename(self.selected_files[0])
        final_path = os.path.join(out_dir, final_name)

        self.show_process()
        self.progress["value"] = 0
        self.text_log.config(state="normal")
        self.text_log.delete(1.0, tk.END)
        self.text_log.config(state="disabled")

        # ---> NEW: Setup the UI for a fresh run
        self.cancel_flag = False
        self.btn_cancel.config(text="🛑 Cancel Process", state="normal")
        self.btn_cancel.pack(pady=10, fill="x", padx=20)
        self.btn_new_batch.pack_forget()
        
        t = threading.Thread(target=run_processing, 
                             args=(self.selected_files, final_path, start_id, settings, 
                                   self.log_msg, self.update_progress, self.job_finished, lambda: self.cancel_flag))
        t.start()

    def log_msg(self, msg):
        self.text_log.config(state="normal")
        self.text_log.insert(tk.END, msg + "\n")
        self.text_log.see(tk.END)
        self.text_log.config(state="disabled")

    def update_progress(self, value): self.progress["value"] = value

    def job_finished(self, success, last_uid=None, cancelled=False):
        self.btn_cancel.pack_forget() # Hide the cancel button
        
        if cancelled:
            self.log_msg("\n🛑 Process Aborted.")
            messagebox.showinfo("Cancelled", "Process was stopped. Temporary files deleted.")
            self.reset_ui() # Instantly kick back to the setup page!
            return

        self.log_msg("\n✨ DONE ✨" if success else "\n💀 FAILED")
        
        # Auto-fill the Last QID box only if it successfully finished
        if success and last_uid is not None:
            self.entry_last_qid.delete(0, tk.END)
            self.entry_last_qid.insert(0, str(last_uid))
            
        self.btn_new_batch.pack(pady=10, fill="x", padx=20)

    def reset_ui(self):
        self.clear_files()
        self.show_setup()

if __name__ == "__main__":
    app = App()
    app.mainloop()