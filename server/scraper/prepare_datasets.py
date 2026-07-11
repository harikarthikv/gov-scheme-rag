import json
import os
import re
from typing import List, Dict, Any
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove markdown artifacts if needed (e.g., **, *, _, etc., but keeping it simple)
    text = text.replace('**', '').replace('*', '').replace('ï»¿', '')
    # Remove duplicate whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_scheme_name(prompt: str) -> str:
    patterns = [
        r"Give me details about (.+?)(?:\.)?$",
        r"What is the application process for (.+?)(?:\?)?$",
        r"Tell me about the (.+?)(?: scheme)?(?:\.)?$",
        r"What is (.+?)(?:\?)?$",
        r"Who can apply for (.+?)(?:\?)?$",
        r"How much money/support does (.+?) give(?:\?)?$",
        r"I want to know about (.+?)\.? What does it offer\?",
        r"Can you explain the (.+?) program(?:\?)?$",
        r"Am I eligible for (.+?)(?:\?)?$"
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Remove trailing words like 'scheme', 'program' if accidentally caught
            if name.lower().endswith(" scheme"):
                name = name[:-7].strip()
            if name.lower().endswith(" program"):
                name = name[:-8].strip()
            return name
    return ""

def parse_satyajitdas() -> List[Dict[str, Any]]:
    logging.info("Loading satyajitdas/bharatschemes-v1 dataset locally...")
    local_path = os.path.join(DATA_DIR, "bharatschemes", "train.jsonl")
    schemes = []
    
    if not os.path.exists(local_path):
        logging.error(f"Failed to load local satyajitdas: {local_path} does not exist.")
        return []
        
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            if not messages or len(messages) < 3:
                continue
            user_msg = messages[1].get("content", "")
            asst_msg = messages[2].get("content", "")
            
            scheme_name = extract_scheme_name(user_msg)
            if not scheme_name:
                continue
                
            # Extract fields from assistant message using regex or simple split
            # Common markers: Eligibility:, Benefits:, How to apply:, Application Process
            
            def extract_section(text, current_marker, next_markers):
                idx = text.lower().find(current_marker.lower())
                if idx == -1:
                    return ""
                start = idx + len(current_marker)
                end = len(text)
                for marker in next_markers:
                    m_idx = text.lower().find(marker.lower(), start)
                    if m_idx != -1 and m_idx < end:
                        end = m_idx
                return text[start:end].strip()
    
            markers = ["Eligibility:", "Benefits:", "How to apply:", "Application Process", "Documents Required", "Frequently Asked Questions"]
            
            eligibility = extract_section(asst_msg, "Eligibility:", markers)
            benefits = extract_section(asst_msg, "Benefits:", markers)
            if not benefits:
                benefits = extract_section(asst_msg, "Benefits under", markers)
            how_to_apply = extract_section(asst_msg, "How to apply:", markers)
            if not how_to_apply:
                how_to_apply = extract_section(asst_msg, "Application Process", markers)
                
            # Description is everything before the first marker usually
            desc_end = len(asst_msg)
            for m in markers:
                idx = asst_msg.lower().find(m.lower())
                if idx != -1 and idx < desc_end:
                    desc_end = idx
            description = asst_msg[:desc_end].strip()
    
            scheme = {
                "scheme_name": clean_text(scheme_name),
                "description": clean_text(description),
                "eligibility": clean_text(eligibility),
                "benefits": clean_text(benefits),
                "how_to_apply": clean_text(how_to_apply),
                "source": "satyajitdas/bharatschemes-v1"
            }
            schemes.append(scheme)
    return schemes

def parse_shrijayan() -> List[Dict[str, Any]]:
    import pypdfium2 as pdfium
    import glob
    from tqdm import tqdm
    logging.info("Loading shrijayan/gov_myscheme dataset locally...")
    local_path = os.path.join(DATA_DIR, "gov_myscheme", "text_data", "*.pdf")
    pdf_files = glob.glob(local_path)
    
    schemes = []
    for i, file_path in enumerate(tqdm(pdf_files, desc="Parsing PDFs (gov_myscheme)")):
        try:
            pdf = pdfium.PdfDocument(file_path)
            text = ""
            for page in pdf:
                textpage = page.get_textpage()
                extracted = textpage.get_text_range()
                if extracted:
                    text += extracted + "\n"
                    
            if not text.strip():
                continue

            # Heuristic parsing for the PDF text
            # Usually the first line or two is the scheme name
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            scheme_name = lines[0] if lines else f"Unknown Scheme {i}"
            
            def extract_section(text, current_marker, next_markers):
                idx = text.lower().find(current_marker.lower())
                if idx == -1:
                    return ""
                start = idx + len(current_marker)
                end = len(text)
                for marker in next_markers:
                    m_idx = text.lower().find(marker.lower(), start)
                    if m_idx != -1 and m_idx < end:
                        end = m_idx
                return text[start:end].strip()

            markers = ["Details", "Benefits", "Eligibility", "Exclusions", "Application Process", "Documents Required", "Frequently Asked Questions"]
            
            description = extract_section(text, "Details", markers)
            if not description and len(lines) > 1:
                description = lines[1] # fallback
            
            eligibility = extract_section(text, "Eligibility", markers)
            benefits = extract_section(text, "Benefits", markers)
            how_to_apply = extract_section(text, "Application Process", markers)

            scheme = {
                "scheme_name": clean_text(scheme_name),
                "description": clean_text(description),
                "eligibility": clean_text(eligibility),
                "benefits": clean_text(benefits),
                "how_to_apply": clean_text(how_to_apply),
                "source": "shrijayan/gov_myscheme"
            }
            schemes.append(scheme)
        except Exception as e:
            logging.warning(f"Error parsing shrijayan row {i}: {e}")
            
    return schemes

def unify_schema(schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unified = []
    for idx, s in enumerate(schemes):
        unified.append({
            "scheme_id": f"SCHEME_{idx:05d}",
            "scheme_name": s.get("scheme_name", ""),
            "description": s.get("description", ""),
            "eligibility": s.get("eligibility", ""),
            "benefits": s.get("benefits", ""),
            "how_to_apply": s.get("how_to_apply", ""),
            "application_link": "",
            "ministry": "",
            "department": "",
            "state": "",
            "scheme_type": "",
            "target_beneficiaries": [],
            "source": s.get("source", ""),
            "last_updated": "",
            "tags": []
        })
    return unified

def classify_and_extract(schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    categories = {
        "Agriculture": ["farmer", "agriculture", "crop", "irrigation", "tractor"],
        "Education": ["student", "school", "scholarship", "education", "college", "degree"],
        "Healthcare": ["health", "medical", "hospital", "disease", "treatment", "pregnant"],
        "Women & Child Welfare": ["women", "child", "girl", "widow", "maternity", "mother"],
        "Employment": ["employment", "job", "worker", "labor", "wage"],
        "Skill Development": ["skill", "training", "vocational", "apprentice"],
        "Housing": ["housing", "house", "shelter", "home"],
        "MSME": ["msme", "enterprise", "business", "startup", "manufacturing", "industry"],
        "Finance": ["loan", "credit", "subsidy", "pension", "financial assistance", "insurance"],
        "Social Security": ["pension", "old age", "disabled", "disability", "senior citizen"]
    }
    
    beneficiaries_map = {
        "Farmers": ["farmer", "agriculture"],
        "Students": ["student", "school", "college", "scholar"],
        "Women": ["women", "girl", "widow", "female", "mother"],
        "Senior Citizens": ["senior citizen", "old age", "elderly", "60 years"],
        "MSMEs": ["msme", "enterprise", "business"],
        "Entrepreneurs": ["entrepreneur", "startup"],
        "Workers": ["worker", "labor", "employee", "unorganized"],
        "SC/ST": ["sc/st", "scheduled caste", "scheduled tribe", "dalit", "backward class"],
        "Differently Abled": ["disabled", "disability", "differently abled", "handicap", "divyang"],
        "General Public": ["citizen", "public", "everyone", "all"]
    }

    for s in schemes:
        text = (s["scheme_name"] + " " + s["description"] + " " + s["eligibility"]).lower()
        
        tags = set()
        for cat, keywords in categories.items():
            if any(kw in text for kw in keywords):
                tags.add(cat)
        if not tags:
            tags.add("Others")
        s["tags"] = list(tags)
        
        bens = set()
        for ben, keywords in beneficiaries_map.items():
            if any(kw in text for kw in keywords):
                bens.add(ben)
        if not bens:
            bens.add("General Public")
        s["target_beneficiaries"] = list(bens)
        
    return schemes

def deduplicate(schemes: List[Dict[str, Any]]) -> (List[Dict[str, Any]], int):
    seen = {}
    duplicates_removed = 0
    for s in schemes:
        name = s["scheme_name"].lower().strip()
        name = re.sub(r'[^a-z0-9]', '', name) # Normalize for deduplication
        if name in seen:
            duplicates_removed += 1
            # Keep the one with longer description
            if len(s["description"]) > len(seen[name]["description"]):
                # Keep ID of the first one to avoid ID churn, but take data from new
                s_id = seen[name]["scheme_id"]
                seen[name] = s
                seen[name]["scheme_id"] = s_id
        else:
            seen[name] = s
    return list(seen.values()), duplicates_removed

def validate(schemes: List[Dict[str, Any]], original_count: int, dup_removed: int) -> Dict[str, int]:
    valid = 0
    invalid = 0
    for s in schemes:
        if s["scheme_name"] and s["description"] and s["eligibility"]:
            valid += 1
        else:
            invalid += 1
            
    return {
        "total_records": original_count,
        "valid_records": valid,
        "invalid_records": invalid,
        "duplicate_records_removed": dup_removed
    }

def generate_rag_chunks(schemes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for s in schemes:
        # Create a rich text representation for the chunk
        chunk_text = f"Scheme Name: {s['scheme_name']}\n"
        if s["tags"]:
            chunk_text += f"Categories: {', '.join(s['tags'])}\n"
        if s["target_beneficiaries"]:
            chunk_text += f"Target Beneficiaries: {', '.join(s['target_beneficiaries'])}\n"
        if s["description"]:
            chunk_text += f"Description: {s['description']}\n"
        if s["eligibility"]:
            chunk_text += f"Eligibility: {s['eligibility']}\n"
        if s["benefits"]:
            chunk_text += f"Benefits: {s['benefits']}\n"
        if s["how_to_apply"]:
            chunk_text += f"How to Apply: {s['how_to_apply']}\n"
            
        chunks.append({
            "scheme_id": s["scheme_id"],
            "chunk_text": chunk_text.strip(),
            "metadata": {
                "scheme_name": s["scheme_name"],
                "category": s["tags"][0] if s["tags"] else "Others",
                "state": s["state"],
                "beneficiaries": s["target_beneficiaries"]
            }
        })
    return chunks

def main():
    satyajitdas = parse_satyajitdas()
    shrijayan = parse_shrijayan()
    
    combined_raw = satyajitdas + shrijayan
    
    with open(os.path.join(PROCESSED_DIR, "schemes_raw_combined.json"), "w", encoding="utf-8") as f:
        json.dump(combined_raw, f, indent=2, ensure_ascii=False)
        
    unified = unify_schema(combined_raw)
    classified = classify_and_extract(unified)
    
    with open(os.path.join(PROCESSED_DIR, "schemes_cleaned.json"), "w", encoding="utf-8") as f:
        json.dump(classified, f, indent=2, ensure_ascii=False)
        
    deduped, dup_count = deduplicate(classified)
    
    with open(os.path.join(PROCESSED_DIR, "schemes_deduplicated.json"), "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
        
    report = validate(deduped, len(combined_raw), dup_count)
    with open(os.path.join(REPORTS_DIR, "cleaning_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    rag_chunks = generate_rag_chunks(deduped)
    with open(os.path.join(PROCESSED_DIR, "schemes_rag_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(rag_chunks, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Pipeline complete. Report: {report}")

if __name__ == "__main__":
    main()
