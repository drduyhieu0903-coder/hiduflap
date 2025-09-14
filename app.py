# app_optimized_v11_fixed.py — HIDU Facial Flap (Optimized Performance v11.0 - Fixed)
# =============================================================================
# Tối ưu hiệu năng: Quản lý phiên làm việc nâng cao, đánh dấu ảnh đã edit,
# Sắp xếp ưu tiên ảnh chưa edit, cải thiện workflow và UX - ĐÃ SỬA LỖI
# =============================================================================

from __future__ import annotations
import os, re, io, json, time, shutil, hashlib, tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError, ExifTags
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

# ========================= CONFIG =========================
DATA_ROOT = Path("facial_flap_data").resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = DATA_ROOT / "metadata.parquet"
CSV_PATH     = DATA_ROOT / "metadata.csv"
HIDDEN_BOOKS_JSON = DATA_ROOT / ".hidden_books.json"
SESSION_STATE_JSON = DATA_ROOT / ".session_state.json"
WORK_SESSION_JSON = DATA_ROOT / ".work_sessions.json"  # Lưu lịch sử phiên làm việc
USER_ACTIVITY_JSON = DATA_ROOT / ".user_activity.json"  # Theo dõi hoạt động người dùng
RULES_EN_DEFAULT  = DATA_ROOT / "flap_rules_en.csv"
CLINICAL_DIR      = DATA_ROOT / "_clinical"
ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)

APP_TITLE = "HIDU Facial Flap — Explorer"

# UI defaults
MIN_IMG_SIZE_DEFAULT = 160
NEARBY_RATIO_DEFAULT = 0.20
SAVE_ALL_FALLBACK    = True

# Work session settings
SESSION_TIMEOUT_MINUTES = 30  # Tự động lưu session sau 30 phút
AUTO_SAVE_INTERVAL = 3        # Tự động lưu mỗi 5 phút

# ========================= ADVANCED SESSION STATE MANAGEMENT =========================
class WorkSessionManager:
    """Quản lý phiên làm việc nâng cao"""
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.start_time = time.time()
        self.last_activity = time.time()
        self.auto_save_counter = 0
    
    def _generate_session_id(self) -> str:
        """Tạo ID phiên làm việc duy nhất"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{hash_part}"
    
    def update_activity(self):
        """Cập nhật thời gian hoạt động cuối"""
        self.last_activity = time.time()
    
    def is_session_expired(self) -> bool:
        """Kiểm tra phiên làm việc có hết hạn không"""
        return (time.time() - self.last_activity) > (SESSION_TIMEOUT_MINUTES * 60)
    
    def get_session_duration(self) -> str:
        """Lấy thời gian làm việc trong phiên hiện tại"""
        duration = time.time() - self.start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"
    
    def save_session_snapshot(self):
        """Lưu snapshot phiên làm việc"""
        try:
            session_data = {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "last_activity": self.last_activity,
                "duration_minutes": (time.time() - self.start_time) / 60,
                "edited_images": list(st.session_state.get("edited_images_set", set())),
                "images_processed": len(st.session_state.get("edited_images_set", set())),
                "timestamp": datetime.now().isoformat()
            }
            
            # Load existing sessions
            sessions = []
            if WORK_SESSION_JSON.exists():
                try:
                    sessions = json.loads(WORK_SESSION_JSON.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            # Add current session
            sessions.append(session_data)
            
            # Keep only last 50 sessions
            sessions = sessions[-50:]
            
            WORK_SESSION_JSON.write_text(
                json.dumps(sessions, ensure_ascii=False, indent=2), 
                encoding="utf-8"
            )
        except Exception:
            pass

def init_session_state():
    """Initialize all session state variables with enhanced tracking"""
    defaults = {
        "hidden_books": [],
        "selected_list": [],
        "lightbox_open": False,
        "lightbox_seq": [],
        "lightbox_idx": 0,
        "library_prefill": {},
        "edited_images": [],  # Deprecated, use edited_images_set
        "pending_updates": {},
        "ui_update_flag": False,
        "last_metadata_update": 0,
        "page_state": {},
        "defer_rerun": False,
        "edited_images_set": set(),  # Track edited images for sorting
        "last_visit_times": {},      # Track when images were last viewed/edited
        "edit_timestamps": {},       # Detailed edit timestamps
        "multi_select_sites": [],
        "work_session_manager": WorkSessionManager(),
        "productivity_stats": {     # Thống kê năng suất
            "images_edited_today": 0,
            "images_edited_session": 0,
            "total_edits": 0,
            "avg_time_per_edit": 0.0
        },
        "ui_preferences": {         # Tùy chỉnh giao diện
            "show_edited_badge": True,
            "sort_edited_first": False,  # False = unedited first
            "auto_save_enabled": True,
            "show_productivity_stats": True
        },
        "quick_filters": {          # Bộ lọc nhanh
            "show_only_unedited": False,
            "show_only_edited": False,
            "show_recent_edits": False
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Load hidden books on first run
    if not st.session_state.hidden_books and HIDDEN_BOOKS_JSON.exists():
        try:
            st.session_state.hidden_books = json.loads(
                HIDDEN_BOOKS_JSON.read_text(encoding="utf-8")
            )
        except Exception:
            pass
    
    # Load session state from previous session
    load_persistent_session_state()
    
    # Update session activity
    if hasattr(st.session_state, "work_session_manager"):
        st.session_state.work_session_manager.update_activity()

def load_persistent_session_state():
    """Load session state from disk with enhanced data"""
    if SESSION_STATE_JSON.exists():
        try:
            session_data = json.loads(SESSION_STATE_JSON.read_text(encoding="utf-8"))
            
            # Load core data
            if "edited_images_set" in session_data:
                st.session_state.edited_images_set = set(session_data["edited_images_set"])
            if "last_visit_times" in session_data:
                st.session_state.last_visit_times = session_data["last_visit_times"]
            if "edit_timestamps" in session_data:
                st.session_state.edit_timestamps = session_data["edit_timestamps"]
            if "productivity_stats" in session_data:
                st.session_state.productivity_stats.update(session_data["productivity_stats"])
            if "ui_preferences" in session_data:
                st.session_state.ui_preferences.update(session_data["ui_preferences"])
                
        except Exception:
            pass

def save_session_state():
    """Save enhanced session state to disk"""
    try:
        session_data = {
            "edited_images_set": list(st.session_state.edited_images_set),
            "last_visit_times": st.session_state.last_visit_times,
            "edit_timestamps": st.session_state.edit_timestamps,
            "productivity_stats": st.session_state.productivity_stats,
            "ui_preferences": st.session_state.ui_preferences,
            "last_save": time.time()
        }
        
        SESSION_STATE_JSON.write_text(
            json.dumps(session_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        
        # Save work session snapshot periodically
        if hasattr(st.session_state, "work_session_manager"):
            st.session_state.work_session_manager.auto_save_counter += 1
            if st.session_state.work_session_manager.auto_save_counter >= AUTO_SAVE_INTERVAL:
                st.session_state.work_session_manager.save_session_snapshot()
                st.session_state.work_session_manager.auto_save_counter = 0
                
    except Exception:
        pass

# ========================= ENHANCED IMAGE EDIT TRACKING =========================
def mark_image_edited(image_path: str, edit_type: str = "manual"):
    """Mark an image as edited with detailed tracking"""
    current_time = time.time()
    
    # Add to edited set
    st.session_state.edited_images_set.add(image_path)
    
    # Update timestamps
    st.session_state.last_visit_times[image_path] = current_time
    
    # Detailed edit tracking
    if image_path not in st.session_state.edit_timestamps:
        st.session_state.edit_timestamps[image_path] = []
    
    st.session_state.edit_timestamps[image_path].append({
        "timestamp": current_time,
        "edit_type": edit_type,
        "session_id": st.session_state.work_session_manager.session_id
    })
    
    # Update productivity stats
    st.session_state.productivity_stats["images_edited_session"] += 1
    st.session_state.productivity_stats["total_edits"] += 1
    
    # Calculate average time per edit
    if st.session_state.productivity_stats["total_edits"] > 0:
        session_duration = time.time() - st.session_state.work_session_manager.start_time
        st.session_state.productivity_stats["avg_time_per_edit"] = (
            session_duration / st.session_state.productivity_stats["images_edited_session"]
        )
    
    # Auto-save state
    if st.session_state.ui_preferences["auto_save_enabled"]:
        save_session_state()

def get_image_edit_info(image_path: str) -> Dict[str, Any]:
    """Get detailed edit information for an image"""
    if image_path not in st.session_state.edited_images_set:
        return {"is_edited": False}
    
    edit_history = st.session_state.edit_timestamps.get(image_path, [])
    last_edit = edit_history[-1] if edit_history else {}
    
    return {
        "is_edited": True,
        "edit_count": len(edit_history),
        "last_edit_time": last_edit.get("timestamp", 0),
        "last_edit_type": last_edit.get("edit_type", "unknown"),
        "time_since_edit": time.time() - last_edit.get("timestamp", 0) if last_edit else 0
    }

def format_time_since_edit(seconds: float) -> str:
    """Format time since last edit in human readable format"""
    if seconds < 60:
        return "vừa xong"
    elif seconds < 3600:
        return f"{int(seconds//60)} phút trước"
    elif seconds < 86400:
        return f"{int(seconds//3600)} giờ trước"
    else:
        return f"{int(seconds//86400)} ngày trước"

# ========================= OPTIMIZED METADATA OPERATIONS =========================
@st.cache_data(
    show_spinner="Đang tải metadata...",
    ttl=300,
    hash_funcs={type(None): lambda x: None}
)
def md_load_cached(last_update_timestamp: float = 0) -> pd.DataFrame:
    """Cached metadata loading with timestamp-based invalidation"""
    path = PARQUET_PATH if _parquet_available() and PARQUET_PATH.exists() else CSV_PATH
    if not path.exists():
        return pd.DataFrame(columns=META_COLS)
    
    try:
        df = (pd.read_parquet(path) if _parquet_available() and path.suffix == ".parquet" 
              else pd.read_csv(path))
    except Exception as e:
        st.error(f"Lỗi đọc metadata: {e}")
        return pd.DataFrame(columns=META_COLS)
    
    for c in META_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ["relevance_score"] else 0
    
    return df

def md_load() -> pd.DataFrame:
    """Load metadata with optimized caching"""
    return md_load_cached(st.session_state.last_metadata_update)

def md_save_deferred(df: pd.DataFrame) -> None:
    """Save metadata with deferred execution to batch multiple updates"""
    st.session_state.pending_updates["metadata"] = df
    st.session_state.defer_rerun = True

def md_save_immediate(df: pd.DataFrame) -> None:
    """Immediate metadata save (use sparingly)"""
    for c in META_COLS:
        if c not in df.columns:
            df[c] = "" if c != "relevance_score" else 0
    
    df = df[META_COLS]
    use_parquet = _parquet_available()
    path = PARQUET_PATH if use_parquet else CSV_PATH
    backup_path = path.with_suffix(f"{path.suffix}.bak")
    
    try:
        if path.exists():
            shutil.copy2(path, backup_path)
        
        if use_parquet:
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        
        st.session_state.last_metadata_update = time.time()
        md_load_cached.clear()
        
    except Exception as e:
        st.error(f"Lưu metadata thất bại: {e}. Có thể khôi phục từ: {backup_path}")

def process_pending_updates():
    """Process all pending updates in batch"""
    if not st.session_state.pending_updates:
        return
    
    if "metadata" in st.session_state.pending_updates:
        md_save_immediate(st.session_state.pending_updates["metadata"])
        del st.session_state.pending_updates["metadata"]
    
    st.session_state.defer_rerun = False

# ========================= UTILITY FUNCTIONS =========================
def safe_book_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u00C0-\u1EF9]+", "_", str(name)).strip("_") or "unknown"

def unique_filename(p: Path) -> Path:
    if not p.exists():
        return p
    stem, ext, i = p.stem, p.suffix, 2
    while True:
        cand = p.with_name(f"{stem}-v{i}{ext}")
        if not cand.exists():
            return cand
        i += 1

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"-\s+\n?", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_chapter_tokens(s: str) -> str:
    if not s:
        return ""
    return re.sub(
        r"(?:Chapter|Chương)\s*\d+[:.\-\s]*", "", s, flags=re.IGNORECASE
    ).strip(" :.-\u00A0")

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def rect_min_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    if a.intersects(b):
        return 0.0
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
    return (dx * dx + dy * dy) ** 0.5

def make_thumb(src: Path, dst: Path, max_side=512) -> Optional[Path]:
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            ensure_dir(dst.parent)
            im.save(dst, format="JPEG", quality=88)
        return dst
    except Exception:
        return None

def thumb_path_for(rel_img: str) -> Path:
    base = re.sub(r"[\\/]+", "__", rel_img)
    return DATA_ROOT / "_thumbs" / (Path(base).with_suffix(".jpg").name)

def highlight(text: str, query: str) -> str:
    if not query:
        return text
    try:
        pat = re.compile(re.escape(query), re.IGNORECASE)
        return pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    except Exception:
        return text

def exif_datetime_str(img: Image.Image) -> str:
    try:
        exif = img.getexif()
        if not exif:
            return ""
        dt_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == "DateTimeOriginal":
                dt_tag = k
                break
        if dt_tag and exif.get(dt_tag):
            return str(exif.get(dt_tag))
    except Exception:
        pass
    return ""

# ========================= METADATA SCHEMA =========================
META_COLS = [
    "book_name", "image_path", "thumb_path", "page_num", "fig_num", "group_key",
    "caption", "context", "anatomical_site", "flap_type", "confidence",
    "source", "saved_at", "bytes_md5", "relevance_score", "notes"
]

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False

# ========================= BATCH OPERATIONS =========================
def md_update_by_paths_batch(updates_dict: Dict[str, Dict]) -> None:
    """Batch update multiple images at once"""
    df = md_load()
    
    for image_paths, updates in updates_dict.items():
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        mask = df["image_path"].isin(image_paths)
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
    
    md_save_deferred(df)

def md_delete_by_paths_batch(image_rels: List[str]) -> None:
    """Batch delete multiple images"""
    df = md_load()
    df = df[~df["image_path"].isin(image_rels)].copy()
    md_save_deferred(df)
def delete_duplicate_edited_images():
    """
    Tìm và xóa các ảnh trùng lặp, ưu tiên giữ lại bản đã được edit.
    Quy tắc: Giữ lại bản ghi đã được edit và được lưu gần đây nhất.
    """
    df = md_load()
    if df.empty or 'bytes_md5' not in df.columns:
        return 0

    edited_set = st.session_state.edited_images_set
    
    # 1. Tìm các mã MD5 bị trùng lặp
    md5_counts = df['bytes_md5'].value_counts()
    duplicate_md5s = md5_counts[md5_counts > 1].index.tolist()

    if not duplicate_md5s:
        return 0 # Không có gì để xoá

    indices_to_delete = []
    
    # 2. Lặp qua từng nhóm ảnh trùng lặp
    for md5 in duplicate_md5s:
        group = df[df['bytes_md5'] == md5].copy()
        
        # Đánh dấu các bản ghi đã được edit trong nhóm
        group['is_edited'] = group['image_path'].apply(lambda path: path in edited_set)
        
        # Sắp xếp để ưu tiên: đã edit > lưu gần nhất
        group_sorted = group.sort_values(by=['is_edited', 'saved_at'], ascending=[False, False])
        
        # Giữ lại bản ghi đầu tiên (tốt nhất) và xóa các bản còn lại
        record_to_keep = group_sorted.iloc[0]
        indices_to_delete.extend(group[group['image_path'] != record_to_keep['image_path']].index.tolist())

    if not indices_to_delete:
        return 0

    records_to_delete = df.loc[indices_to_delete]

    # 3. Thực hiện xóa file vật lý và thumbnail
    for _, row in records_to_delete.iterrows():
        try:
            rel_path = row['image_path']
            (DATA_ROOT / rel_path).unlink(missing_ok=True)
            
            thumb_path = row.get('thumb_path')
            if thumb_path and (DATA_ROOT / thumb_path).exists():
                (DATA_ROOT / thumb_path).unlink(missing_ok=True)
        except Exception as e:
            st.warning(f"Lỗi khi xoá file {rel_path}: {e}")

    # 4. Xóa metadata khỏi DataFrame
    df_cleaned = df.drop(index=indices_to_delete)
    md_save_immediate(df_cleaned)

    # 5. Dọn dẹp session_state để đảm bảo tính nhất quán
    paths_deleted = records_to_delete['image_path'].tolist()
    st.session_state.edited_images_set -= set(paths_deleted)
    
    return len(indices_to_delete)

# ========================= KEYWORDS & SCORING =========================
ANATOMY_KEYWORDS = {
    "nose": ["mũi","nasal","nose","ala","alar","tip","dorsum","cánh mũi","thắp mũi","columella","sidewall","rim","bridge","lobule","soft triangle","nasal bone","upper lateral cartilage","lower lateral cartilage","septum","nasal tip","nasal-cheek","nasal sidewall"],
    "cheek": ["má","cheek","zygoma","malar","midface","buccal","gò má","suborbital","infraorbital","preauricular","mandibular","nasal-cheek","cheek advancement"],
    "lip": ["môi","lip","vermilion","labial","philtrum","commissure","vermilion","orbicularis oris","modiolus","cupid's bow","white roll","prolabium","tubercle","upper lip","lower lip","commissure deformity"],
    "eye": ["mắt","eye","palpebral","eyelid","canthus","orbital","mí mắt","lower lid","upper lid","medial canthus","lateral canthus","tarsus","conjunctiva","levator","orbicularis oculi","septum","fornix","supra-eyebrow","medial canthal","lateral canthal"],
    "forehead": ["trán","forehead","glabella","frontal","brow","nasion","supraorbital","supratrochlear","supra-eyebrow","glabellar"],
    "chin": ["cằm","chin","mental","submental","labiomental","mentum","submental"],
    "ear": ["tai","ear","auricle","pinna","helical rim","conchal","helix","antihelix","scapha","concha","tragus","lobule","triangular fossa","external auditory canal","helical rim","conchal bowl","lobule defects","upper third defects","lower two-third of auricle"],
    "neck": ["cổ","neck","cervical","supraclavicular","platysma","sternocleidomastoid"],
    "temple": ["thái dương","temple","temporal","hairline","temporoparietal fascia","temporal skin defects"],
    "scalp": ["da đầu","scalp","vertex","occipital","parietal","frontal scalp","temporal scalp","aponeurosis","pericranium","anterior scalp","frontotemporal region"]
}

FLAP_TYPE_KEYWORDS = {
    "rotation": [
        "rotation", "xoay", "rotational", "mustardé", "semicircular", "tenzel",
        "karapandzic", "worthen", "bilateral rotation", "fan",
        "anteroposterior scalp rotation", "cheek rotational", "miter",
        "rotational miter", "unilateral rotation", "worthen rotation",
        "yin-yang", "pinwheel", "o-z", "gilles fan", "double opposing rotation"
    ],
    "advancement": [
        "advancement", "trượt", "v-y", "vy", "island", "wedge",
        "direct closure", "a-t", "s-plasty", "bernard", "rintala",
        "perialar crescentic", "cheek advancement", "webster cheek advancement",
        "step", "bilateral transverse forehead advancement",
        "central prolabial advancement", "full-thickness perialar crescentic advancement",
        "lateral advancement", "rim advancement", "v-y advancement",
        "hughes", "bipedicle", "antia-buch", "primary closure", "mini-advancement",
        "tarsoconjunctival", "cutler-beard"
    ],
    "transposition": [
        "transposition", "chuyển vị", "bilobed", "rhombic", "limberg",
        "nasolabial", "dorsal nasal", "dufourmentel", "banner", "finger",
        "tripier", "rhomboid", "double rhomboid", "single rhomboid",
        "triple rhomboid", "ll", "nasolabial island",
        "bilobed and glabellar finger", "glabellar island", "revolving door",
        "temporalis fascia", "supraclavicular artery island", "famm", "submental"
    ],
    "interpolation": [
        "interpolation", "nội suy", "pmff", "forehead flap", "postauricular",
        "pull-through", "lip-switch", "abbe", "estlander", "schmid",
        "cervical tubed pedicle", "washio", "scalping", "forehead scalping",
        "schmid forehead", "paramedian forehead", "long-tubed pedicle",
        "fricke", "split-finger", "tongue flap"
    ],
    "free": [
        "free flap", "tự do", "alt", "rfff", "microvascular", "free flaps",
        "anterolateral thigh", "latissimus dorsi"
    ],
    "composite": [
        "composite", "chondrocutaneous", "osteocutaneous",
        "composite grafts", "chondromucosal graft", "full thickness skin graft",
        "skin graft", "famm"
    ]
}
ANATOMY_OPTIONS = ["unknown"] + list(ANATOMY_KEYWORDS.keys())
FLAP_OPTIONS = ["unknown"] + list(FLAP_TYPE_KEYWORDS.keys())

# Scored keywords
SCORED_KEYWORDS = {
    "very_high": {
        "bilobed": 15, "rhombic": 15, "limberg": 15, "nasolabial": 12,
        "forehead flap": 12, "pmff": 12, "v-y": 10, "vy": 10,
        "dorsal nasal": 10, "tenzel": 10, "hughes": 12, "abbe": 11, "estlander": 11,
        "karapandzic": 12, "bernard": 10, "antia-buch": 10, "scaif": 10
    },
    "high": {
        "rotation flap": 8, "advancement flap": 8, "transposition flap": 8,
        "interpolation flap": 8, "free flap": 8, "vạt nội suy": 8, "vạt chuyển vị": 8,
    },
    "medium": {
        "vạt": 5, "tái tạo": 5, "tạo hình": 5, "che phủ": 4, "đóng khuyết": 4,
        "reconstruction": 5, "closure": 4, "coverage": 4, "skin graft": 3, "defect": 2
    },
    "anatomy": {kw: 3 for kws in ANATOMY_KEYWORDS.values() for kw in kws},
    "low": {
        "fig.": 1, "figure": 1, "preop": 1, "postop": 1
    },
    "negative": {
        "schematic": -20, "diagram": -20, "illustration": -20, "sơ đồ": -20
    }
}

# ========================= CAPTION & SCORING FUNCTIONS =========================
FIG_PATTERNS = [
    r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*([\dA-Za-z]{1,4}(?:\.\d+)?)\s*[.:‒-]\s*(.+)",
    r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*[.:‒-]\s*(.+)",
]

def extract_page_caption(page_text: str) -> Tuple[str, str]:
    for pat in FIG_PATTERNS:
        m = re.search(pat, page_text, flags=re.IGNORECASE)
        if m:
            gs = m.groups()
            fig = gs[0].strip() if len(gs) >= 2 else ""
            cap = strip_chapter_tokens(gs[-1].strip())
            return fig, cap
    return "", strip_chapter_tokens((page_text.split(".")[0] if page_text else "")[:180])

def get_image_rect(page: fitz.Page, xref: int) -> Optional[fitz.Rect]:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return None
    
    for info in infos:
        if info.get("xref") == xref and "bbox" in info and len(info["bbox"]) == 4:
            return fitz.Rect(info["bbox"])
    return None

def nearby_text_for_image(page: fitz.Page, img_rect: Optional[fitz.Rect], nearby_ratio: float) -> str:
    text_all = normalize_text(page.get_text() or "")
    if not img_rect:
        return strip_chapter_tokens(text_all[:400])
    
    radius = max(12.0, page.rect.height * max(0.05, float(nearby_ratio)))
    blocks = page.get_text("blocks")
    bag = []
    
    for b in blocks:
        if len(b) < 5 or not isinstance(b[4], str):
            continue
        rect = fitz.Rect(b[:4])
        if rect.intersects(img_rect) or rect_min_distance(rect, img_rect) <= radius:
            bag.append(normalize_text(b[4]))
    
    ctx = " ".join(bag).strip()
    m = re.search(
        r"(?i)(?:Fig(?:ure)?|Hình|Ảnh)\s*[\dA-Za-z.\-]*\s*[.:‒-]\s*(.+)", ctx
    )
    if m:
        return strip_chapter_tokens(m.group(1))[:350]
    
    return strip_chapter_tokens(ctx[:350]) or strip_chapter_tokens(text_all[:300])

def guess_labels(text: str) -> Tuple[str, str, str]:
    s = (text or "").lower()
    site_scores = {site: sum(1 for kw in kws if kw in s) for site, kws in ANATOMY_KEYWORDS.items()}
    flap_scores = {ft: sum(1 for kw in kws if kw in s) for ft, kws in FLAP_TYPE_KEYWORDS.items()}
    
    site = max(site_scores, key=site_scores.get) if any(site_scores.values()) else "unknown"
    ft = max(flap_scores, key=flap_scores.get) if any(flap_scores.values()) else "unknown"
    
    if site != "unknown" and ft != "unknown":
        conf = "high"
    elif site != "unknown" or ft != "unknown":
        conf = "medium"
    else:
        conf = "low"
    
    return site, ft, conf

def calculate_relevance_score(text: str) -> int:
    if not text:
        return 0
    
    t_lower = text.lower()
    score = 0
    found_groups = set()
    
    for group, kws in SCORED_KEYWORDS.items():
        for kw, s in (kws.items() if isinstance(kws, dict) else []):
            if kw in t_lower:
                score += s
                found_groups.add(group)
    
    if "anatomy" in found_groups and ("very_high" in found_groups or "high" in found_groups):
        score += 15  # bonus phối hợp site + flap
    
    return score

# ========================= ENHANCED PDF PROCESSING =========================
def process_pdf(uploaded, *, min_px: int, allow_duplicates: bool, save_all_if_no_kw: bool,
                nearby_ratio: float, relevance_threshold: int, progress=None) -> Tuple[List[str], int, Path]:
    """Enhanced PDF processing with better tracking and session management"""
    
    book = Path(uploaded.name).stem
    safe_b = safe_book_name(book)
    out_book_dir = DATA_ROOT / safe_b
    ensure_dir(out_book_dir)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    saved_files: List[str] = []
    new_rows = []

    try:
        with fitz.open(tmp_path) as doc:
            total_pages = len(doc)
            for pno in range(total_pages):
                if progress:
                    progress((pno) / max(1, total_pages), 
                            f"Đang xử lý trang {pno+1}/{total_pages} — {book}")
                
                page = doc.load_page(pno)
                page_text = normalize_text(page.get_text() or "")
                page_fig, page_cap = extract_page_caption(page_text)
                images = page.get_images(full=True)
                
                if not images:
                    continue

                for idx, im in enumerate(images, start=1):
                    try:
                        xref = im[0] if im else None
                        if not isinstance(xref, int):
                            continue
                        
                        base = doc.extract_image(xref)
                        if not base or "image" not in base:
                            continue
                        
                        img_bytes: bytes = base["image"]
                        md5 = md5_bytes(img_bytes)
                        
                        if (not allow_duplicates) and md5 in known:
                            continue
                        
                        try:
                            with Image.open(io.BytesIO(img_bytes)) as im_pil:
                                w, h = im_pil.size
                            if max(w, h) < int(min_px):
                                continue
                        except Exception:
                            pass

                        rect = get_image_rect(page, xref)
                        near_cap = nearby_text_for_image(page, rect, nearby_ratio)
                        caption = near_cap if near_cap else page_cap
                        combined_text = f"{caption} {page_text}"
                        rel_score = calculate_relevance_score(combined_text)
                        
                        if (rel_score < relevance_threshold) and (not save_all_if_no_kw):
                            continue

                        m = re.search(
                            r"(?:Fig(?:ure)?|Hình|Ảnh)\s*([\dA-Za-z.]+)", 
                            near_cap or "", flags=re.IGNORECASE
                        )
                        fig_local = m.group(1) if m else ""
                        group_key = f"fig_{fig_local}" if fig_local else f"p{pno+1}"

                        page_folder = out_book_dir / f"p{pno+1}"
                        ensure_dir(page_folder)
                        ext = (base.get("ext") or "png").lower()
                        if ext not in ("png", "jpg", "jpeg"):
                            ext = "png"
                        
                        fname = f"{safe_b}_p{pno+1}_img{idx}.{ext}"
                        out_path = unique_filename(page_folder / fname)
                        out_path.write_bytes(img_bytes)

                        rel = str(out_path.relative_to(DATA_ROOT))
                        tpath = thumb_path_for(rel)
                        make_thumb(out_path, tpath, max_side=512)

                        site, flap, conf = guess_labels(combined_text)
                        row = {
                            "book_name": book, "image_path": rel,
                            "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                            "page_num": pno+1, "fig_num": fig_local, "group_key": group_key,
                            "caption": caption, "context": (near_cap or page_text)[:600],
                            "anatomical_site": site, "flap_type": flap, "confidence": conf,
                            "source": "pdf", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "bytes_md5": md5, "relevance_score": int(rel_score), "notes": "",
                        }
                        new_rows.append(row)
                        saved_files.append(str(out_path))
                        known.add(md5)
                        
                    except Exception:
                        continue

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            md_save_immediate(df)
        
        return saved_files, len(saved_files), out_book_dir
        
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# ========================= WEB INGEST =========================
def _guess_book_from_url(url: str) -> str:
    p = urlparse(url)
    host = (p.netloc or "web").replace("www.", "")
    path = (p.path or "/").strip("/").split("/")
    slug = path[0] if path and path[0] else "index"
    return safe_book_name(f"{host}__{slug}")

def _extract_captions_soup(soup: BeautifulSoup, img_tag) -> str:
    cap = ""
    fig = img_tag.find_parent("figure")
    if fig and (fc := fig.find("figcaption")):
        cap = normalize_text(fc.get_text(" ", strip=True))
    
    if cap:
        return cap
    
    alt = img_tag.get("alt") or ""
    title = img_tag.get("title") or ""
    cand = " ".join(x for x in [alt, title] if x).strip()
    return normalize_text(cand) if cand else ""

def ingest_web_html(url: str, *, min_px: int, allow_duplicates: bool, 
                   save_all_if_no_kw: bool, min_score: int) -> Tuple[int, int, Path]:
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        st.error(f"Tải trang thất bại: {e}")
        return 0, 0, DATA_ROOT

    soup = BeautifulSoup(r.text, "html.parser")
    book = _guess_book_from_url(url)
    out_book_dir = DATA_ROOT / safe_book_name(book) / "web"
    ensure_dir(out_book_dir)

    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    kept = 0
    new_rows = []

    imgs = soup.find_all("img")
    for idx, img in enumerate(imgs, start=1):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        
        img_url = urljoin(url, src)
        try:
            ir = requests.get(img_url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            if ir.status_code != 200 or not ir.content:
                continue
            
            md5 = md5_bytes(ir.content)
            if (not allow_duplicates) and md5 in known:
                continue

            try:
                with Image.open(io.BytesIO(ir.content)) as im:
                    w, h = im.size
                    fmt = (im.format or "JPEG").lower()
                if max(w, h) < int(min_px):
                    continue
            except UnidentifiedImageError:
                continue

            caption = _extract_captions_soup(soup, img)
            score = calculate_relevance_score(caption)
            if score < int(min_score) and not save_all_if_no_kw:
                continue

            ext = ".png" if fmt == "png" else ".jpg"
            fname = unique_filename(out_book_dir / f"{book}_web_img{idx}{ext}")
            fname.write_bytes(ir.content)

            rel = str(fname.relative_to(DATA_ROOT))
            tpath = thumb_path_for(rel)
            make_thumb(fname, tpath, max_side=512)

            site, flap, conf = guess_labels(caption)
            row = {
                "book_name": book, "image_path": rel,
                "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                "page_num": 1, "fig_num": "", "group_key": "web",
                "caption": caption, "context": caption,
                "anatomical_site": site, "flap_type": flap, "confidence": conf,
                "source": f"web:{urlparse(url).netloc}", 
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bytes_md5": md5, "relevance_score": int(score), "notes": "",
            }
            new_rows.append(row)
            kept += 1
            known.add(md5)
            
        except Exception:
            continue

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        md_save_deferred(df)
    
    return kept, kept, out_book_dir

# ========================= CLINICAL IMAGE UPLOAD =========================
def ingest_clinical_images(files: List, *, case_name: str, caption_prefix: str, 
                          site: str, flap: str, allow_duplicates: bool) -> int:
    if not files:
        return 0
    
    case = safe_book_name(case_name or "Clinical")
    out_dir = CLINICAL_DIR / case
    ensure_dir(out_dir)
    df = md_load()
    known = set(df.get("bytes_md5", [])) if "bytes_md5" in df.columns else set()
    added = 0
    new_rows = []
    
    for f in files:
        try:
            b = f.read()
            if not b:
                continue
            
            md5 = md5_bytes(b)
            if (not allow_duplicates) and md5 in known:
                continue
            
            with Image.open(io.BytesIO(b)) as im:
                w, h = im.size
                fmt = (im.format or "JPEG").lower()
                dt_str = exif_datetime_str(im)
            
            ext = ".png" if fmt == "png" else ".jpg"
            fname = unique_filename(out_dir / f"{case}_{int(time.time()*1000)}{ext}")
            fname.write_bytes(b)
            rel = str(fname.relative_to(DATA_ROOT))
            tpath = thumb_path_for(rel)
            make_thumb(fname, tpath, max_side=512)

            base_cap = Path(f.name).stem.replace("_", " ").replace("-", " ")
            cap = " ".join([x for x in [caption_prefix.strip(), base_cap] if x]).strip()
            ctx = f"clinical upload {dt_str}".strip()
            combo_text = f"{cap} {ctx}"
            site_g, flap_g, conf = guess_labels(combo_text)
            site_final = site if site != "(Giữ nguyên)" else site_g
            flap_final = flap if flap != "(Giữ nguyên)" else flap_g
            rel_score = calculate_relevance_score(combo_text)

            row = {
                "book_name": f"Clinical::{case}",
                "image_path": rel, 
                "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                "page_num": 1, "fig_num": "", "group_key": "clinical",
                "caption": cap, "context": ctx,
                "anatomical_site": site_final, "flap_type": flap_final, "confidence": conf,
                "source": "clinical", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bytes_md5": md5, "relevance_score": int(rel_score), "notes": "",
            }
            new_rows.append(row)
            known.add(md5)
            added += 1
            
        except Exception:
            continue
    
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        md_save_deferred(df)
    
    return added

# ========================= RULES MANAGEMENT =========================
SUGG_REQUIRED_COLS = [
    "region_en", "subunit_en", "unit", "size_lo", "size_hi", "depth_set", "depth_layers",
    "flap_type", "flap_name", "priority", "first_line", "steps", "notes", "pitfalls",
    "alternatives", "assoc", "keywords", "refs"
]

@st.cache_data(show_spinner=False)
def load_rules_en(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SUGG_REQUIRED_COLS)
    
    try:
        df = pd.read_csv(path)
        for c in SUGG_REQUIRED_COLS:
            if c not in df.columns:
                df[c] = ""
        
        # type conversions
        df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(9999).astype(int)
        for c in ["size_lo", "size_hi"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["region_en", "subunit_en", "unit", "depth_set", "depth_layers", 
                 "flap_type", "flap_name", "assoc", "keywords"]:
            df[c] = df[c].astype(str).fillna("")
        return df
        
    except Exception as e:
        st.error(f"Không đọc được rules CSV: {e}")
        return pd.DataFrame(columns=SUGG_REQUIRED_COLS)

def persist_rules_upload(up_file) -> Path:
    """Lưu file CSV đã upload để các phiên sau tự dùng."""
    try:
        RULES_EN_DEFAULT.write_bytes(up_file.getbuffer())
        load_rules_en.clear()
        return RULES_EN_DEFAULT
    except Exception as e:
        st.error(f"Lưu CSV thất bại: {e}")
        return RULES_EN_DEFAULT

def filter_rules_tree(df: pd.DataFrame, region: str, subunit: str, unit: str,
                     size_val: Optional[float], depth_sel: List[str],
                     layers_sel: List[str], assoc_sel: List[str], q: str) -> pd.DataFrame:
    v = df.copy()
    
    if region and region != "(Any)":
        v = v[v["region_en"].str.fullmatch(region, case=False, na=False)]
    if subunit and subunit != "(Any)":
        v = v[v["subunit_en"].str.fullmatch(subunit, case=False, na=False)]
    if unit and unit != "(Any)":
        v = v[v["unit"].str.lower() == unit.lower()]
    
    if size_val is not None and not pd.isna(size_val):
        v = v[(v["size_lo"].fillna(-1) <= float(size_val)) & 
              (float(size_val) <= v["size_hi"].fillna(1e9))]
    
    if depth_sel:
        v = v[v["depth_set"].str.contains(
            "|".join([re.escape(d) for d in depth_sel]), case=False, na=False)]
    
    if layers_sel:
        v = v[v["depth_layers"].str.contains(
            "|".join([re.escape(d) for d in layers_sel]), case=False, na=False)]
    
    if assoc_sel:
        v = v[v["assoc"].str.contains(
            "|".join([re.escape(d) for d in assoc_sel]), case=False, na=False)]
    
    if q:
        bag = (v["first_line"].fillna("") + " " + v["steps"].fillna("") + " " + 
               v["notes"].fillna("") + " " + v["pitfalls"].fillna("") + " " + 
               v["alternatives"].fillna("") + " " + v["keywords"].fillna(""))
        v = v[bag.str.lower().str.contains(q.lower(), na=False)]
    
    return v.sort_values(["priority", "flap_name"]).reset_index(drop=True)

# ========================= ENHANCED UI COMPONENTS =========================
def save_hidden_books_optimized(lst: List[str]):
    """Optimized hidden books saving without forcing rerun"""
    try:
        HIDDEN_BOOKS_JSON.write_text(
            json.dumps(sorted(set(lst)), ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        st.session_state.hidden_books = sorted(set(lst))
    except Exception:
        pass

def render_productivity_dashboard():
    """Enhanced productivity dashboard with session management"""
    if not st.session_state.ui_preferences.get("show_productivity_stats", True):
        return
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Thống kê phiên làm việc")
        
        # Session info
        session_duration = st.session_state.work_session_manager.get_session_duration()
        st.metric("Thời gian làm việc", session_duration)
        
        # Productivity metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Đã chỉnh sửa", 
                st.session_state.productivity_stats["images_edited_session"],
                help="Số ảnh đã chỉnh sửa trong phiên này"
            )
        
        with col2:
            total_edited = len(st.session_state.edited_images_set)
            st.metric("Tổng đã edit", total_edited, help="Tổng số ảnh đã chỉnh sửa")
        
        # Average time per edit
        if st.session_state.productivity_stats["avg_time_per_edit"] > 0:
            avg_time = st.session_state.productivity_stats["avg_time_per_edit"] / 60  # Convert to minutes
            st.metric("TB thời gian/ảnh", f"{avg_time:.1f} phút")
        
        # Quick actions
        if st.button("💾 Lưu phiên làm việc"):
            st.session_state.work_session_manager.save_session_snapshot()
            save_session_state()
            st.success("✅ Đã lưu!")

def render_image_card_enhanced(row: pd.Series, sel_list: List[str], seq: List[str], 
                               query: str = "", col_key: str = "") -> None:
    """Enhanced image card with edit status and better UX"""
    rel = row["image_path"]
    img_abs = DATA_ROOT / rel
    if not img_abs.exists():
        return

    # Get edit info
    edit_info = get_image_edit_info(rel)
    is_edited = edit_info["is_edited"]
    
    # Badge styling
    if st.session_state.ui_preferences.get("show_edited_badge", True):
        if is_edited:
            edit_count = edit_info["edit_count"]
            time_since = format_time_since_edit(edit_info["time_since_edit"])
            edited_badge = f" ✅ (x{edit_count}, {time_since})"
        else:
            edited_badge = " 🔄 chưa edit"
    else:
        edited_badge = ""

    with st.container(border=True):
        tp = row.get("thumb_path", "")
        show_img = DATA_ROOT / tp if tp and (DATA_ROOT / tp).exists() else img_abs
        st.image(str(show_img), use_container_width=True)
        
        st.markdown(
            f"<div class='meta'>📕 <b>{row['book_name']}</b> • 📄 p{int(row['page_num'])} • "
            f"🏷️ {row.get('group_key') or '-'}{edited_badge}</div>", 
            unsafe_allow_html=True
        )
        
        conf = (row.get("confidence") or "low").lower()
        badge_class = ('badge-h' if conf == 'high' else 
                      ('badge-m' if conf == 'medium' else 'badge-l'))
        
        # Priority indicator for unedited images
        priority_indicator = ""
        if not is_edited:
            priority_indicator = " 🎯"
        
        st.markdown(
            f"<span class='badge {badge_class}'>conf: {conf}</span> "
            f"<span class='badge'>{int(row.get('relevance_score') or 0)} pts</span>"
            f"{priority_indicator}", 
            unsafe_allow_html=True
        )
        st.caption(f"_{(row.get('caption') or '')[:120]}_")

        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            is_selected = rel in sel_list
        if st.checkbox("Chọn", key=f"pick::{rel}::{col_key}", value=is_selected):
            if not is_selected:
                st.session_state.selected_list.append(rel)
        elif is_selected:
            st.session_state.selected_list.remove(rel)
    
    with c2:
        with st.popover("📝 Note", use_container_width=True):
            # Load existing notes for this image
            df = md_load()
            existing_notes = ""
            img_row = df[df["image_path"] == rel]
            if not img_row.empty and "notes" in df.columns:
                existing_notes = img_row.iloc[0].get("notes", "")
            
            # Notes input form
            with st.form(f"notes_form_{rel}_{col_key}"):
                user_notes = st.text_area(
                    "Ghi chú cá nhân cho ảnh này:",
                    value=existing_notes,
                    height=120,
                    placeholder="Nhập ghi chú, quan sát, nhận xét về ảnh này..."
                )
                
                if st.form_submit_button("💾 Lưu ghi chú", type="primary"):
                    # Update notes in dataframe
                    df = md_load()
                    if "notes" not in df.columns:
                        df["notes"] = ""
                    
                    mask = df["image_path"] == rel
                    df.loc[mask, "notes"] = user_notes
                    md_save_immediate(df)
                    
                    # Mark as edited if notes added
                    if user_notes.strip():
                        mark_image_edited(rel, "notes_added")
                    
                    st.success("✅ Đã lưu ghi chú!")
            
            # Show preview of existing notes
            if existing_notes:
                st.markdown("**Ghi chú hiện tại:**")
                st.caption(existing_notes[:200] + "..." if len(existing_notes) > 200 else existing_notes)
    
    with c3:
        with st.popover("Sửa/Xoá", use_container_width=True):
            render_edit_controls_enhanced(row, rel, query, col_key)

def render_edit_controls_enhanced(row: pd.Series, rel: str, query: str, col_key: str):
    """Enhanced edit controls with edit status warning but allows re-editing."""
    edit_form_key = f"edit_form_{rel}_{col_key}_{hash(str(row))}"
    
    edit_info = get_image_edit_info(rel)
    
    # HIỂN THỊ CẢNH BÁO NẾU ĐÃ EDIT NHƯNG VẪN CHO PHÉP SỬA
    if edit_info["is_edited"]:
        st.warning("⚠️ Ảnh này đã được edit trước đó.")
        last_edit_time_str = format_time_since_edit(edit_info.get("time_since_edit", 0))
        st.write(f"Lần cuối edit: {last_edit_time_str}")
        st.caption("Bạn vẫn có thể chỉnh sửa lại nhưng hãy cẩn thận để đảm bảo tính nhất quán.")
    
    # HIỂN THỊ FORM EDIT CHO TẤT CẢ ẢNH
    with st.form(edit_form_key):
        new_caption = st.text_area("Caption", value=row.get("caption", ""))
        
        as_idx = (ANATOMY_OPTIONS.index(row.get("anatomical_site", "unknown")) 
                 if row.get("anatomical_site") in ANATOMY_OPTIONS else 0)
        ft_idx = (FLAP_OPTIONS.index(row.get("flap_type", "unknown")) 
                 if row.get("flap_type") in FLAP_OPTIONS else 0)
        
        sel_site = st.selectbox("Vị trí", ANATOMY_OPTIONS, index=as_idx)
        sel_flap = st.selectbox("Loại vạt", FLAP_OPTIONS, index=ft_idx)
        new_page = st.number_input("Đổi trang", min_value=1, value=int(row["page_num"]))
        new_group = st.text_input("Group (fig_/p...)", value=row.get("group_key", ""))
        
        # XEM CHI TIẾT ẢNH
        if st.toggle("Xem chi tiết ảnh"):
            # Hiển thị ảnh lớn
            img_abs = DATA_ROOT / rel
            if img_abs.exists():
                st.image(str(img_abs), caption=f"Ảnh gốc: {Path(rel).name}", use_container_width=True)
            
            # Thông tin chi tiết
            st.markdown("**📋 Thông tin chi tiết:**")
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.write(f"**📕 Sách:** {row.get('book_name', 'N/A')}")
                st.write(f"**📄 Trang:** {int(row.get('page_num', 0))}")
                st.write(f"**🏷️ Group:** {row.get('group_key', 'N/A')}")
                st.write(f"**🎯 Độ tin cậy:** {row.get('confidence', 'unknown')}")
                
            with info_cols[1]:
                st.write(f"**🎪 Vị trí giải phẫu:** {row.get('anatomical_site', 'unknown')}")
                st.write(f"**🔧 Loại vạt:** {row.get('flap_type', 'unknown')}")
                st.write(f"**📊 Điểm liên quan:** {int(row.get('relevance_score', 0))}")
                st.write(f"**📅 Lưu lúc:** {row.get('saved_at', 'N/A')}")
            
            # Ngữ cảnh
            if row.get("context"):
                st.markdown("**📝 Ngữ cảnh:**")
                ctx = highlight(row.get("context", ""), query)
                st.markdown(ctx, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        if col1.form_submit_button("💾 Lưu thay đổi", type="primary"):
            updates = {
                "caption": new_caption, 
                "anatomical_site": sel_site, 
                "flap_type": sel_flap, 
                "group_key": new_group
            }
            
            if int(new_page) != int(row["page_num"]):
                src = DATA_ROOT / rel
                book_dir = DATA_ROOT / safe_book_name(row["book_name"]) / f"p{int(new_page)}"
                ensure_dir(book_dir)
                dst = unique_filename(book_dir / src.name)
                
                if dst.resolve() != src.resolve():
                    shutil.move(str(src), str(dst))
                
                updates["image_path"] = str(dst.relative_to(DATA_ROOT))
                updates["page_num"] = int(new_page)
                new_t = thumb_path_for(updates["image_path"])
                make_thumb(dst, new_t)
                updates["thumb_path"] = str(new_t.relative_to(DATA_ROOT)) if new_t.exists() else ""
            
            md_update_by_paths_batch({rel: updates})
            mark_image_edited(rel, "manual_edit")
            st.success("✅ Đã lưu!")
        
        if col2.form_submit_button("🗑️ Xoá ảnh"):
            (DATA_ROOT / rel).unlink(missing_ok=True)
            tpr = row.get("thumb_path", "")
            if tpr:
                (DATA_ROOT / tpr).unlink(missing_ok=True)
            
            md_delete_by_paths_batch([rel])
            if rel in st.session_state.selected_list:
                st.session_state.selected_list.remove(rel)
            
            if rel in st.session_state.edited_images_set:
                st.session_state.edited_images_set.remove(rel)
            if rel in st.session_state.last_visit_times:
                del st.session_state.last_visit_times[rel]
            if rel in st.session_state.edit_timestamps:
                del st.session_state.edit_timestamps[rel]
            
            st.success("✅ Đã xoá!")

def apply_enhanced_sorting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply enhanced sorting with edit status priority"""
    if df.empty:
        return df
    
    # Add edit status columns
    df = df.copy()
    df["is_edited"] = df["image_path"].apply(lambda x: x in st.session_state.edited_images_set)
    df["last_visit"] = df["image_path"].apply(lambda x: st.session_state.last_visit_times.get(x, 0))
    
    # Apply quick filters
    quick_filters = st.session_state.quick_filters
    if quick_filters.get("show_only_unedited"):
        df = df[~df["is_edited"]]
    elif quick_filters.get("show_only_edited"):
        df = df[df["is_edited"]]
    elif quick_filters.get("show_recent_edits"):
        recent_threshold = time.time() - (24 * 3600)  # Last 24 hours
        df = df[df["last_visit"] > recent_threshold]
    
    # Sort: unedited first (priority), then by last visit time
    if st.session_state.ui_preferences.get("sort_edited_first", False):
        # Edited first mode
        df = df.sort_values(
            ["is_edited", "last_visit"], 
            ascending=[False, False]
        )
    else:
        # Unedited first mode (default - better for workflow)
        df = df.sort_values(
            ["is_edited", "last_visit"], 
            ascending=[True, False]
        )
    
    return df.drop(columns=["is_edited", "last_visit"])

def render_lightbox_enhanced():
    """Enhanced lightbox with edit capabilities"""
    if not (st.session_state.get("lightbox_open") and st.session_state.get("lightbox_seq")):
        return
    
    seq = st.session_state.lightbox_seq
    idx = st.session_state.lightbox_idx
    idx = max(0, min(idx, len(seq) - 1))
    rel = seq[idx]
    p = DATA_ROOT / rel
    
    if not p.exists():
        st.session_state.lightbox_open = False
        return
    
    st.markdown("<div class='lightbox'><div class='lightbox-inner'>", unsafe_allow_html=True)
    st.image(str(p), use_container_width=True)
    
    # Get image info
    df = md_load()
    row = df[df["image_path"] == rel]
    
    if not row.empty:
        r = row.iloc[0]
        edit_info = get_image_edit_info(rel)
        
        # Enhanced caption with edit status
        cap = r["caption"]
        edit_status = "✅ Đã edit" if edit_info["is_edited"] else "🔄 Chưa edit"
        
        st.markdown(f"<div class='lightbox-caption'><strong>{cap}</strong><br>"
                   f"<small>{r['book_name']} • p{r['page_num']} • {edit_status}</small></div>", 
                   unsafe_allow_html=True)
    
    # Navigation with edit shortcuts
    c1, c2, c3, c4 = st.columns([1, 3, 2, 1])
    
    with c1:
        if st.button("⟵ Prev", key="lb_prev"):
            st.session_state.lightbox_idx = (idx - 1) % len(seq)
    
    with c2:
        if st.button("✖ Close", use_container_width=True, key="lb_close"):
            st.session_state.lightbox_open = False
    
    with c3:
        if not row.empty and not edit_info["is_edited"]:
            if st.button("⚡ Quick Edit", use_container_width=True, key="lb_quick_edit"):
                mark_image_edited(rel, "lightbox_quick")
                st.success("Đã đánh dấu đã chỉnh sửa!")
                st.rerun()
    
    with c4:
        if st.button("Next ⟶", key="lb_next"):
            st.session_state.lightbox_idx = (idx + 1) % len(seq)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# ========================= MAIN APPLICATION =========================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🖼️", layout="wide")
    
    # Custom CSS with enhanced styling
    st.markdown("""
    <style>
    :root { --fz-lg:16.5px; --fz-sm:13.5px; }
    .block-container{padding-top:0.6rem; padding-bottom:1.2rem; font-size:var(--fz-lg);}
    h1,h2,h3 { letter-spacing:.2px }
    .meta { color:#666; font-size:var(--fz-sm); margin:4px 0 8px; }
    .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;}
    .badge-h{background:#e8f5e9;color:#1b5e20}.badge-m{background:#fff3e0;color:#e65100}.badge-l{background:#ffebee;color:#b71c1c}
    .sel-strip{ position:sticky; top:52px; z-index:99; background:#fff; border:1px dashed #dfe3e7; border-radius:10px; padding:6px 8px; margin:6px 0 12px; display:flex; gap:8px; flex-wrap:wrap; align-items:center }
    .sel-thumb{display:flex; align-items:center; gap:6px; border:1px solid #eee; padding:2px 6px; border-radius:6px;}
    .sel-thumb img{ height:42px; width:auto; border-radius:4px; border:1px solid #f0f0f0 }
    .lightbox{position:fixed;left:0;top:0;width:100vw;height:100vh;background:rgba(0,0,0,.8);z-index:1000;display:flex;align-items:center;justify-content:center;}
    .lightbox-inner{width:min(96vw,1200px); background:#111; padding:8px; border-radius:8px;}
    .lightbox-caption{color:#ddd; font-size:14px; margin-top:8px;}
    .priority-badge{background:#ff9800;color:white;padding:1px 4px;border-radius:3px;font-size:10px;margin-left:4px;}
    .edited-badge{background:#4caf50;color:white;padding:1px 4px;border-radius:3px;font-size:10px;margin-left:4px;}
    .unedited-badge{background:#f44336;color:white;padding:1px 4px;border-radius:3px;font-size:10px;margin-left:4px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Process any pending updates
    if st.session_state.defer_rerun or st.session_state.pending_updates:
        process_pending_updates()
    
    # Auto-save session periodically
    if st.session_state.ui_preferences.get("auto_save_enabled", True):
        save_session_state()
    
    # Render productivity dashboard in sidebar
    render_productivity_dashboard()
    
    st.title("🖼️ " + APP_TITLE)
    st.caption(f"Data: `{DATA_ROOT}` • Metadata: {'Parquet (pyarrow)' if _parquet_available() else 'CSV'}")
    
    # Session info header
    if st.session_state.ui_preferences.get("show_productivity_stats", True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Phiên làm việc", st.session_state.work_session_manager.get_session_duration())
        with col2:
            st.metric("Đã edit (phiên)", st.session_state.productivity_stats["images_edited_session"])
        with col3:
            st.metric("Tổng đã edit", len(st.session_state.edited_images_set))
        with col4:
            if st.button("⚙️ Tùy chỉnh UI"):
                st.session_state.show_ui_preferences = not st.session_state.get("show_ui_preferences", False)

    # UI Preferences Panel
    if st.session_state.get("show_ui_preferences", False):
        with st.expander("🎛️ Tùy chỉnh giao diện", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state.ui_preferences["show_edited_badge"] = st.checkbox(
                    "Hiển thị badge đã edit", 
                    value=st.session_state.ui_preferences.get("show_edited_badge", True)
                )
                st.session_state.ui_preferences["sort_edited_first"] = st.checkbox(
                    "Ảnh đã edit lên đầu", 
                    value=st.session_state.ui_preferences.get("sort_edited_first", False),
                    help="Nếu tắt: ảnh chưa edit sẽ lên đầu (khuyến nghị cho workflow)"
                )
            
            with col2:
                st.session_state.ui_preferences["auto_save_enabled"] = st.checkbox(
                    "Tự động lưu", 
                    value=st.session_state.ui_preferences.get("auto_save_enabled", True)
                )
                st.session_state.ui_preferences["show_productivity_stats"] = st.checkbox(
                    "Hiển thị thống kê", 
                    value=st.session_state.ui_preferences.get("show_productivity_stats", True)
                )
            
            with col3:
                st.markdown("**Bộ lọc nhanh:**")
                st.session_state.quick_filters["show_only_unedited"] = st.checkbox(
                    "Chỉ ảnh chưa edit", 
                    value=st.session_state.quick_filters.get("show_only_unedited", False)
                )
                st.session_state.quick_filters["show_only_edited"] = st.checkbox(
                    "Chỉ ảnh đã edit", 
                    value=st.session_state.quick_filters.get("show_only_edited", False)
                )
                st.session_state.quick_filters["show_recent_edits"] = st.checkbox(
                    "Chỉ edit gần đây (24h)", 
                    value=st.session_state.quick_filters.get("show_recent_edits", False)
                )
            
            if st.button("💾 Lưu tùy chỉnh"):
                save_session_state()
                st.success("✅ Đã lưu tùy chỉnh!")
    
    # Main tabs
    tab_extract, tab_web, tab_clinical, tab_library, tab_suggest, tab_settings = st.tabs([
        "📥 PDF", "🌐 Web (HTML)", "🧑‍⚕️ Ảnh lâm sàng", 
        "📚 Thư viện", "💡 Quyết định", "⚙️ Cài đặt"
    ])
    
    # -------------------- PDF TAB --------------------
    with tab_extract:
        st.subheader("Trích xuất ảnh từ PDF")
        up = st.file_uploader("Chọn PDF (≤ 200MB)", type=["pdf"])
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            min_px = st.number_input("Lọc ảnh nhỏ hơn (px)", 100, 3000, 
                                   MIN_IMG_SIZE_DEFAULT, step=20)
        with c2:
            allow_dup = st.checkbox("Cho phép trùng ảnh (MD5)", value=False)
        with c3:
            save_all = st.checkbox("Lưu tất cả (bỏ điểm liên quan)", value=SAVE_ALL_FALLBACK)
        with c4:
            nearby_ratio = st.slider("Bán kính caption quanh ảnh", 0.05, 0.35, 
                                   NEARBY_RATIO_DEFAULT, 0.01)
        with c5:
            min_score_pdf = st.slider("Ngưỡng điểm liên quan", 0, 60, 12)
        
        if st.button("🚀 BẮT ĐẦU", type="primary", disabled=up is None):
            if up and up.size > 200 * 1024 * 1024:
                st.error("File vượt 200MB.")
            else:
                ph = st.empty()
                pb = st.progress(0.0)
                
                def _upd(p, msg):
                    pb.progress(min(max(float(p), 0.0), 1.0))
                    ph.info(msg)
                
                with st.spinner("Đang phân tích & trích xuất..."):
                    paths, n, book_dir = process_pdf(
                        up, min_px=min_px, allow_duplicates=allow_dup, 
                        save_all_if_no_kw=save_all, nearby_ratio=nearby_ratio, 
                        relevance_threshold=int(min_score_pdf), progress=_upd
                    )
                
                pb.progress(1.0)
                ph.success("Hoàn tất.")
                
                if n > 0:
                    st.success(f"✅ Đã lưu {n} ảnh vào thư mục:")
                    st.code(str(book_dir))
                else:
                    st.warning("Không thấy ảnh phù hợp (hãy nới lỏng bộ lọc hoặc hạ ngưỡng điểm).")

    # -------------------- WEB TAB --------------------
    with tab_web:
        st.subheader("Nhập ảnh từ trang web (HTML parsing)")
        url = st.text_input("URL", placeholder="https://example.com/article")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            min_px_w = st.number_input("Lọc ảnh nhỏ hơn (px)", 80, 3000, 
                                     MIN_IMG_SIZE_DEFAULT, step=20, key="min_px_web")
        with c2:
            allow_dup_w = st.checkbox("Trùng ảnh (MD5)", value=False, key="allow_dup_web")
        with c3:
            save_all_w = st.checkbox("Lưu tất cả", value=True, key="save_all_web")
        with c4:
            min_score_web = st.slider("Ngưỡng điểm liên quan", 0, 60, 10, key="min_score_web")
        
        if st.button("🌐 TẢI ẢNH", type="primary", disabled=not bool(url.strip())):
            with st.spinner(f"Đang tải & phân tích HTML từ {url.strip()}..."):
                kept, _, out_dir = ingest_web_html(
                    url.strip(), min_px=int(min_px_w), allow_duplicates=allow_dup_w,
                    save_all_if_no_kw=save_all_w, min_score=int(min_score_web)
                )
            
            if kept > 0:
                st.success(f"✅ Đã lưu {kept} ảnh vào: {out_dir}")
            else:
                st.warning("Không lấy được ảnh phù hợp. Kiểm tra URL/hạ ngưỡng điểm.")

        st.markdown("---")
        st.markdown("Hoặc **nhập nhiều URL** (mỗi dòng một URL):")
        urls_bulk = st.text_area("Danh sách URL", height=120, placeholder="https://...\nhttps://...\n")
        
        if st.button("🧺 TẢI LOT URL", disabled=not bool(urls_bulk.strip())):
            totals = 0
            url_list = [u.strip() for u in (urls_bulk or "").splitlines() if u.strip()]
            prog = st.progress(0.0, "Bắt đầu...")
            
            for i, u in enumerate(url_list):
                prog.progress((i) / len(url_list), f"Đang xử lý {i+1}/{len(url_list)}")
                try:
                    kept, _, _ = ingest_web_html(
                        u, min_px=int(min_px_w), allow_duplicates=allow_dup_w,
                        save_all_if_no_kw=save_all_w, min_score=int(min_score_web)
                    )
                    totals += kept
                except Exception:
                    continue
            
            prog.progress(1.0, "Hoàn tất!")
            
            if totals > 0:
                st.success(f"✅ Tổng cộng lưu được {totals} ảnh.")
            else:
                st.warning("Không lưu được ảnh nào từ danh sách URL.")

    # -------------------- CLINICAL TAB --------------------
    with tab_clinical:
        st.subheader("Upload ảnh lâm sàng trực tiếp")
        files = st.file_uploader(
            "Chọn ảnh (png/jpg/jpeg/webp, nhiều file)", 
            type=["png", "jpg", "jpeg", "webp"], 
            accept_multiple_files=True
        )
        
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            case_name = st.text_input("Case/Folder", value="Case001")
        with c2:
            caption_prefix = st.text_input("Caption prefix", value="")
        with c3:
            site_pick = st.selectbox("Site", ["(Giữ nguyên)"] + ANATOMY_OPTIONS)
        with c4:
            flap_pick = st.selectbox("Flap", ["(Giữ nguyên)"] + FLAP_OPTIONS)
        
        allow_dup_c = st.checkbox("Cho phép trùng MD5", value=False)
        
        if st.button("⬆️ TẢI ẢNH", type="primary", disabled=not files):
            with st.spinner("Đang lưu ảnh lâm sàng..."):
                added = ingest_clinical_images(
                    files, case_name=case_name, caption_prefix=caption_prefix,
                    site=site_pick, flap=flap_pick, allow_duplicates=allow_dup_c
                )
            
            if added > 0:
                st.success(f"✅ Đã thêm {added} ảnh vào thư mục: "
                         f"{CLINICAL_DIR / safe_book_name(case_name)}")
            else:
                st.warning("Không thêm được ảnh nào (có thể do trùng).")

    # -------------------- ENHANCED LIBRARY TAB --------------------
    with tab_library:
        df = md_load()
        
        if df.empty:
            st.info("Chưa có dữ liệu. Hãy trích xuất từ PDF/Web hoặc upload lâm sàng.", icon="📥")
        else:
            # Sidebar filters
            with st.sidebar:
                st.header("🔍 Lọc & Tìm")
                
                # Prefill filters from decision assist
                preset = st.session_state.get("library_prefill") or {}
                all_books = sorted(df["book_name"].dropna().unique().tolist())
                hidden_books = set(st.session_state.hidden_books)
                visible_books = [b for b in all_books if b not in hidden_books]

                fb = st.selectbox("Sách/Domain", ["(All)"] + visible_books, index=0, key="lib_book")
                fp = st.selectbox("Trang", ["(All)"] + sorted(df["page_num"].dropna().astype(int).unique().tolist()), key="lib_page")
                fgroup = st.text_input("Group (fig_ / p...)", value=preset.get("group", ""), key="lib_group")
                
                # Multi-select for anatomical sites
                fsite_multi = st.multiselect(
                    "Vị trí (có thể chọn nhiều)", 
                    ANATOMY_OPTIONS,
                    default=preset.get("site_multi", []),
                    key="lib_site_multi"
                )
                
                fflap = st.selectbox("Loại vạt", ["(All)"] + FLAP_OPTIONS, 
                                   index=(["(All)"] + FLAP_OPTIONS).index(preset.get("flap", "(All)")), key="lib_flap")
                fsrc = st.selectbox("Nguồn", ["(All)", "PDF only", "Web only", "Clinical only"], key="lib_src")
                
                # Enhanced filter by edited status
                fedited = st.selectbox(
                    "Trạng thái chỉnh sửa", 
                    ["(All)", "Chưa chỉnh sửa", "Đã chỉnh sửa", "Edit gần đây (24h)"], 
                    key="lib_edited"
                )
                
                fkw = st.text_input("Keyword (caption/context)", value=preset.get("kw", ""), 
                                  key="lib_kw", placeholder="vd: bilobed, nasal tip...")
                
                c1, c2 = st.columns(2)
                if c1.button("↺ Reset"):
                    for k in ["lib_book", "lib_page", "lib_group", "lib_site_multi", "lib_flap", "lib_src", "lib_edited", "lib_kw"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state.library_prefill = {}
                    st.rerun()
                
                if fb != "(All)" and fb not in hidden_books:
                    if c2.button(f"🙈 Ẩn '{fb}'"):
                        hidden_books.add(fb)
                        save_hidden_books_optimized(list(hidden_books))
                        st.rerun()

            # Apply filters
            view = df[df["book_name"].isin(visible_books)].copy()
            if fb != "(All)":
                view = view[view["book_name"] == fb]
            if fp != "(All)":
                view = view[view["page_num"] == int(fp)]
            if fgroup:
                view = view[view["group_key"].astype(str).str.contains(fgroup, case=False, na=False)]
            
            # Multi-select site filter
            if fsite_multi:
                view = view[view["anatomical_site"].isin(fsite_multi)]
            
            if fflap != "(All)":
                view = view[view["flap_type"] == fflap]

            if fsrc == "PDF only":
                view = view[view["source"].astype(str).str.startswith("pdf", na=False)]
            elif fsrc == "Web only":
                view = view[view["source"].astype(str).str.startswith("web", na=False)]
            elif fsrc == "Clinical only":
                view = view[view["source"].astype(str).str.startswith("clinical", na=False)]

            # Enhanced edited status filter
            if fedited == "Chưa chỉnh sửa":
                view = view[~view["image_path"].isin(st.session_state.edited_images_set)]
            elif fedited == "Đã chỉnh sửa":
                view = view[view["image_path"].isin(st.session_state.edited_images_set)]
            elif fedited == "Edit gần đây (24h)":
                recent_threshold = time.time() - (24 * 3600)
                recent_images = [img for img, last_time in st.session_state.last_visit_times.items() 
                               if last_time > recent_threshold]
                view = view[view["image_path"].isin(recent_images)]

            if fkw:
                bag = (view["caption"].fillna("") + " " + view["context"].fillna("")).str.lower()
                view = view[bag.str.lower().str.contains(fkw.lower(), na=False)]

            # Apply enhanced sorting
            view = apply_enhanced_sorting(view)
            
            # Enhanced statistics
            total_images = len(df)
            edited_images = len(st.session_state.edited_images_set)
            unedited_images = total_images - edited_images
            completion_rate = (edited_images / total_images * 100) if total_images > 0 else 0
            
            st.caption(f"Tìm thấy **{len(view)}** ảnh • "
                      f"Đã edit: **{edited_images}** • "
                      f"Chưa edit: **{unedited_images}** • "
                      f"Tiến độ: **{completion_rate:.1f}%** • "
                      f"Đang ẩn **{len(hidden_books)}** sách")
            
            # Progress bar for completion
            if total_images > 0:
                # Giới hạn giá trị tối đa là 1.0 để tránh lỗi
                progress_value = min(completion_rate / 100, 1.0)
                st.progress(progress_value, f"Tiến độ chỉnh sửa: {edited_images}/{total_images}")
            
            st.session_state.library_prefill = {}  # clear after use

            # Enhanced selection strip
            sel_list: List[str] = st.session_state.selected_list
            if sel_list:
                st.markdown("<div class='sel-strip'>", unsafe_allow_html=True)
                st.write(f"🎯 Đang chọn: {len(sel_list)} ảnh (hiển thị tối đa 18 thumb)")
                
                thumb_cols = st.columns(min(6, len(sel_list[:18])))
                for i, rel in enumerate(sel_list[:18]):
                    with thumb_cols[i % len(thumb_cols)]:
                        p = DATA_ROOT / rel
                        t = thumb_path_for(rel)
                        show = t if t.exists() else p
                        try:
                            edit_info = get_image_edit_info(rel)
                            status_icon = "✅" if edit_info["is_edited"] else "🔄"
                            st.image(str(show), caption=f"{status_icon} {Path(rel).name[:12]}", width=60)
                        except Exception:
                            pass
                
                c1, c2, c3 = st.columns([1, 1, 1])
                if c1.button("Bỏ chọn tất cả"):
                    st.session_state.selected_list = []
                    st.rerun()
                if c2.button("Làm mới thư viện"):
                    md_load_cached.clear()
                    st.rerun()
                if c3.button("⚡ Đánh dấu đã edit"):
                    for rel in sel_list:
                        mark_image_edited(rel, "bulk_mark")
                    st.success(f"✅ Đã đánh dấu {len(sel_list)} ảnh!")
                    st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)

            # Pagination
            page_size = 12
            total_pages = max(1, (len(view) - 1) // page_size + 1)
            page_idx = st.number_input("Trang thư viện", min_value=1, max_value=total_pages, value=1)
            
            s, e = (page_idx - 1) * page_size, min(page_idx * page_size, len(view))
            current_page_paths = view.iloc[s:e]["image_path"].tolist()

            cstrip1, cstrip2, cstrip3 = st.columns([1.4, 1.4, 2])
            if cstrip1.button(f"Chọn tất cả {len(current_page_paths)} (trang)"):
                st.session_state.selected_list = sorted(list(set(sel_list + current_page_paths)))
            if cstrip2.button("Bỏ chọn trang này"):
                st.session_state.selected_list = [x for x in sel_list if x not in set(current_page_paths)]
            
            # Quick stats for current page
            page_unedited = [p for p in current_page_paths 
                           if p not in st.session_state.edited_images_set]
            cstrip3.metric("Chưa edit (trang này)", len(page_unedited))

            # Enhanced batch operations
            with st.expander("🛠️ Thao tác hàng loạt", expanded=bool(sel_list)):
                with st.form("batch_operations_form"):
                    b1, b2, b3, b4, b5 = st.columns([1.3, 1.1, 1.1, 1.1, 1])
                    
                    with b1:
                        cap_mode = st.selectbox("Caption", ["(Giữ nguyên)", "Thay thế toàn bộ", "Thêm tiền tố", "Thêm hậu tố"])
                        cap_text = st.text_input("Nội dung caption", value="")
                    with b2:
                        site_pick_b = st.selectbox("Vị trí", ["(Giữ nguyên)"] + ANATOMY_OPTIONS)
                    with b3:
                        flap_pick_b = st.selectbox("Loại vạt", ["(Giữ nguyên)"] + FLAP_OPTIONS)
                    with b4:
                        group_pick_b = st.text_input("Group (optional)")
                    with b5:
                        new_page_all = st.number_input("Chuyển trang →", min_value=1, value=1)
                    
                    apply_batch = st.form_submit_button("✅ ÁP DỤNG", type="primary", disabled=not sel_list)

               
                if apply_batch:
                    if not sel_list:
                        st.warning("Chưa chọn ảnh nào.")
                    else:
                        # TÁCH NHỮNG ẢNH CHƯA EDIT ĐỂ XỬ LÝ
                        unedited_to_process = [p for p in sel_list if not get_image_edit_info(p)["is_edited"]]
                        edited_skipped_count = len(sel_list) - len(unedited_to_process)

                        if not unedited_to_process:
                            st.warning("Tất cả ảnh đã chọn đều đã được edit trước đó. Không có gì thay đổi.")
                        else:
                            with st.spinner(f"Đang áp dụng cho {len(unedited_to_process)} ảnh..."):
                                df_all = md_load()
                                # CHỈ ÁP DỤNG CHO NHỮNG ẢNH CHƯA EDIT
                                mask = df_all["image_path"].isin(unedited_to_process)
                                
                                # Apply caption changes
                                if cap_mode != "(Giữ nguyên)":
                                    if cap_mode == "Thay thế toàn bộ":
                                        df_all.loc[mask, "caption"] = cap_text
                                    elif cap_mode == "Thêm tiền tố":
                                        df_all.loc[mask, "caption"] = cap_text + df_all.loc[mask, "caption"].fillna("").astype(str)
                                    elif cap_mode == "Thêm hậu tố":
                                        df_all.loc[mask, "caption"] = df_all.loc[mask, "caption"].fillna("").astype(str) + cap_text
                                
                                if site_pick_b != "(Giữ nguyên)":
                                    df_all.loc[mask, "anatomical_site"] = site_pick_b
                                if flap_pick_b != "(Giữ nguyên)":
                                    df_all.loc[mask, "flap_type"] = flap_pick_b
                                if group_pick_b.strip():
                                    df_all.loc[mask, "group_key"] = group_pick_b.strip()
                                
                                # Handle page moves
                                moved = 0
                                if isinstance(new_page_all, int) and new_page_all > 0:
                                    # CHỈ DI CHUYỂN NHỮNG ẢNH CHƯA EDIT
                                    for rel in unedited_to_process:
                                        rows = df_all[df_all["image_path"] == rel]
                                        if rows.empty: continue
                                        r = rows.iloc[0]
                                        
                                        if int(r["page_num"]) != int(new_page_all):
                                            src = DATA_ROOT / rel
                                            book_dir = DATA_ROOT / safe_book_name(r["book_name"]) / f"p{int(new_page_all)}"
                                            ensure_dir(book_dir)
                                            dst = unique_filename(book_dir / src.name)
                                            
                                            if dst.resolve() != src.resolve():
                                                try:
                                                    shutil.move(str(src), str(dst))
                                                    moved += 1
                                                except Exception: continue
                                            
                                            new_rel = str(dst.relative_to(DATA_ROOT))
                                            df_all.loc[df_all["image_path"] == rel, "image_path"] = new_rel
                                            df_all.loc[df_all["image_path"] == new_rel, "page_num"] = int(new_page_all)
                                            t = thumb_path_for(new_rel)
                                            make_thumb(dst, t)
                                            df_all.loc[df_all["image_path"] == new_rel, "thumb_path"] = str(t.relative_to(DATA_ROOT)) if t.exists() else ""
                                
                                # Mark all processed images as edited
                                for rel in unedited_to_process:
                                    mark_image_edited(rel, "batch_operation")
                                
                                md_save_immediate(df_all)
                            
                            st.success(f"✅ Xong! Đã áp dụng cho {len(unedited_to_process)} ảnh. (di chuyển: {moved})")
                            if edited_skipped_count > 0:
                                st.info(f"ℹ️ Đã bỏ qua {edited_skipped_count} ảnh vì chúng đã được edit trước đó.")
                            time.sleep(0.2)
                            st.rerun()            

                st.divider()
                
                with st.form("delete_selected_form"):
                    if st.form_submit_button("🗑️ Xoá các ảnh đã chọn", disabled=not sel_list):
                        md_delete_by_paths_batch(sel_list)
                        
                        for rel in list(sel_list):
                            p = DATA_ROOT / rel
                            t = thumb_path_for(rel)
                            try:
                                if p.exists():
                                    p.unlink()
                                if t.exists():
                                    t.unlink()
                            except Exception:
                                pass
                        
                        st.session_state.selected_list = []
                        st.success("Đã xoá các ảnh đã chọn.")
                        time.sleep(0.3)
                        st.rerun()

            # Enhanced image grid with edit status
            cols = st.columns(3)
            seq = view["image_path"].tolist()
            
            for i, (_, row) in enumerate(view.iloc[s:e].iterrows()):
                with cols[i % 3]:
                    render_image_card_enhanced(row, st.session_state.selected_list, seq, 
                                              query=fkw or "", col_key=f"lib_{i}")

            render_lightbox_enhanced()

    # -------------------- DECISION ASSIST TAB --------------------
    with tab_suggest:
        st.subheader("💡 Hỗ trợ quyết định & Flap Suggestion")
        
        col_up, col_info = st.columns([1, 2])
        with col_up:
            up_rules = st.file_uploader(
                "Upload (một lần) file flap_rules_en.csv", type=["csv"], 
                help="Ứng dụng sẽ lưu lại cho các phiên sau"
            )
            if up_rules is not None:
                p = persist_rules_upload(up_rules)
                st.success("✅ Đã lưu CSV quy tắc. Bạn có thể tải trang để dùng ngay.")
        
        with col_info:
            if RULES_EN_DEFAULT.exists():
                st.success(f"Đang dùng quy tắc: {RULES_EN_DEFAULT.name} (persist)")
            else:
                st.warning("Chưa có CSV quy tắc. Hãy upload để kích hoạt gợi ý.")

        rules = load_rules_en(RULES_EN_DEFAULT)
        if rules.empty:
            st.info("Chưa có rules CSV hợp lệ.", icon="ℹ️")
        else:
            # Tree selector
            r1, r2, r3 = st.columns(3)
            
            regions = ["(Any)"] + sorted([x for x in rules["region_en"].dropna().unique().tolist() if x])
            region = r1.selectbox("Region", regions, index=0, key="ds_region")

            subunits_avail = (sorted(rules[rules["region_en"].str.fullmatch(region, case=False, na=False)]["subunit_en"].dropna().unique().tolist()) 
                            if region != "(Any)" else sorted(rules["subunit_en"].dropna().unique().tolist()))
            subunits = ["(Any)"] + subunits_avail
            subunit = r2.selectbox("Subunit", subunits, index=0, key="ds_subunit")

            units_avail = sorted(rules["unit"].dropna().unique().tolist())
            unit = r3.selectbox("Unit", ["(Any)"] + units_avail, index=0, key="ds_unit")

            csize, cdepth, clayer = st.columns(3)
            
            if unit == "%":
                size_val = csize.slider("Size (%)", 0, 100, 20)
            elif unit == "cm":
                lo = int(rules["size_lo"].min(skipna=True) or 0)
                hi = float(rules["size_hi"].max(skipna=True) or 5.0)
                size_val = float(csize.number_input("Size (cm)", min_value=0.0, 
                                                  max_value=max(hi, 0.1), value=min(1.0, max(0.0, hi))))
            else:
                size_val = None

            depth_opts = sorted(set("|".join(rules["depth_set"].astype(str)).split("|")) - set([""]))
            depth_sel = cdepth.multiselect("Depth set", depth_opts)

            layer_opts = sorted(set("|".join(rules["depth_layers"].astype(str)).split("|")) - set([""]))
            layers_sel = clayer.multiselect("Depth layers", layer_opts)

            assoc_opts = sorted(set("|".join(rules["assoc"].dropna().astype(str)).split("|")) - set([""]))
            assoc_sel = st.multiselect("Yếu tố kèm theo (assoc)", assoc_opts)

            q = st.text_input("Từ khoá tự do (keywords, steps...)", 
                            placeholder="bilobed, nasal tip, cartilage graft...")

            view_rules = filter_rules_tree(rules, region, subunit, unit, size_val, 
                                         depth_sel, layers_sel, assoc_sel, q)

            st.caption(f"🎯 Tìm thấy **{len(view_rules)}** gợi ý phù hợp (sort theo priority)")
            topN = view_rules.head(30)

            if topN.empty:
                st.info("Không có gợi ý phù hợp với tiêu chí hiện tại.")
            else:
                for _, rr in topN.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{rr['flap_name']}** — *{rr['flap_type']}* &nbsp; "
                                  f"<span class='badge'>prio {int(rr['priority'])}</span>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"> {rr['first_line']}")
                        
                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown("**Steps**")
                            st.write(rr["steps"])
                            st.markdown("**Notes**")
                            st.write(rr["notes"])
                        with cB:
                            st.markdown("**Pitfalls**")
                            st.write(rr["pitfalls"])
                            st.markdown("**Alternatives**")
                            st.write(rr["alternatives"])
                            st.caption(f"Assoc: {rr['assoc']}  •  Keywords: {rr['keywords']}  •  Refs: {rr['refs']}")

                        # Inline similar cases + push to library
                        kw_for_search = (rr['flap_name'] or rr['keywords'] or rr['flap_type']).split("|")[0]
                        btn_col1, btn_col2 = st.columns([1, 2])
                        
                        if btn_col1.button("🔎 Xem ca tương tự (inline)", 
                                         key=f"inline_sim::{rr['flap_name']}::{rr['subunit_en']}"):
                            df_all = md_load()
                            bag = (df_all["caption"].fillna("") + " " + df_all["context"].fillna("")).str.lower()
                            mask = bag.str.contains(str(kw_for_search).lower(), na=False)
                            
                            if isinstance(rr["region_en"], str) and rr["region_en"]:
                                mask &= df_all["anatomical_site"].str.contains(
                                    rr["region_en"].split("/")[0].lower().split()[0], na=False
                                )
                            
                            subset = df_all[mask].sort_values("saved_at", ascending=False).head(12)
                            
                            if subset.empty:
                                st.info("Chưa có ca tương tự trong thư viện.")
                            else:
                                cols = st.columns(4)
                                seq = subset["image_path"].tolist()
                                for i, (_, row) in enumerate(subset.iterrows()):
                                    with cols[i % 4]:
                                        render_image_card_enhanced(row, st.session_state.selected_list, 
                                                                   seq, query=str(kw_for_search), 
                                                                   col_key=f"suggest_{i}")

                        if btn_col2.button("📚 Mở trong Thư viện", 
                                         key=f"push_lib::{rr['flap_name']}::{rr['subunit_en']}"):
                            st.session_state.library_prefill = {
                                "kw": kw_for_search,
                                "site_multi": [rr["region_en"].lower().split("/")[0]] if isinstance(rr["region_en"], str) else [],
                                "flap": "(All)",
                                "group": ""
                            }
                            st.info("Đã đẩy bộ lọc sang Thư viện (chuyển sang tab 📚 để xem).")

    # -------------------- ENHANCED SETTINGS TAB --------------------
    with tab_settings:
        st.subheader("Bảo trì dữ liệu")
        
        c1, c2, c3, c4 = st.columns(4)
        
        if c1.button("🧹 Xoá metadata (reset)", type="secondary"):
            with st.spinner("Đang xóa..."):
                PARQUET_PATH.unlink(missing_ok=True)
                CSV_PATH.unlink(missing_ok=True)
                PARQUET_PATH.with_suffix(".parquet.bak").unlink(missing_ok=True)
                CSV_PATH.with_suffix(".csv.bak").unlink(missing_ok=True)
                md_load_cached.clear()
            st.success("Đã xoá metadata.")
        
        if c2.button("🧱 Rebuild thumbnails"):
            with st.spinner("Đang tạo lại thumbnails..."):
                df = md_load()
                ok = 0
                for i, r in df.iterrows():
                    p = DATA_ROOT / str(r["image_path"])
                    if not p.exists():
                        continue
                    t = thumb_path_for(str(r["image_path"]))
                    if make_thumb(p, t):
                        ok += 1
                    df.loc[i, "thumb_path"] = str(t.relative_to(DATA_ROOT)) if t.exists() else ""
                md_save_immediate(df)
            st.success(f"Đã tạo/cập nhật {ok} thumbnails.")
        
        if c3.button("🧰 Data Health Check"):
            with st.spinner("Đang kiểm tra..."):
                df = md_load()
                missing_files = df[~df["image_path"].apply(lambda p: (DATA_ROOT / str(p)).exists())]
                
                # Find orphan files
                all_files = []
                for root, _, files in os.walk(DATA_ROOT):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            try:
                                all_files.append(str(Path(root, f).relative_to(DATA_ROOT)))
                            except Exception:
                                pass
                
                orphan = sorted(set(all_files) - set(df["image_path"].astype(str)))
            
            st.warning(f"Thiếu file: {len(missing_files)} • Orphan: {len(orphan)}")
            
            if len(missing_files) > 0:
                st.write("Ảnh bị thiếu (có metadata, mất file):")
                st.dataframe(missing_files[["book_name", "image_path", "caption"]])
            
            if len(orphan) > 0:
                st.write("Ảnh mồ côi (có file, không metadata) — có thể copy vào export/manual:")
                st.code("\n".join(orphan[:200]))

        if os.name == "nt" and c4.button("📂 Mở thư mục dữ liệu (Windows)"):
            try:
                os.startfile(str(DATA_ROOT))  # type: ignore
            except Exception:
                st.warning("Không mở được thư mục.")

        # Enhanced session management section
        st.subheader("📊 Quản lý phiên làm việc")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Phiên hiện tại", st.session_state.work_session_manager.get_session_duration())
            st.metric("Ảnh đã edit (phiên)", st.session_state.productivity_stats["images_edited_session"])
            
            if st.button("💾 Lưu phiên làm việc"):
                st.session_state.work_session_manager.save_session_snapshot()
                save_session_state()
                st.success("✅ Đã lưu phiên làm việc!")
        
        with col2:
            st.metric("Tổng ảnh đã edit", len(st.session_state.edited_images_set))
            
            # Reset edit tracking
            if st.button("🔄 Reset tracking đã edit", type="secondary"):
                if st.button("Xác nhận reset", key="confirm_reset"):
                    st.session_state.edited_images_set = set()
                    st.session_state.last_visit_times = {}
                    st.session_state.edit_timestamps = {}
                    st.session_state.productivity_stats = {
                        "images_edited_today": 0,
                        "images_edited_session": 0,
                        "total_edits": 0,
                        "avg_time_per_edit": 0.0
                    }
                    save_session_state()
                    st.success("✅ Đã reset tracking!")
                    st.rerun()
        
        with col3:
            # Work session history
            if WORK_SESSION_JSON.exists():
                try:
                    sessions = json.loads(WORK_SESSION_JSON.read_text(encoding="utf-8"))
                    if sessions:
                        last_session = sessions[-1]
                        st.metric(
                            "Phiên trước", 
                            f"{int(last_session.get('duration_minutes', 0))} phút",
                            f"{last_session.get('images_processed', 0)} ảnh"
                        )
                        
                        if st.button("📈 Xem lịch sử phiên"):
                            st.subheader("Lịch sử các phiên làm việc")
                            df_sessions = pd.DataFrame(sessions[-10:])  # Last 10 sessions
                            df_sessions['duration_hours'] = df_sessions['duration_minutes'] / 60
                            df_sessions['timestamp'] = pd.to_datetime(df_sessions['timestamp'])
                            st.dataframe(df_sessions[['timestamp', 'duration_hours', 'images_processed']])
                except Exception:
                    pass

        st.subheader("Quản lý Sách (Ẩn/Hiện/Xoá)")
        
        df = md_load()
        books = sorted(df["book_name"].dropna().unique().tolist())
        
        if not books:
            st.info("Chưa có sách.")
        else:
            b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
            target_book = b1.selectbox("Chọn sách", books)
            
            if b2.button("🙈 Ẩn sách"):
                hb = set(st.session_state.hidden_books)
                hb.add(target_book)
                save_hidden_books_optimized(list(hb))
                st.success(f"Đã ẩn: {target_book}")
                time.sleep(0.2)
                st.rerun()
            
            if b3.button("👁️ Hiện tất cả"):
                save_hidden_books_optimized([])
                st.success("Đã hiện tất cả sách.")
                time.sleep(0.2)
                st.rerun()
            
            if b4.button("🗑️ Xoá sách", type="secondary", 
                        help="Xoá ảnh & metadata của sách này!"):
                with st.spinner(f"Đang xoá '{target_book}'..."):
                    book_dir = DATA_ROOT / safe_book_name(target_book)
                    if book_dir.exists():
                        shutil.rmtree(book_dir, ignore_errors=True)
                    d = md_load()
                    d = d[d["book_name"] != target_book]
                    md_save_immediate(d)
                st.success("Đã xoá sách.")
                time.sleep(0.2)
                st.rerun()

        st.subheader("Xuất ảnh theo Site/Flap")
        
        exp1, exp2 = st.columns([2, 1])
        out_base = exp1.text_input("Thư mục xuất", value=str(DATA_ROOT / "_export"))
        
        if exp2.button("📦 EXPORT"):
            with st.spinner("Đang xuất ảnh..."):
                out_root = Path(out_base)
                copied = 0
                df_export = md_load()
                
                # Enhanced export with edit status
                for _, r in df_export.iterrows():
                    site = (r.get("anatomical_site") or "unknown").strip() or "unknown"
                    flap = (r.get("flap_type") or "unknown").strip() or "unknown"
                    
                    # Add edit status to folder structure
                    edit_info = get_image_edit_info(r["image_path"])
                    status = "edited" if edit_info["is_edited"] else "unedited"
                    
                    dst_dir = out_root / site / flap / status
                    ensure_dir(dst_dir)
                    src = DATA_ROOT / str(r["image_path"])
                    
                    if src.exists():
                        try:
                            shutil.copy2(src, unique_filename(dst_dir / src.name))
                            copied += 1
                        except Exception:
                            continue
            
            st.success(f"✅ Đã sao chép {copied} ảnh → {out_root}")
            st.info("Cấu trúc: Site/Flap/EditStatus/")

        # Enhanced backup and restore
        st.subheader("Sao lưu & Khôi phục")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            if st.button("💾 Tạo bản sao lưu đầy đủ"):
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_dir = DATA_ROOT.parent / backup_name
                
                with st.spinner("Đang tạo bản sao lưu..."):
                    try:
                        shutil.copytree(DATA_ROOT, backup_dir)
                        st.success(f"✅ Đã tạo bản sao lưu: {backup_dir}")
                    except Exception as e:
                        st.error(f"Lỗi tạo bản sao lưu: {e}")
        
        with backup_col2:
            # Show available backups
            backup_dirs = [d for d in DATA_ROOT.parent.iterdir() 
                         if d.is_dir() and d.name.startswith("backup_")]
            
            if backup_dirs:
                selected_backup = st.selectbox("Chọn bản sao lưu", 
                                             [d.name for d in sorted(backup_dirs, reverse=True)])
                
                if st.button("🔄 Khôi phục từ bản sao lưu", type="secondary"):
                    if st.button("⚠️ Xác nhận khôi phục", key="confirm_restore"):
                        backup_path = DATA_ROOT.parent / selected_backup
                        with st.spinner("Đang khôi phục..."):
                            try:
                                if DATA_ROOT.exists():
                                    shutil.rmtree(DATA_ROOT)
                                shutil.copytree(backup_path, DATA_ROOT)
                                st.success("✅ Đã khôi phục thành công!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi khôi phục: {e}")


if __name__ == "__main__":
    main()