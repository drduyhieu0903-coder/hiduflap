
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
import re
from typing import Dict, Optional 
import gc
import base64
import functools
from threading import Lock
from collections import OrderedDict
import gc 

# ========================= CACHING & MEMOIZATION =========================
class LRUCache:
    """Simple thread-safe LRU cache"""
    def __init__(self, maxsize=1024):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = Lock()
    def get(self, key):
        with self.lock:
            if key not in self.cache: return None
            self.cache.move_to_end(key)
            return self.cache[key]
    def set(self, key, value):
        with self.lock:
            if key in self.cache: self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize: self.cache.popitem(last=False)

# Khởi tạo cache toàn cục cho thumbnails
_thumbnail_cache = LRUCache(maxsize=500)

# ========================= SINGLETON METADATA MANAGER =========================
@st.cache_resource
def get_metadata_manager():
    """Sử dụng @st.cache_resource để tạo một Singleton MetadataManager."""
    class MetadataManager:
        def __init__(self):
            self._df = None
            self._lock = Lock()
            self.last_mod_time = 0

        def _load_df(self):
            path = PARQUET_PATH if _parquet_available() and PARQUET_PATH.exists() else CSV_PATH
            if not path.exists():
                self._df = pd.DataFrame(columns=META_COLS)
                return
            try:
                mod_time = path.stat().st_mtime
                if self._df is None or mod_time > self.last_mod_time:
                    self._df = pd.read_parquet(path) if _parquet_available() and path.suffix == ".parquet" else pd.read_csv(path)
                    self.last_mod_time = mod_time
                    # Đảm bảo tất cả các cột đều tồn tại
                    for c in META_COLS:
                        if c not in self._df.columns:
                            self._df[c] = "" if c not in ["relevance_score", "page_num", "pos_y", "pos_x"] else 0.0
            except Exception as e:
                st.error(f"Lỗi tải metadata: {e}")
                self._df = pd.DataFrame(columns=META_COLS)

        def get(self, force_reload=False):
            with self._lock:
                if force_reload: self.last_mod_time = 0 # Force reload on next get()
                self._load_df()
                return self._df.copy() # Trả về một bản copy để tránh thay đổi ngầm

        def save(self, df_to_save):
            with self._lock:
                df_to_save = df_to_save[META_COLS].copy()
                use_parquet = _parquet_available()
                path = PARQUET_PATH if use_parquet else CSV_PATH
                try:
                    if use_parquet: df_to_save.to_parquet(path, index=False)
                    else: df_to_save.to_csv(path, index=False)
                    self.last_mod_time = path.stat().st_mtime
                    self._df = df_to_save # Cập nhật df trong bộ nhớ
                except Exception as e:
                    st.error(f"Lưu metadata thất bại: {e}")
    return MetadataManager()

# --- Thay thế các hàm md_load và md_save_immediate cũ ---
def md_load() -> pd.DataFrame:
    """Hàm md_load được tối ưu, gọi qua manager."""
    manager = get_metadata_manager()
    return manager.get()

def md_save_immediate(df: pd.DataFrame) -> None:
    """Hàm md_save được tối ưu, gọi qua manager."""
    manager = get_metadata_manager()
    manager.save(df)
    # Không cần clear cache của Streamlit nữa vì manager tự xử lý

# ==============================================================================
# ======================= END OF OPTIMIZATION PATCHES ========================
# ==============================================================================
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
CLINICAL_DIR      = DATA_ROOT / "_clinical"
SOURCE_PDF_DIR    = DATA_ROOT / "_source_pdfs" 
CLINICAL_METADATA_JSON = CLINICAL_DIR / ".clinical_patients.json"
ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
ensure_dir(SOURCE_PDF_DIR) 
ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)

APP_TITLE = "HIDU Facial Flap — Explorer ⚕️ "

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
        "comparison_list": [],
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

    # === LOGIC MỚI ĐỂ GIỚI HẠN LỊCH SỬ ===
    # Giữ lại tối đa 50 entry gần nhất cho mỗi ảnh
    if len(st.session_state.edit_timestamps[image_path]) > 50:
        st.session_state.edit_timestamps[image_path] = \
            st.session_state.edit_timestamps[image_path][-50:]
    # ====================================
    
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
@st.cache_data(show_spinner=False)

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
    """Hàm này giờ sẽ kiểm tra cache trước khi tính toán."""
    cached_path = _thumbnail_cache.get(rel_img)
    if cached_path:
        return Path(cached_path)
    
    base = re.sub(r"[\\/]+", "__", rel_img)
    path = DATA_ROOT / "_thumbs" / (Path(base).with_suffix(".jpg").name)
    _thumbnail_cache.set(rel_img, str(path))
    return path

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
    "source", "saved_at", "bytes_md5", "relevance_score", "notes", "source_document_path",
    "patient_id", "patient_name", "patient_age", "patient_gender",
    "diagnosis", "surgery_date", "surgery_type", "surgeon_name",
    "image_stage", "followup_date", "complications", "outcome_notes"
]

def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False

# ========================= BATCH OPERATIONS =========================
def reorganize_all_images_sequentially(progress_callback=None):
    """
    Sắp xếp lại toàn bộ ảnh trong thư viện theo thứ tự: Sách -> Trang -> Vị trí trên trang.
    Thực hiện đổi tên file và cập nhật metadata để phản ánh đúng thứ tự tuần tự.
    """
    df = md_load()
    if df.empty:
        return 0, 0

    st.info("Đang chuẩn bị sắp xếp. Vui lòng không đóng ứng dụng...")
    
    # Đảm bảo các cột sắp xếp là kiểu số để sort chính xác
    for col in ['page_num', 'pos_y', 'pos_x']:
        if col not in df.columns:
            df[col] = 0 # Thêm cột nếu chưa có
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sắp xếp DataFrame theo thứ tự tuyệt đối: Sách -> Trang -> Vị trí Y -> Vị trí X
    df_sorted = df.sort_values(by=['book_name', 'page_num', 'pos_y', 'pos_x'])
    
    processed_count = 0
    error_count = 0
    total_images = len(df_sorted)

    # Nhóm theo từng trang của từng sách
    for (book_name, page_num), group in df_sorted.groupby(['book_name', 'page_num']):
        # Sắp xếp lại một lần nữa trong nhóm để chắc chắn (dù df_sorted đã sort)
        group = group.sort_values(by=['pos_y', 'pos_x'])
        
        # Bắt đầu đổi tên file cho từng ảnh trong trang
        for new_idx, (old_index, row) in enumerate(group.iterrows(), start=1):
            try:
                # Cập nhật tiến trình
                if progress_callback:
                    percent = (processed_count + 1) / total_images
                    msg = f"Đang xử lý {processed_count+1}/{total_images}: {Path(row['image_path']).name}"
                    progress_callback(percent, msg)

                old_rel_path_str = row['image_path']
                old_path = DATA_ROOT / old_rel_path_str
                if not old_path.exists():
                    error_count += 1
                    continue

                # Tạo tên file mới theo đúng thứ tự: ..._img1, ..._img2, ...
                old_stem = old_path.stem
                # Tìm phần base của tên file (vd: 'BookName_p1_')
                base_name_match = re.match(r'(.*_p\d+_)', old_stem)
                if not base_name_match:
                    error_count += 1
                    continue # Bỏ qua nếu tên file không đúng định dạng
                
                base_name = base_name_match.group(1)
                new_stem = f"{base_name}img{new_idx}"
                new_path = old_path.with_stem(new_stem)
                
                # Chỉ đổi tên nếu tên mới khác tên cũ
                if old_path.resolve() != new_path.resolve():
                    # Đổi tên file ảnh chính
                    new_path = unique_filename(new_path) # Đảm bảo tên file là duy nhất
                    shutil.move(str(old_path), str(new_path))
                    
                    # Cập nhật đường dẫn mới trong DataFrame
                    new_rel_path_str = str(new_path.relative_to(DATA_ROOT))
                    df.loc[old_index, 'image_path'] = new_rel_path_str
                    
                    # Đổi tên/Tạo lại thumbnail
                    old_thumb_path_str = row.get('thumb_path')
                    if old_thumb_path_str:
                        old_thumb_path = DATA_ROOT / old_thumb_path_str
                        if old_thumb_path.exists():
                            old_thumb_path.unlink() # Xóa thumbnail cũ
                    
                    new_thumb_path = thumb_path_for(new_rel_path_str)
                    make_thumb(new_path, new_thumb_path)
                    df.loc[old_index, 'thumb_path'] = str(new_thumb_path.relative_to(DATA_ROOT))
                
                processed_count += 1
            
            except Exception as e:
                st.error(f"Lỗi khi xử lý {old_path.name}: {e}")
                error_count += 1
                continue
    
    # Lưu lại toàn bộ metadata đã được cập nhật
    st.info("Đang lưu lại cơ sở dữ liệu...")
    md_save_immediate(df)
    
    return processed_count, error_count

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

# ==============================================================================
# KHỐI CODE NÂNG CẤP HOÀN CHỈNH - SAO CHÉP VÀ THAY THẾ TOÀN BỘ 3 HÀM DƯỚI ĐÂY
# ==============================================================================

def normalize_text(s: str) -> str:
    """Chuẩn hóa text - tối ưu cho caption y khoa, đã tích hợp xử lý hyphenation."""
    if not s:
        return ""
    
    # CRITICAL: Xử lý hyphenation TRƯỚC khi xử lý xuống dòng và khoảng trắng
    s = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', s)
    
    # Sau đó mới chuẩn hóa các loại khoảng trắng và xuống dòng khác
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)
    
    return s.strip()

def _clean_extraction_artifacts(text: str) -> str:
    """
    Làm sạch các lỗi phổ biến khi trích xuất text từ PDF y khoa (phiên bản nâng cấp).
    """
    if not text:
        return ""
    
    # Xử lý dấu gạch nối ở cuối dòng (word hyphenation)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s*\n', r'\1', text)
    
    # Sửa các lỗi ngắt từ phổ biến trong y khoa
    # TỐI ƯU: Dùng \s* để bắt cả khoảng trắng và ký tự xuống dòng
    common_medical_fixes = {
        r'\bf\s*l\s*a\s*p\b': 'flap', # <-- Sửa lỗi "fl ap" triệt để hơn
        r'\bde\s*fect\b': 'defect',
        r'\bfi\s*g\b': 'fig',
        r'\bfi\s*gure\b': 'figure',
        r'\bdi\s*ff\s*icult\b': 'difficult',
        r'\bsu\s*pra\s*orbital\b': 'supraorbital',
        r'\bin\s*fra\s*orbital\b': 'infraorbital',
        r'\bre\s*con\s*struction\b': 'reconstruction',
        r'\bcar\s*ti\s*lage\b': 'cartilage',
        r'\bvas\s*cu\s*lar\b': 'vascular',
        r'\ban\s*a\s*tomical\b': 'anatomical',
        r'\bpro\s*ce\s*dure\b': 'procedure',
        r'\bin\s*ci\s*sion\b': 'incision',
    }
    for pattern, replacement in common_medical_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Sửa lỗi khoảng trắng trong số và đơn vị
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    text = re.sub(r'(Fig(?:ure)?)\s*\.\s*(\d)', r'\1. \2', text, flags=re.IGNORECASE)
    
    # Loại bỏ ký tự đặc biệt thừa
    text = text.replace('\u00ad', '').replace('\u200b', '').replace('\ufeff', '')
    
    # Sửa dấu câu bị tách
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    
    return text

def _normalize_fig_label(label: str) -> str:
    """Chuẩn hóa label của caption theo format "FIG. X.Y"."""
    match = re.match(r'(Fig(?:ure)?|Plate|Image|Diagram|Table)\s*\.?\s*([\d.A-Za-z\-]+)', 
                     label, re.IGNORECASE)
    if match:
        keyword, number = match.groups()
        keyword = 'FIG.' if keyword.lower().startswith('fig') else keyword.capitalize() + '.'
        number = number.replace(' ', '')
        return f"**{keyword} {number}**" # Thêm định dạng bold ở đây
    return f"**{label}**" # Fallback, vẫn in đậm

def _finalize_caption(caption: str) -> str:
    """Hoàn thiện caption: làm sạch cuối, cắt bớt nếu cần, đảm bảo kết thúc đúng."""
    caption = normalize_text(caption)
    caption = re.sub(r'\s+\d+\s*$', '', caption) # Loại bỏ số trang thừa ở cuối
    caption = re.sub(r'\s*\(cont(?:inued)?\.?\)\s*', '', caption, flags=re.IGNORECASE)
    
    if caption and not caption[-1] in '.!?':
        caption += '.'
    
    if len(caption) > 600:
        cut_point = caption[:600].rfind('.')
        caption = caption[:cut_point + 1] if cut_point > 300 else caption[:600] + "..."
        
    return ' '.join(caption.split())

def clean_raw_caption(raw_text: str) -> str:
    """
    Trích xuất và làm sạch caption từ text thô của PDF y khoa,
    sử dụng một chuỗi các hàm chuyên biệt.
    """
    if not raw_text:
        return ""

    # BƯỚC 1: LÀM SẠCH LỖI TRÍCH XUẤT CƠ BẢN
    cleaned_text = _clean_extraction_artifacts(raw_text)
    
    # BƯỚC 2: ĐỊNH NGHĨA PATTERNS VÀ TÌM TẤT CẢ ỨNG VIÊN
    FIG_LABEL = r"(?:Fig(?:ure)?|Plate|Image|Diagram|Table)"
    FIG_NUMBER = r"[\d]+(?:\.[\d]+)*[A-Za-z]?(?:-[\d]+)?"
    
    # SỬA LỖI: Pattern content cần linh hoạt hơn
    # ((?:[^.!?]+[.!?])+) -> (.+?)
    # Thêm lookahead để dừng lại một cách thông minh
    caption_pattern = re.compile(
        rf"({FIG_LABEL}\s*\.?\s*{FIG_NUMBER})\s*[.:–\-\s]*"
        r"(.+?)"
        fr"(?=\n\s*\n|{FIG_LABEL}\s*\.?\s*{FIG_NUMBER}|$)",
        re.DOTALL | re.IGNORECASE
    )
    matches = list(caption_pattern.finditer(cleaned_text))
    
    if not matches:
        first_sentence = re.search(r'^\s*([A-Z][^.!?]*[.!?])', cleaned_text)
        if first_sentence:
            return normalize_text(first_sentence.group(1))
        return normalize_text(cleaned_text[:250])

    # BƯỚC 3: CHỌN CAPTION TỐT NHẤT DỰA TRÊN ĐIỂM SỐ
    best_match = None
    best_score = -1
    
    for match in matches:
        caption_text = match.group(2).strip()
        score = len(caption_text)
        
        medical_keywords = ['nerve', 'artery', 'vein', 'muscle', 'flap', 'incision', 'suture', 'reconstruction']
        for keyword in medical_keywords:
            if keyword in caption_text.lower():
                score += 30
        
        if 50 <= len(caption_text) <= 400: score += 50
        if len(caption_text) < 20: score -= 100
        
        if score > best_score:
            best_score = score
            best_match = match
    
    if not best_match: best_match = matches[0]

    # BƯỚC 4: TRÍCH XUẤT, CHUẨN HÓA VÀ HOÀN THIỆN
    fig_label = best_match.group(1).strip()
    caption_text = best_match.group(2).strip()
    
    # Chuẩn hóa fig_label (vd: "Fig 2.1" -> "**FIG. 2.1**")
    formatted_label = _normalize_fig_label(fig_label)
    
    # Ghép và hoàn thiện caption cuối cùng
    final_caption = f"{formatted_label} {caption_text}"
    
    return _finalize_caption(final_caption)

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
    """
    Hàm trích xuất caption xung quanh ảnh được tối ưu cho tài liệu y khoa,
    với logic đa chiến lược, chấm điểm ưu tiên và xử lý thông minh.
    """
    text_all = normalize_text(page.get_text() or "")
    if not img_rect:
        return strip_chapter_tokens(text_all[:400])
    
    blocks = page.get_text("blocks")
    
    # === BƯỚC 1: PHÂN LOẠI KÍCH THƯỚC ẢNH ===
    page_height = page.rect.height
    is_large_image = img_rect.height > (page_height * 0.25)
    
    # === BƯỚC 2: XÁC ĐỊNH VÙNG TÌM KIẾM VÀ TÌM ỨNG VIÊN ===
    caption_candidates = []
    
    if is_large_image:
        # ẢNH LỚN/CỤM ẢNH: Caption thường ở ngay dưới
        search_height = page_height * 0.20
        search_rect = fitz.Rect(page.rect.x0, img_rect.y1, page.rect.x1, min(img_rect.y1 + search_height, page.rect.y1))
        
        for b in blocks:
            if len(b) < 5 or not isinstance(b[4], str): continue
            block_rect = fitz.Rect(b[:4])
            
            if block_rect.intersects(search_rect):
                caption_candidates.append({
                    'text': normalize_text(b[4]),
                    'distance': block_rect.y0 - img_rect.y1,
                    'y0': block_rect.y0,
                    'priority': 3 # Ưu tiên cao cho text ngay dưới ảnh lớn
                })
    else:
        # ẢNH NHỎ: Tìm kiếm linh hoạt hơn
        radius = max(20.0, page_height * float(nearby_ratio))
        vertical_tolerance = 10
        
        for b in blocks:
            if len(b) < 5 or not isinstance(b[4], str): continue
            block_rect = fitz.Rect(b[:4])
            
            distance = rect_min_distance(block_rect, img_rect)
            if distance > radius: continue
            
            # Bỏ qua text hoàn toàn phía trên
            if block_rect.y1 <= img_rect.y0 + vertical_tolerance: continue

            text = normalize_text(b[4])
            priority = 1 # Mặc định
            if block_rect.y0 >= img_rect.y1 - vertical_tolerance:
                priority = 3 # Ưu tiên cao nhất cho text dưới ảnh
            elif block_rect.y1 > img_rect.y0 and block_rect.y0 < img_rect.y1:
                priority = 2 # Ưu tiên trung bình cho text bên cạnh
            
            if re.search(r'(?i)(?:Fig(?:ure)?|Hình)\s*[\d.]+', text):
                priority += 5 # Bonus lớn cho block có pattern caption
            
            caption_candidates.append({
                'text': text, 'distance': distance, 'y0': block_rect.y0, 'priority': priority
            })

    # === BƯỚC 3: CHỌN LỌC KẾT QUẢ ===
    if not caption_candidates:
        return strip_chapter_tokens(text_all[:300])
    
    # Sắp xếp theo: priority (cao→thấp), distance (gần→xa), y0 (trên→dưới)
    caption_candidates.sort(key=lambda x: (-x['priority'], x['distance'], x['y0']))
    
    # Ưu tiên tìm caption có cấu trúc chuẩn trong top 5 ứng viên
    fig_pattern = re.compile(r'(?i)(?:Fig(?:ure)?|Hình)\s*[\d.]+[A-Za-z]?\s*[.:–-]?\s*.{15,}', re.DOTALL)
    for candidate in caption_candidates[:5]:
        match = fig_pattern.search(candidate['text'])
        if match:
            # Tìm thấy caption có cấu trúc, trả về ngay lập tức
            return strip_chapter_tokens(match.group(0)[:500])
    
    # Nếu không, ghép nối các ứng viên tốt nhất lại
    selected_texts = [c['text'] for c in caption_candidates[:3]] # Lấy top 3
    ctx = " ".join(selected_texts).strip()
    
    # Xử lý sau và cắt bớt nếu quá dài
    ctx = strip_chapter_tokens(ctx)
    if len(ctx) > 450:
        last_period = ctx[:450].rfind('.')
        ctx = ctx[:last_period + 1] if last_period > 200 else ctx[:450]
        
    return ctx

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
def _process_single_page(doc, pno, *, book, safe_b, out_book_dir, known, min_px, 
                         allow_duplicates, save_all_if_no_kw, nearby_ratio, relevance_threshold, 
                         original_pdf_path: Path): # <-- THÊM THAM SỐ NÀY
    """
    Hàm chuyên xử lý cho MỘT trang PDF để tối ưu bộ nhớ.
    Tất cả các biến tạo ra trong hàm này sẽ được giải phóng khi hàm kết thúc.
    """
    new_rows_for_page = []
    try:
        page = doc.load_page(pno)
        images = page.get_images(full=True)
        if not images:
            return []

        page_text = normalize_text(page.get_text() or "")
        
        for idx, im in enumerate(images, start=1):
            try:
                xref = im[0] if im else None
                if not isinstance(xref, int): continue
                
                base = doc.extract_image(xref)
                if not base or "image" not in base: continue
                
                img_bytes: bytes = base["image"]
                md5 = md5_bytes(img_bytes)
                
                if (not allow_duplicates) and md5 in known: continue

                try:
                    with Image.open(io.BytesIO(img_bytes)) as im_pil:
                        if max(im_pil.size) < int(min_px): continue
                except Exception:
                    pass

                rect = get_image_rect(page, xref)
                if not rect: continue

                raw_context = nearby_text_for_image(page, rect, nearby_ratio) or page_text
                final_caption = clean_raw_caption(raw_context)
                
                combined_text = f"{final_caption} {raw_context}"
                rel_score = calculate_relevance_score(combined_text)
                
                if (rel_score < relevance_threshold) and (not save_all_if_no_kw): continue

                m = re.search(r"\*\*(.*?)\*\*", final_caption or "")
                fig_local = m.group(1) if m else ""
                group_key = f"fig_{fig_local.replace(' ', '_')}" if fig_local else f"p{pno+1}"

                page_folder = out_book_dir / f"p{pno+1}"
                ensure_dir(page_folder)
                ext = (base.get("ext") or "png").lower()
                if ext not in ("png", "jpg", "jpeg"): ext = "png"
                
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
                    "caption": final_caption, "context": raw_context[:600],
                    "anatomical_site": site, "flap_type": flap, "confidence": conf,
                    "source": "pdf", "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "bytes_md5": md5, "relevance_score": int(rel_score), "notes": "",
                    "source_document_path": str(original_pdf_path) #
                }
                new_rows_for_page.append(row)
                known.add(md5)
            except Exception:
                continue
    finally:
        # Đảm bảo đối tượng page được giải phóng ngay cả khi có lỗi
        page = None
        
    return new_rows_for_page


def process_pdf(uploaded, *, min_px: int, allow_duplicates: bool, save_all_if_no_kw: bool,
                nearby_ratio: float, relevance_threshold: int, progress=None) -> Tuple[List[str], int, Path]:
    """
    Hàm xử lý PDF chính, đã được tối ưu hóa bộ nhớ và lưu file PDF gốc.
    """
    book = Path(uploaded.name).stem
    safe_b = safe_book_name(book)
    out_book_dir = DATA_ROOT / safe_b
    ensure_dir(out_book_dir)

    # === BƯỚC THAY ĐỔI QUAN TRỌNG: LƯU FILE PDF VÀO THƯ MỤC CỐ ĐỊNH ===
    # Tạo đường dẫn lưu file cố định, dùng unique_filename để tránh trùng lặp
    permanent_pdf_path = unique_filename(SOURCE_PDF_DIR / uploaded.name)
    # Ghi nội dung file được upload vào đường dẫn cố định này
    permanent_pdf_path.write_bytes(uploaded.getbuffer())
    # =================================================================

    df = md_load()
    known = set(df.get("bytes_md5", pd.Series(dtype=str)))
    saved_files: List[str] = []
    new_rows = []

    try:
        # Mở file PDF từ đường dẫn cố định, thay vì file tạm
        with fitz.open(permanent_pdf_path) as doc:
            total_pages = len(doc)
            for pno in range(total_pages):
                if progress:
                    progress((pno + 1) / total_pages, f"Đang xử lý trang {pno+1}/{total_pages} — {book}")
                
                # Gọi hàm chuyên xử lý cho từng trang, truyền đường dẫn cố định vào
                rows_from_page = _process_single_page(
                    doc, pno, book=book, safe_b=safe_b, out_book_dir=out_book_dir, known=known,
                    min_px=min_px, allow_duplicates=allow_duplicates, save_all_if_no_kw=save_all_if_no_kw,
                    nearby_ratio=nearby_ratio, relevance_threshold=relevance_threshold,
                    original_pdf_path=permanent_pdf_path.resolve() # <-- Dùng đường dẫn mới
                )
                
                if rows_from_page:
                    new_rows.extend(rows_from_page)
                
                if (pno + 1) % 20 == 0:
                    gc.collect()

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            saved_files = df_new["image_path"].apply(lambda p: str(DATA_ROOT / p)).tolist()
            df = pd.concat([df, df_new], ignore_index=True)
            md_save_immediate(df)
        
        return saved_files, len(new_rows), out_book_dir
        
    finally:
        # Bây giờ chúng ta không cần xóa file tạm nữa
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
def ingest_clinical_images(
    files: List,
    patient_info: Dict,
    case_name: str,
    image_stage: str,
    followup_date: Optional[str] = None,
    site: str = "(Giữ nguyên)",
    flap: str = "(Giữ nguyên)",
    allow_duplicates: bool = False
) -> int:
    """
    Upload ảnh lâm sàng với thông tin bệnh nhân đầy đủ
    """
    if not files:
        return 0
    
    # Validate thông tin bắt buộc
    required = ["patient_id", "patient_name", "diagnosis", "surgery_date"]
    for field in required:
        if not patient_info.get(field):
            st.error(f"Thiếu thông tin bắt buộc: {field}")
            return 0
    
    # Lưu thông tin bệnh nhân
    add_or_update_patient(patient_info["patient_id"], patient_info)
    
    # Tạo thư mục
    case = safe_book_name(case_name or f"Patient_{patient_info['patient_id']}")
    out_dir = CLINICAL_DIR / case / image_stage
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
            
            # Xử lý ảnh
            with Image.open(io.BytesIO(b)) as im:
                w, h = im.size
                fmt = (im.format or "JPEG").lower()
                dt_str = exif_datetime_str(im)
            
            ext = ".png" if fmt == "png" else ".jpg"
            fname = f"{patient_info['patient_id']}_{image_stage}_{int(time.time()*1000)}{ext}"
            fname = unique_filename(out_dir / fname)
            fname.write_bytes(b)
            
            rel = str(fname.relative_to(DATA_ROOT))
            tpath = thumb_path_for(rel)
            make_thumb(fname, tpath, max_side=512)
            
            # Tạo caption tự động
            stage_display = image_stage.replace('_', ' ').title()
            caption = f"{patient_info['patient_name']} - {stage_display}"
            if followup_date:
                caption += f" ({followup_date})"
            
            # Context
            ctx = f"Chẩn đoán: {patient_info['diagnosis']} | Phẫu thuật: {patient_info['surgery_date']}"
            if patient_info.get('surgery_type'):
                ctx += f" | Loại: {patient_info['surgery_type']}"
            
            combo_text = f"{caption} {ctx}"
            site_g, flap_g, conf = guess_labels(combo_text)
            
            # Xử lý site và flap
            site_final = site if site != "(Giữ nguyên)" else site_g
            flap_final = flap if flap != "(Giữ nguyên)" else flap_g
            
            rel_score = calculate_relevance_score(combo_text)
            
            row = {
                "book_name": f"Clinical::{case}",
                "image_path": rel,
                "thumb_path": str(tpath.relative_to(DATA_ROOT)) if tpath and tpath.exists() else "",
                "page_num": 1,
                "fig_num": "",
                "group_key": f"{patient_info['patient_id']}_{image_stage}",
                "caption": caption,
                "context": ctx,
                "anatomical_site": site_final,
                "flap_type": flap_final,
                "confidence": conf,
                "source": "clinical",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "bytes_md5": md5,
                "relevance_score": int(rel_score),
                "notes": "",
                "source_document_path": "",
                # Clinical fields
                "patient_id": patient_info["patient_id"],
                "patient_name": patient_info["patient_name"],
                "patient_age": str(patient_info.get("patient_age", "")),
                "patient_gender": patient_info.get("patient_gender", ""),
                "diagnosis": patient_info["diagnosis"],
                "surgery_date": patient_info["surgery_date"],
                "surgery_type": patient_info.get("surgery_type", ""),
                "surgeon_name": patient_info.get("surgeon_name", ""),
                "image_stage": image_stage,
                "followup_date": followup_date or "",
                "complications": patient_info.get("complications", ""),
                "outcome_notes": patient_info.get("outcome_notes", "")
            }
            new_rows.append(row)
            known.add(md5)
            added += 1
            
        except Exception as e:
            st.warning(f"Lỗi xử lý {f.name}: {e}")
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
# =============================================================================
# ================= CRITICAL PERFORMANCE FIXES - KHỐI CODE MỚI ================
# =============================================================================

def render_image_card_lightweight(row: pd.Series, df: pd.DataFrame, col_key: str):
    """
    Card ảnh siêu nhẹ với checkbox không gây rerun.
    """
    rel = row["image_path"]
    safe_key = rel.replace('/', '_').replace('\\', '_').replace('.', '_').replace('-', '_')
    
    with st.container(border=True):
        tp = row.get("thumb_path", "")
        show_img_path = thumb_path_for(rel) if tp else DATA_ROOT / rel
        if show_img_path.exists():
            st.image(str(show_img_path), use_container_width=True)
        
        site = row.get("anatomical_site", "Chưa rõ")
        flap = row.get("flap_type", "Chưa rõ")
        info_text = f"""
            <div style="font-size: 0.85rem; line-height: 1.4;">
                📕 {row['book_name'][:25]}.. trang {int(row['page_num'])} <br>
                📍 **Vị trí:** {site.capitalize()} <br>
                🔧 **Loại vạt:** {flap.capitalize()}
            </div>
            """
        st.markdown(info_text, unsafe_allow_html=True)
        st.caption(f"_{row.get('caption', '')[:80]}_")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            is_selected = rel in st.session_state.selected_list
            # CRITICAL: Sử dụng on_change callback KHÔNG có rerun
            st.checkbox(
                "Chọn", 
                value=is_selected, 
                key=f"cb_{col_key}_{safe_key}", 
                on_change=toggle_image_selection, 
                args=(rel,),
                # Không thêm label_visibility để tránh conflict
            )
        with col2:
            if st.button("Sửa ✏️", key=f"qe_{col_key}_{safe_key}", help="Chỉnh sửa nhanh", use_container_width=True):
                st.session_state.quick_edit_image = rel
                st.rerun()  # Chỉ rerun khi cần thiết
        with col3:
            with st.popover("Xóa 🗑️", use_container_width=True):
                st.markdown("**⚠️ Bạn có chắc muốn xóa vĩnh viễn ảnh này không?**")
                st.caption("Thao tác này không thể hoàn tác.")
                if st.button("🔴 Vâng, xóa ngay", key=f"del_confirm_{safe_key}", type="primary"):
                    try:
                        (DATA_ROOT / rel).unlink(missing_ok=True)
                        tpr = row.get("thumb_path", "")
                        if tpr and (DATA_ROOT / tpr).exists():
                            (DATA_ROOT / tpr).unlink(missing_ok=True)
                        
                        df_all = md_load()
                        df_cleaned = df_all[df_all["image_path"] != rel]
                        md_save_immediate(df_cleaned)

                        if rel in st.session_state.selected_list:
                            st.session_state.selected_list.remove(rel)
                        
                        st.toast(f"Đã xóa ảnh: {Path(rel).name}", icon="🗑️")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi xóa: {e}")
                    
        with col4:
            if st.button("Xem 👁️", key=f"qv_{col_key}_{safe_key}", help="Xem ảnh lớn", use_container_width=True):
                st.session_state.lightbox_open = True
                st.session_state.lightbox_seq = [rel]
                st.session_state.lightbox_idx = 0
                st.rerun()

def render_selection_strip_optimized(sel_list: List[str], current_page_images: List[str]):
    """Thanh lựa chọn được tối ưu với logic chọn trang thông minh."""
    if not sel_list:
        return

    with st.container(border=True):
        st.markdown(f"**🎯 Đã chọn: {len(sel_list)} ảnh**")
        
        # Kiểm tra trạng thái trang
        all_on_page_selected = all(img in sel_list for img in current_page_images)
        some_on_page_selected = any(img in sel_list for img in current_page_images)
        
        col1, col2, col3, col4 = st.columns(4)
        
        if col1.button("Bỏ chọn tất cả", key="clear_all_sel"):
            st.session_state.selected_list = []
            save_session_state()
            st.rerun()
        
        # Logic thông minh cho nút chọn trang
        if all_on_page_selected:
            btn_label = "✖️ Bỏ chọn trang này"
            btn_type = "secondary"
        elif some_on_page_selected:
            btn_label = "➕ Chọn thêm còn lại"
            btn_type = "primary"
        else:
            btn_label = "✅ Chọn cả trang"
            btn_type = "primary"
            
        if col2.button(btn_label, key="toggle_page", type=btn_type):
            if all_on_page_selected:
                # Bỏ chọn toàn bộ trang
                st.session_state.selected_list = [x for x in sel_list if x not in current_page_images]
            else:
                # Thêm các ảnh chưa chọn vào danh sách (GIỮ các ảnh đã chọn)
                new_selections = [img for img in current_page_images if img not in sel_list]
                st.session_state.selected_list = sorted(list(set(sel_list + new_selections)))
            
            save_session_state()
            st.rerun()
            
        if col3.button("⚡ Đánh dấu đã edit", key="mark_batch"):
            for rel in sel_list:
                mark_image_edited(rel, "bulk_mark")
            save_session_state()
            st.success(f"Đã đánh dấu {len(sel_list)} ảnh!")
        
        if col4.button("🔬 So sánh", key="compare_batch"):
            if 2 <= len(sel_list) <= 4:
                st.session_state.comparison_list = sel_list
                st.rerun()
            else:
                st.warning("Chọn từ 2 đến 4 ảnh để so sánh.")
        
        # Hiển thị trạng thái trang hiện tại
        st.caption(f"Trang này: {sum(1 for img in current_page_images if img in sel_list)}/{len(current_page_images)} ảnh đã chọn")
def load_clinical_patients() -> Dict:
    """Load danh sách bệnh nhân từ file JSON"""
    if not CLINICAL_METADATA_JSON.exists():
        return {}
    try:
        return json.loads(CLINICAL_METADATA_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_clinical_patients(patients: Dict):
    """Lưu thông tin bệnh nhân vào file JSON"""
    try:
        ensure_dir(CLINICAL_METADATA_JSON.parent)
        CLINICAL_METADATA_JSON.write_text(
            json.dumps(patients, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        st.error(f"Lỗi lưu thông tin bệnh nhân: {e}")

def add_or_update_patient(patient_id: str, patient_info: Dict):
    """Thêm hoặc cập nhật thông tin bệnh nhân"""
    patients = load_clinical_patients()
    patients[patient_id] = {
        **patient_info,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_clinical_patients(patients)

def get_patient_images_count(patient_id: str) -> int:
    """Đếm số ảnh của bệnh nhân"""
    df = md_load()
    if "patient_id" not in df.columns:
        return 0
    return len(df[df["patient_id"] == patient_id])
def render_quick_edit_dialog():
    """
    Dialog chỉnh sửa nhanh, đã được nâng cấp để có thể xóa ảnh trực tiếp.
    """
    if "quick_edit_image" not in st.session_state or not st.session_state.quick_edit_image:
        return
    
    rel = st.session_state.quick_edit_image
    df = md_load()
    row = df[df["image_path"] == rel]
    if row.empty:
        del st.session_state.quick_edit_image
        return
    r = row.iloc[0]

    @st.dialog("Chỉnh sửa nhanh", width="large")
    def show_edit():
        st.image(str(DATA_ROOT / rel), use_container_width=True)
        
        current_caption = r.get('caption', '')
        current_notes = r.get('notes', '')
        as_idx = ANATOMY_OPTIONS.index(r.get("anatomical_site", "unknown")) if r.get("anatomical_site") in ANATOMY_OPTIONS else 0
        ft_idx = FLAP_OPTIONS.index(r.get("flap_type", "unknown")) if r.get("flap_type") in FLAP_OPTIONS else 0
        
        new_caption = st.text_area("Caption", value=current_caption)
        new_notes = st.text_area("Notes", value=current_notes, placeholder="Thêm ghi chú cá nhân...")
        
        c1, c2 = st.columns(2)
        with c1:
            new_site = st.selectbox("Vị trí", ANATOMY_OPTIONS, index=as_idx, key=f"qe_site_{rel}")
        with c2:
            new_flap = st.selectbox("Loại vạt", FLAP_OPTIONS, index=ft_idx, key=f"qe_flap_{rel}")
        
        st.markdown("---")
        st.divider() # Thêm một đường kẻ để tách biệt

        source_path_str = r.get("source_document_path")
        if source_path_str:
            # Sử dụng toggle để hiển thị/ẩn PDF viewer
            show_pdf_viewer = st.toggle("📖 Xem trang tài liệu gốc", key=f"toggle_pdf_{rel}")
            
            if show_pdf_viewer:
                source_path = Path(source_path_str)
                page_number = int(r.get("page_num", 1))

                if not source_path.exists():
                    st.warning(f"Không tìm thấy file gốc tại: {source_path}")
                else:
                    try:
                        # Thử mở PDF và kiểm tra mã hóa
                        doc = fitz.open(source_path)
                        
                        # Kiểm tra PDF có bị mã hóa không
                        if doc.is_encrypted:
                            doc.close()
                            st.error("⚠️ File PDF được bảo vệ bằng mật khẩu. Không thể xem trước trong ứng dụng.")
                            st.info(f"📂 Đường dẫn file: `{source_path}`")
                            
                            # Nút mở file trực tiếp (Windows)
                            if os.name == "nt":
                                if st.button("🔓 Mở PDF bằng ứng dụng mặc định", key=f"open_ext_{rel}"):
                                    try:
                                        os.startfile(str(source_path))
                                        st.success("✅ Đã mở file! Vui lòng nhập mật khẩu trong ứng dụng PDF.")
                                    except Exception:
                                        st.warning("Không thể mở file tự động.")
                        else:
                            # PDF không bị mã hóa - hiển thị trực tiếp
                            if page_number > len(doc):
                                st.warning(f"Trang {page_number} vượt quá số trang ({len(doc)} trang)")
                                doc.close()
                            else:
                                # Trích xuất trang
                                new_doc = fitz.open()
                                new_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                                pdf_bytes = new_doc.tobytes()
                                new_doc.close()
                                doc.close()

                                # Hiển thị PDF inline (không dùng nested dialog)
                                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                                
                                st.markdown(f"**📄 Trang {page_number}** từ `{source_path.name}`")
                                iframe_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" style="border: 1px solid #ddd; border-radius: 4px;"></iframe>'
                                st.markdown(iframe_html, unsafe_allow_html=True)
                    
                    except RuntimeError as e:
                        # Lỗi mã hóa/mật khẩu
                        error_msg = str(e).lower()
                        if "encrypted" in error_msg or "password" in error_msg:
                            st.error("⚠️ File PDF được bảo vệ bằng mật khẩu.")
                            st.info(f"Vui lòng mở file: `{source_path}`")
                            
                            if os.name == "nt" and st.button("📂 Mở PDF", key=f"open_enc_{rel}"):
                                try:
                                    os.startfile(str(source_path))
                                except Exception:
                                    pass
                        else:
                            st.error(f"Lỗi khi đọc PDF: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Không thể mở trang tài liệu: {str(e)}")
                        st.caption(f"File: `{source_path}`")
        # === NÚT LƯU VÀ HỦY ===
        btn_save, btn_cancel = st.columns(2)
        if btn_save.button("💾 Lưu thay đổi", type="primary", key=f"qe_save_{rel}", use_container_width=True):
            df.loc[df["image_path"] == rel, "caption"] = new_caption
            df.loc[df["image_path"] == rel, "notes"] = new_notes
            df.loc[df["image_path"] == rel, "anatomical_site"] = new_site
            df.loc[df["image_path"] == rel, "flap_type"] = new_flap
            md_save_immediate(df)
            mark_image_edited(rel, "quick_edit")
            st.session_state.quick_edit_image = None
            st.rerun()
            
        if btn_cancel.button("✖ Hủy", key=f"qe_cancel_{rel}", use_container_width=True):
            st.session_state.quick_edit_image = None
            st.rerun()

        # ============================================
        # === THÊM NÚT XÓA VÀO VÙNG RIÊNG BIỆT ===
        # ============================================
        st.divider()
        st.error("🔴 Vùng nguy hiểm")
        if st.button(f"🗑️ Xóa vĩnh viễn ảnh này", key=f"qe_delete_{rel}", use_container_width=True):
            # Xóa file ảnh và thumbnail
            (DATA_ROOT / rel).unlink(missing_ok=True)
            tpr = r.get("thumb_path", "")
            if tpr:
                (DATA_ROOT / tpr).unlink(missing_ok=True)
            
            # Xóa metadata
            md_delete_by_paths_batch([rel])

            # Dọn dẹp session state
            if rel in st.session_state.selected_list:
                st.session_state.selected_list.remove(rel)
            if rel in st.session_state.edited_images_set:
                st.session_state.edited_images_set.remove(rel)
            
            st.toast(f"✅ Đã xóa ảnh: {Path(rel).name}", icon="🗑️")
            st.session_state.quick_edit_image = None # Đóng dialog
            st.rerun()

    show_edit()

def render_library_gallery_optimized(view: pd.DataFrame):
    """
    Grid ảnh được tối ưu với pagination, card siêu nhẹ và đã tích hợp lại
    chức năng "Thao tác hàng loạt".
    """
    page_size = 12
    total_items = len(view)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    st.session_state.current_page = min(st.session_state.current_page, total_pages)

    # Giao diện điều khiển trang
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    with c1:
        if st.button("◀ Trang trước", disabled=st.session_state.current_page <= 1):
            st.session_state.current_page -= 1
            st.rerun()
    with c3:
        if st.button("Trang sau ▶", disabled=st.session_state.current_page >= total_pages, use_container_width=True):
            st.session_state.current_page += 1
            st.rerun()
    with c2:
        st.markdown(f"<div style='text-align: center; margin-top: 8px;'>Trang {st.session_state.current_page} / {total_pages}</div>", unsafe_allow_html=True)
    
    start_idx = (st.session_state.current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)
    
    page_view = view.iloc[start_idx:end_idx]
    current_page_images = page_view["image_path"].tolist()
    
    sel_list = st.session_state.selected_list
       
    # =======================================================================
    # === KHỐI THAO TÁC HÀNG LOẠT ĐÃ ĐƯỢC THÊM LẠI VÀO ĐÂY ===
    # =======================================================================
    with st.expander("🛠️ Thao tác hàng loạt", expanded=bool(sel_list)):
        with st.form("batch_operations_form"):
            b1, b2, b3, b4 = st.columns([1.5, 1.2, 1.2, 1.2])
            
            with b1:
                cap_mode = st.selectbox("Caption", ["(Giữ nguyên)", "Thay thế toàn bộ", "Thêm tiền tố", "Thêm hậu tố"])
                cap_text = st.text_input("Nội dung caption", value="")
            with b2:
                site_pick_b = st.selectbox("Vị trí", ["(Giữ nguyên)"] + ANATOMY_OPTIONS, key="batch_site")
            with b3:
                flap_pick_b = st.selectbox("Loại vạt", ["(Giữ nguyên)"] + FLAP_OPTIONS, key="batch_flap")
            with b4:
                group_pick_b = st.text_input("Group (optional)", key="batch_group")
            
            apply_batch = st.form_submit_button("✅ ÁP DỤNG HÀNG LOẠT", type="primary", use_container_width=True, disabled=not sel_list)

        if apply_batch:
            unedited_to_process = [p for p in sel_list if p not in st.session_state.edited_images_set]
            edited_skipped_count = len(sel_list) - len(unedited_to_process)

            if not unedited_to_process:
                st.warning("Tất cả ảnh đã chọn đều đã được edit trước đó. Không có gì thay đổi.")
            else:
                with st.spinner(f"Đang áp dụng cho {len(unedited_to_process)} ảnh chưa edit..."):
                    df_all = md_load()
                    mask = df_all["image_path"].isin(unedited_to_process)
                    
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
                    
                    for rel in unedited_to_process:
                        mark_image_edited(rel, "batch_operation")
                    
                    md_save_immediate(df_all)
                
                st.success(f"✅ Xong! Đã áp dụng cho {len(unedited_to_process)} ảnh.")
                if edited_skipped_count > 0:
                    st.info(f"ℹ️ Đã bỏ qua {edited_skipped_count} ảnh vì chúng đã được edit trước đó.")
                st.divider()
        st.error("🔴 Vùng nguy hiểm: Xóa vĩnh viễn")
        
        with st.form("batch_delete_form"):
            delete_batch = st.form_submit_button(
                f"🗑️ Xóa vĩnh viễn {len(sel_list)} ảnh đã chọn", 
                type="primary", 
                use_container_width=True, 
                disabled=not sel_list
            )
        
        if delete_batch:
            with st.spinner(f"Đang xóa {len(sel_list)} ảnh..."):
                # 1. Xóa các mục trong metadata
                md_delete_by_paths_batch(sel_list)
                
                # 2. Xóa các file ảnh và thumbnail vật lý
                for rel in sel_list:
                    try:
                        (DATA_ROOT / rel).unlink(missing_ok=True)
                        thumb = thumb_path_for(rel)
                        if thumb.exists():
                            thumb.unlink()
                    except Exception as e:
                        st.warning(f"Lỗi khi xóa file {rel}: {e}")
            
            st.toast(f"✅ Đã xóa vĩnh viễn {len(sel_list)} ảnh.", icon="🗑️")
            st.session_state.selected_list = []
            st.rerun()
    
    # =======================================================================

     
    render_selection_strip_optimized(st.session_state.selected_list, current_page_images)
    
    df_full = md_load() # Load 1 lần để truyền vào card
    cols = st.columns(3)
    for i, (_, row) in enumerate(page_view.iterrows()):
        with cols[i % 3]:
            # Sử dụng hàm render card siêu nhẹ mới
            render_image_card_lightweight(row, df_full, f"p{st.session_state.current_page}_{i}")
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
def toggle_image_selection(image_path: str):
    """
    Hàm callback được gọi mỗi khi checkbox chọn ảnh thay đổi.
    KHÔNG gây rerun để tránh mất focus.
    """
    if image_path in st.session_state.selected_list:
        st.session_state.selected_list.remove(image_path)
    else:
        st.session_state.selected_list.append(image_path)
    
    # Lưu state ngay lập tức nhưng KHÔNG rerun
    save_session_state()

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

def render_image_card_enhanced(row: pd.Series, df: pd.DataFrame, sel_list: List[str], seq: List[str], 
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
            # Xác định xem ảnh hiện tại có đang được chọn hay không
            is_selected = rel in st.session_state.selected_list
            safe_rel = rel.replace('/', '_').replace('\\', '_').replace('.', '_')
            # Hiển thị checkbox và gắn callback on_change
            st.checkbox(
            "Chọn",
            value=is_selected,
            key=f"pick_{col_key}_{safe_rel}",
            )
    
    with c2:
        with st.popover("📝 Note", use_container_width=True):
            # Load existing notes for this image
            existing_notes = row.get("notes", "")
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

def render_comparison_dialog():
    """
    Render một dialog để so sánh ảnh, đã được thêm các bước kiểm tra an toàn
    để chống lỗi KeyError.
    """
    if not st.session_state.get("comparison_list"):
        return

    images_to_compare = st.session_state.comparison_list

    @st.dialog(f"So sánh {len(images_to_compare)} ảnh", width="large")
    def show_comparison_ui():
        df = md_load()
        
        # =================== LOGIC CHỐNG LỖI QUAN TRỌNG ===================
        # 1. Lọc ra những đường dẫn thực sự tồn tại trong metadata
        existing_paths = [p for p in images_to_compare if p in df["image_path"].values]
        
        # 2. Xử lý nếu không có ảnh nào hợp lệ
        if not existing_paths:
            st.warning("Không tìm thấy ảnh được chọn trong metadata. Có thể chúng đã bị xóa hoặc đổi tên. Vui lòng chọn lại.")
            if st.button("Đóng"):
                st.session_state.comparison_list = []
                st.rerun()
            return # Dừng hàm nếu không có gì để hiển thị
        
        # 3. Thông báo nếu có một số ảnh bị thiếu
        if len(existing_paths) != len(images_to_compare):
            st.info(f"Chỉ có thể hiển thị {len(existing_paths)}/{len(images_to_compare)} ảnh được chọn vì một số đã bị thay đổi.")
        
        # 4. Chỉ sử dụng danh sách các đường dẫn đã được xác thực
        image_data = df[df["image_path"].isin(existing_paths)].set_index("image_path").loc[existing_paths].reset_index()
        # ====================================================================

        cols = st.columns(len(image_data))

        for i, (_, row) in enumerate(image_data.iterrows()):
            with cols[i]:
                img_path = DATA_ROOT / row["image_path"]
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                    
                    st.markdown(f"**Caption:** {row.get('caption', 'N/A')}")
                    st.markdown(f"---")
                    st.markdown(f"**Sách:** `{row.get('book_name')}`")
                    st.markdown(f"**Vị trí:** {row.get('anatomical_site')}")
                    st.markdown(f"**Loại vạt:** {row.get('flap_type')}")
                    st.metric("Điểm liên quan", int(row.get('relevance_score', 0)))

        st.markdown("---")
        if st.button("✖ Đóng cửa sổ", use_container_width=True):
            st.session_state.comparison_list = []
            st.rerun()

    # Gọi hàm nội bộ để hiển thị dialog
    show_comparison_ui()

    # Sử dụng st.dialog để tạo cửa sổ modal
    @st.dialog(f"So sánh {len(images_to_compare)} ảnh", width="large")
    def show_comparison_ui():
        df = md_load()
        # Lấy dữ liệu metadata cho các ảnh được chọn
        image_data = df[df["image_path"].isin(images_to_compare)].set_index("image_path").loc[images_to_compare].reset_index()

        # Tạo số cột bằng số ảnh
        cols = st.columns(len(image_data))

        for i, (_, row) in enumerate(image_data.iterrows()):
            with cols[i]:
                img_path = DATA_ROOT / row["image_path"]
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                    
                    st.markdown(f"**Caption:** {row.get('caption', 'N/A')}")
                    st.markdown(f"---")
                    st.markdown(f"**Sách:** `{row.get('book_name')}`")
                    st.markdown(f"**Vị trí:** {row.get('anatomical_site')}")
                    st.markdown(f"**Loại vạt:** {row.get('flap_type')}")
                    st.markdown(f"**Tags:** `{row.get('tags', 'chưa có')}`")
                    st.metric("Điểm liên quan", int(row.get('relevance_score', 0)))

        st.markdown("---")
        if st.button("✖ Đóng cửa sổ", use_container_width=True):
            st.session_state.comparison_list = []
            st.rerun()

    # Gọi hàm nội bộ để hiển thị dialog
    show_comparison_ui()

def render_lightbox_enhanced():
    """Enhanced lightbox with source document viewing."""
    if not st.session_state.get("lightbox_open"):
        return
    
    seq = st.session_state.get("lightbox_seq", [])
    if not seq:
        st.session_state.lightbox_open = False
        return
    
    idx = st.session_state.get("lightbox_idx", 0)
    idx = max(0, min(idx, len(seq) - 1))
    
    if idx >= len(seq):
        st.session_state.lightbox_open = False
        return
        
    rel = seq[idx]
    p = DATA_ROOT / rel
    
    if not p.exists():
        st.session_state.lightbox_open = False
        st.toast(f"Lỗi: Không tìm thấy ảnh: {rel}", icon="❌")
        return
    
    @st.dialog("Xem ảnh chi tiết", width="large")
    def show_lightbox():
        # Lấy thông tin ảnh
        df = md_load()
        row = df[df["image_path"] == rel]
        r = row.iloc[0] if not row.empty else {}
        
        # Bố cục 2 cột: Ảnh bên trái, thông tin bên phải
        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.image(str(p), use_container_width=True)
        
        with col_info:
            edit_info = get_image_edit_info(rel)
            cap = r.get("caption", "Chưa có caption")
            edit_status = "✅ Đã edit" if edit_info["is_edited"] else "🔄 Chưa edit"
            
            st.markdown(f"**{cap}**")
            st.caption(f"{r.get('book_name', '')} • trang {r.get('page_num', '')} • {edit_status}")
            
            # === PHẦN XEM TÀI LIỆU GỐC ĐÃ ĐƯỢC CHUYỂN VÀO ĐÂY ===
            source_path_str = r.get("source_document_path")
            if source_path_str:
                show_pdf_viewer = st.toggle("📖 Xem trang tài liệu gốc", key=f"lb_toggle_pdf_{rel}")
                
                if show_pdf_viewer:
                    source_path = Path(source_path_str)
                    page_number = int(r.get("page_num", 1))

                    if not source_path.exists():
                        st.warning(f"Không tìm thấy file gốc tại: {source_path}")
                    else:
                        try:
                            doc = fitz.open(source_path)
                            if doc.is_encrypted:
                                doc.close()
                                st.error("⚠️ File PDF được bảo vệ bằng mật khẩu.")
                            else:
                                new_doc = fitz.open()
                                new_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                                pdf_bytes = new_doc.tobytes()
                                new_doc.close()
                                doc.close()

                                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                                st.markdown(f"**📄 Trang {page_number}**")
                                iframe_html = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400px" style="border: 1px solid #ddd;"></iframe>'
                                st.markdown(iframe_html, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Lỗi khi đọc PDF: File có thể bị hỏng.")

        # === Nút điều khiển ===
        st.divider()
        c1, c2, c3, c4 = st.columns([1.2, 2.5, 2, 1.2])
        
        if c1.button("⟵ Trước", key="lb_prev", disabled=(idx <= 0)):
            st.session_state.lightbox_idx -= 1
            st.rerun()
        
        if c2.button("✖ Đóng", use_container_width=True, key="lb_close"):
            st.session_state.lightbox_open = False
            st.rerun()
        
        if c3.button("Sửa ảnh này ✏️", use_container_width=True, key="lb_edit"):
            st.session_state.quick_edit_image = rel
            st.session_state.lightbox_open = False # Đóng lightbox để mở dialog edit
            st.rerun()
        
        if c4.button("Sau ⟶", key="lb_next", disabled=(idx >= len(seq) - 1)):
            st.session_state.lightbox_idx += 1
            st.rerun()
    
    show_lightbox()
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
        st.subheader("Trích xuất ảnh từ nhiều PDF")
    
        # THAY ĐỔI: Cho phép upload nhiều file
        uploaded_files = st.file_uploader(
            "Chọn PDF (≤ 200MB mỗi file, có thể chọn nhiều file)", 
            type=["pdf"],
            accept_multiple_files=True  # Cho phép chọn nhiều file
        )
    
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            min_px = st.number_input("Lọc ảnh nhỏ hơn (px)", 100, 3000, MIN_IMG_SIZE_DEFAULT, step=20, key="multi_pdf_min_px")
        with c2:
            allow_dup = st.checkbox("Cho phép trùng ảnh (MD5)", value=False, key="multi_pdf_allow_dup")
        with c3:
            save_all = st.checkbox("Lưu tất cả (bỏ điểm)", value=SAVE_ALL_FALLBACK, key="multi_pdf_save_all")
        with c4:
            nearby_ratio = st.slider("Bán kính caption", 0.05, 0.35, NEARBY_RATIO_DEFAULT, 0.01, key="multi_pdf_ratio")
        with c5:
            min_score_pdf = st.slider("Ngưỡng điểm", 0, 60, 12, key="multi_pdf_min_score")
    
        if uploaded_files:
            st.info(f"📚 Đã chọn {len(uploaded_files)} file PDF để xử lý.")
            with st.expander("Xem danh sách file đã chọn"):
                for i, f in enumerate(uploaded_files, 1):
                    size_mb = f.size / (1024 * 1024)
                    st.write(f"{i}. **{f.name}** ({size_mb:.1f} MB)")
    
            if st.button("🚀 BẮT ĐẦU XỬ LÝ HÀNG LOẠT", type="primary", use_container_width=True, disabled=not uploaded_files):
                oversized = [f for f in uploaded_files if f.size > 200 * 1024 * 1024]
                if oversized:
                    st.error(f"❌ Các file vượt 200MB: {', '.join([f.name for f in oversized])}")
                else:
                    total_files = len(uploaded_files)
                    overall_progress = st.progress(0.0, text="Bắt đầu xử lý hàng loạt...")
                    
                    results_summary = []
                    total_images_saved = 0
                    total_processing_time = 0
    
                    for file_idx, uploaded_file in enumerate(uploaded_files, 1):
                        st.markdown(f"--- \n ### 📄 File {file_idx}/{total_files}: **{uploaded_file.name}**")
                        file_progress = st.progress(0.0)
                        file_status = st.empty()
                        
                        start_time = time.time()
    
                        def _update_progress(p, msg):
                            file_progress.progress(min(max(float(p), 0.0), 1.0), text=msg)
    
                        try:
                            paths, n, book_dir = process_pdf(
                                uploaded_file, 
                                min_px=min_px, 
                                allow_duplicates=allow_dup, 
                                save_all_if_no_kw=save_all, 
                                nearby_ratio=nearby_ratio, 
                                relevance_threshold=int(min_score_pdf), 
                                progress=_update_progress
                            )
                            
                            processing_time = time.time() - start_time
                            total_processing_time += processing_time
                            file_progress.progress(1.0)
    
                            if n > 0:
                                file_status.success(f"✅ Hoàn tất: Lưu {n} ảnh trong {processing_time:.1f}s")
                                total_images_saved += n
                                results_summary.append({"file": uploaded_file.name, "images": n, "time": processing_time, "status": "success"})
                            else:
                                file_status.warning("⚠️ Không tìm thấy ảnh phù hợp.")
                                results_summary.append({"file": uploaded_file.name, "images": 0, "time": processing_time, "status": "no_images"})
    
                        except Exception as e:
                            file_status.error(f"❌ Lỗi nghiêm trọng: {str(e)}")
                            results_summary.append({"file": uploaded_file.name, "images": 0, "time": 0, "status": "error", "error": str(e)})
                        
                        overall_progress.progress((file_idx) / total_files, text=f"Đã xử lý {file_idx}/{total_files} file...")
    
                    overall_progress.progress(1.0, text="🎉 HOÀN THÀNH XỬ LÝ TẤT CẢ FILE!")
    
                    st.markdown("---")
                    st.subheader("📊 Báo cáo Tổng hợp")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Tổng file đã xử lý", total_files)
                    col2.metric("Tổng ảnh đã lưu", total_images_saved)
                    col3.metric("Tổng thời gian", f"{total_processing_time:.1f}s")
    
                    if results_summary:
                        st.markdown("##### Chi tiết từng file")
                        df_results = pd.DataFrame(results_summary)
                        st.dataframe(df_results, use_container_width=True)

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
    # ==================== THAY THẾ TAB CLINICAL BẰNG CODE NÀY ====================

    # ==================== THAY THẾ TAB CLINICAL ====================

    with tab_clinical:
        st.subheader("Upload ảnh lâm sàng với thông tin bệnh nhân")
    
    # CSS cho stage selector có thể cuộn
    st.markdown("""
    <style>
    .stage-selector {
        display: flex;
        overflow-x: auto;
        gap: 10px;
        padding: 10px 0;
        white-space: nowrap;
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }
    .stage-selector::-webkit-scrollbar {
        height: 8px;
    }
    .stage-selector::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .stage-selector::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    .stage-btn {
        padding: 8px 16px;
        border: 2px solid #ddd;
        border-radius: 8px;
        cursor: pointer;
        background: white;
        transition: all 0.3s;
        text-align: center;
        min-width: 120px;
    }
    .stage-btn:hover {
        border-color: #4CAF50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stage-btn.active {
        background: #4CAF50;
        color: white;
        border-color: #4CAF50;
    }
    .stage-icon {
        font-size: 1.5em;
        display: block;
    }
    .stage-label {
        font-size: 0.85em;
        display: block;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Labels và icons cho giai đoạn
    stage_config = {
        "preop": {"label": "Trước mổ", "icon": "📸"},
        "intraop": {"label": "Trong mổ", "icon": "⚕️"},
        "postop_immediate": {"label": "Ngay sau mổ", "icon": "🏥"},
        "postop_1week": {"label": "1 tuần sau", "icon": "📅"},
        "followup_1month": {"label": "Khám 1 tháng", "icon": "🔍"},
        "followup_3months": {"label": "Khám 3 tháng", "icon": "📊"},
        "followup_6months": {"label": "Khám 6 tháng", "icon": "✅"},
        "followup_1year": {"label": "Khám 1 năm", "icon": "🎯"},
        "followup_other": {"label": "Theo dõi khác", "icon": "📝"}
    }
    
    tab_upload, tab_manage = st.tabs(["📤 Upload ảnh", "👥 Quản lý bệnh nhân"])
    
    # ========== TAB UPLOAD ==========
    with tab_upload:
        patients = load_clinical_patients()
        patient_list = ["➕ Thêm bệnh nhân mới"] + [
            f"{pid} - {info.get('patient_name', 'N/A')}" 
            for pid, info in sorted(patients.items())
        ]
        
        selected = st.selectbox("Chọn bệnh nhân", patient_list)
        
        # Form thêm bệnh nhân mới
        if selected.startswith("➕"):
            with st.expander("📋 Nhập thông tin bệnh nhân mới", expanded=True):
                with st.form("new_patient"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        patient_id = st.text_input("*Mã BN", placeholder="BN001")
                        patient_name = st.text_input("*Họ tên", placeholder="Nguyễn Văn A")
                    with col2:
                        patient_age = st.number_input("Tuổi", 0, 120, 30)
                        patient_gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"])
                    with col3:
                        surgery_date = st.date_input("*Ngày PT", datetime.now())
                        surgeon_name = st.text_input("Bác sĩ", placeholder="BS...")
                    
                    diagnosis = st.text_area("*Chẩn đoán", placeholder="Mô tả bệnh lý...")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        surgery_type = st.text_input("Phương pháp PT", placeholder="VD: Nasolabial flap")
                        complications = st.text_area("Biến chứng", placeholder="Nếu có...")
                    with col2:
                        outcome_notes = st.text_area("Ghi chú kết quả", placeholder="Đánh giá...")
                    
                    if st.form_submit_button("💾 Lưu bệnh nhân", type="primary"):
                        if not all([patient_id, patient_name, diagnosis, surgery_date]):
                            st.error("Thiếu thông tin bắt buộc (*)")
                        else:
                            patient_info = {
                                "patient_id": patient_id, "patient_name": patient_name,
                                "patient_age": patient_age, "patient_gender": patient_gender,
                                "diagnosis": diagnosis, "surgery_date": surgery_date.strftime("%Y-%m-%d"),
                                "surgery_type": surgery_type, "surgeon_name": surgeon_name,
                                "complications": complications, "outcome_notes": outcome_notes
                            }
                            add_or_update_patient(patient_id, patient_info)
                            st.success(f"Đã lưu {patient_name}")
                            time.sleep(1)
                            st.rerun()
            
            st.info("Vui lòng lưu thông tin bệnh nhân trước khi upload ảnh")
            can_upload = False
        
        else:
            patient_id = selected.split(" - ")[0]
            patient_data = patients.get(patient_id, {})
            
            # Thông tin bệnh nhân thu gọn
            with st.expander("📋 Thông tin bệnh nhân", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mã BN", patient_id)
                col2.metric("Họ tên", patient_data.get('patient_name', 'N/A'))
                col3.metric("Tuổi/Giới", f"{patient_data.get('patient_age', 'N/A')}/{patient_data.get('patient_gender', 'N/A')}")
                col4.metric("Số ảnh", get_patient_images_count(patient_id))
                
                st.markdown(f"**Chẩn đoán:** {patient_data.get('diagnosis', 'N/A')}")
                st.markdown(f"**Ngày PT:** {patient_data.get('surgery_date', 'N/A')} | **BS:** {patient_data.get('surgeon_name', 'N/A')}")
            
            can_upload = True
        
        # ========== PHẦN UPLOAD VỚI STAGE SELECTOR CÓ THỂ CUỘN ==========
        if can_upload:
            st.markdown("---")
            st.markdown("### 📸 Upload ảnh theo giai đoạn")
            
            # File uploader
            files = st.file_uploader(
                "Chọn ảnh (có thể chọn nhiều file)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True
            )
            
            if files:
                st.success(f"✅ Đã chọn {len(files)} ảnh")
                st.markdown("---")
                
                # Preview ảnh dạng grid
                st.markdown("#### 🖼️ Xem trước ảnh đã chọn")
                num_cols = 4
                cols = st.columns(num_cols)
                for idx, file in enumerate(files):
                    with cols[idx % num_cols]:
                        st.image(file, use_container_width=True)
                        st.caption(f"{idx+1}. {file.name[:15]}")
                
                st.markdown("---")
                
                # STAGE SELECTOR CÓ THỂ CUỘN - DÙNG RADIO
                st.markdown("#### 📅 Chọn giai đoạn (cuộn ngang để xem tất cả)")
                
                # Tạo session state để lưu stage đã chọn
                if "selected_stage" not in st.session_state:
                    st.session_state.selected_stage = "preop"
                
                # Tạo columns cho các stage buttons
                stage_cols = st.columns(len(stage_config))
                
                for idx, (stage_key, stage_info) in enumerate(stage_config.items()):
                    with stage_cols[idx]:
                        # Tạo button với styling
                        button_class = "active" if st.session_state.selected_stage == stage_key else ""
                        
                        if st.button(
                            f"{stage_info['icon']}\n{stage_info['label']}",
                            key=f"stage_{stage_key}",
                            use_container_width=True,
                            type="primary" if st.session_state.selected_stage == stage_key else "secondary"
                        ):
                            st.session_state.selected_stage = stage_key
                            st.rerun()
                
                # Hiển thị stage đã chọn
                selected_stage_info = stage_config[st.session_state.selected_stage]
                st.info(f"Giai đoạn đã chọn: {selected_stage_info['icon']} **{selected_stage_info['label']}**")
                
                # Nếu là followup thì nhập ngày
                followup_date = None
                if "followup" in st.session_state.selected_stage:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        followup_date = st.date_input(
                            "📅 Ngày khám lại",
                            datetime.now(),
                            key="followup_date_input"
                        ).strftime("%Y-%m-%d")
                    with col2:
                        st.metric("Ngày đã chọn", followup_date)
                
                st.markdown("---")
                
                # Tùy chọn nâng cao
                with st.expander("⚙️ Tùy chọn nâng cao"):
                    col1, col2 = st.columns(2)
                    with col1:
                        case_name = st.text_input("Tên case", value=f"Patient_{patient_id}")
                        site_pick = st.selectbox("Vị trí giải phẫu", ["(Tự động)"] + ANATOMY_OPTIONS)
                    with col2:
                        flap_pick = st.selectbox("Loại vạt", ["(Tự động)"] + FLAP_OPTIONS)
                        allow_dup = st.checkbox("Cho phép ảnh trùng lặp", value=False)
                
                st.markdown("---")
                
                # Nút upload lớn
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(
                        f"⬆️ UPLOAD {len(files)} ẢNH VÀO '{selected_stage_info['label'].upper()}'",
                        type="primary",
                        use_container_width=True
                    ):
                        patient_info = patients.get(patient_id, {})
                        
                        with st.spinner(f"Đang upload {len(files)} ảnh..."):
                            added = ingest_clinical_images(
                                files=files,
                                patient_info=patient_info,
                                case_name=case_name,
                                image_stage=st.session_state.selected_stage,
                                followup_date=followup_date,
                                site=site_pick if site_pick != "(Tự động)" else "(Giữ nguyên)",
                                flap=flap_pick if flap_pick != "(Tự động)" else "(Giữ nguyên)",
                                allow_duplicates=allow_dup
                            )
                        
                        if added > 0:
                            st.success(f"✅ Đã thêm {added}/{len(files)} ảnh vào '{selected_stage_info['label']}'")
                            st.balloons()
                            # Reset stage selector
                            st.session_state.selected_stage = "preop"
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.warning("⚠️ Không thêm được ảnh (có thể do trùng lặp)")
            
            else:
                st.info("👆 Chọn ảnh để bắt đầu upload")
    
    # ========== TAB QUẢN LÝ ==========
    with tab_manage:
        patients = load_clinical_patients()
        
        if not patients:
            st.info("Chưa có bệnh nhân. Hãy thêm bệnh nhân mới trong tab Upload.")
        else:
            subtab_list, subtab_detail, subtab_edit = st.tabs([
                "📋 Danh sách", "🔍 Chi tiết", "✏️ Chỉnh sửa & Xóa"
            ])
            
            # --- DANH SÁCH ---
            with subtab_list:
                st.markdown("### 👥 Danh sách bệnh nhân")
                
                patient_data = []
                df_all = md_load()
                
                for pid, info in patients.items():
                    img_count = get_patient_images_count(pid)
                    
                    if "patient_id" in df_all.columns:
                        patient_imgs = df_all[df_all["patient_id"] == pid]
                        stages = len(patient_imgs["image_stage"].unique()) if not patient_imgs.empty else 0
                    else:
                        stages = 0
                    
                    patient_data.append({
                        "Mã": pid,
                        "Họ tên": info.get('patient_name', 'N/A'),
                        "Tuổi": info.get('patient_age', 'N/A'),
                        "Giới": info.get('patient_gender', 'N/A'),
                        "Chẩn đoán": (info.get('diagnosis', 'N/A')[:35] + "...") 
                                     if len(info.get('diagnosis', '')) > 35 else info.get('diagnosis', 'N/A'),
                        "Ngày PT": info.get('surgery_date', 'N/A'),
                        "Ảnh": img_count,
                        "Giai đoạn": stages,
                        "BS": info.get('surgeon_name', 'N/A')[:15]
                    })
                
                df_patients = pd.DataFrame(patient_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    search = st.text_input("🔍 Tìm kiếm", placeholder="Tên hoặc mã BN...")
                with col2:
                    sort = st.selectbox("Sắp xếp", 
                                       ["Ngày PT (mới)", "Ngày PT (cũ)", "Tên A-Z", "Số ảnh (nhiều)"])
                
                if search:
                    mask = (df_patients["Mã"].str.contains(search, case=False, na=False) |
                           df_patients["Họ tên"].str.contains(search, case=False, na=False))
                    df_patients = df_patients[mask]
                
                if sort == "Ngày PT (mới)":
                    df_patients = df_patients.sort_values("Ngày PT", ascending=False)
                elif sort == "Ngày PT (cũ)":
                    df_patients = df_patients.sort_values("Ngày PT")
                elif sort == "Tên A-Z":
                    df_patients = df_patients.sort_values("Họ tên")
                else:
                    df_patients = df_patients.sort_values("Ảnh", ascending=False)
                
                st.dataframe(df_patients, use_container_width=True, hide_index=True)
                st.caption(f"Hiển thị: {len(df_patients)} bệnh nhân")
            
            # --- CHI TIẾT ---
            with subtab_detail:
                st.markdown("### 🔍 Xem chi tiết bệnh nhân")
                
                selected_view = st.selectbox(
                    "Chọn bệnh nhân",
                    [f"{p['Mã']} - {p['Họ tên']}" for p in patient_data]
                )
                
                patient_id = selected_view.split(" - ")[0]
                patient_info = patients.get(patient_id, {})
                
                st.markdown(f"## {patient_info.get('patient_name', 'N/A')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mã BN", patient_id)
                col2.metric("Tuổi", patient_info.get('patient_age', 'N/A'))
                col3.metric("Giới tính", patient_info.get('patient_gender', 'N/A'))
                col4.metric("Tổng ảnh", get_patient_images_count(patient_id))
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Chẩn đoán**")
                    st.info(patient_info.get('diagnosis', 'N/A'))
                    st.markdown(f"**📅 Ngày PT:** {patient_info.get('surgery_date', 'N/A')}")
                    if patient_info.get('surgery_type'):
                        st.markdown(f"**⚕️ Phương pháp:** {patient_info['surgery_type']}")
                
                with col2:
                    if patient_info.get('surgeon_name'):
                        st.markdown(f"**👨‍⚕️ Bác sĩ:** {patient_info['surgeon_name']}")
                    if patient_info.get('complications'):
                        st.markdown("**⚠️ Biến chứng**")
                        st.warning(patient_info['complications'])
                    if patient_info.get('outcome_notes'):
                        st.markdown("**📝 Ghi chú**")
                        st.success(patient_info['outcome_notes'])
                
                if "patient_id" in df_all.columns:
                    patient_images = df_all[df_all["patient_id"] == patient_id]
                    
                    if not patient_images.empty:
                        st.markdown("---")
                        st.markdown("### 📸 Thư viện ảnh")
                        
                        stage_counts = patient_images["image_stage"].value_counts()
                        cols = st.columns(len(stage_counts))
                        for idx, (stage, count) in enumerate(stage_counts.items()):
                            cols[idx].metric(stage_config[stage]["label"], f"{count} ảnh")
                        
                        st.markdown("---")
                        
                        for stage in sorted(patient_images["image_stage"].unique()):
                            stage_imgs = patient_images[patient_images["image_stage"] == stage]
                            
                            with st.expander(
                                f"{stage_config[stage]['icon']} {stage_config[stage]['label']} - {len(stage_imgs)} ảnh",
                                expanded=(stage == "preop")
                            ):
                                cols = st.columns(4)
                                for idx, (_, row) in enumerate(stage_imgs.iterrows()):
                                    with cols[idx % 4]:
                                        img_path = DATA_ROOT / row["image_path"]
                                        if img_path.exists():
                                            st.image(str(img_path), use_container_width=True)
                                            date = row.get('followup_date') or row.get('saved_at', '')
                                            st.caption(f"📅 {date[:10]}")
                    else:
                        st.info("Bệnh nhân chưa có ảnh")
            
            # --- CHỈNH SỬA & XÓA ---
            with subtab_edit:
                st.markdown("### ✏️ Chỉnh sửa thông tin & Xóa case")
                
                selected_edit = st.selectbox(
                    "Chọn bệnh nhân",
                    [f"{p['Mã']} - {p['Họ tên']}" for p in patient_data],
                    key="edit_select"
                )
                
                patient_id = selected_edit.split(" - ")[0]
                patient_info = patients.get(patient_id, {})
                
                # Form chỉnh sửa
                st.markdown(f"#### ✏️ Chỉnh sửa: {patient_info.get('patient_name', 'N/A')}")
                
                with st.form("edit_patient"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        new_name = st.text_input("Họ tên", value=patient_info.get('patient_name', ''))
                        new_age = st.number_input("Tuổi", 0, 120, 
                                                  value=int(patient_info.get('patient_age', 0) or 0))
                    with col2:
                        new_gender = st.selectbox("Giới tính", ["Nam", "Nữ", "Khác"],
                                                 index=["Nam", "Nữ", "Khác"].index(
                                                     patient_info.get('patient_gender', 'Nam')))
                        
                        surgery_date_str = patient_info.get('surgery_date', '')
                        try:
                            surgery_date = datetime.strptime(surgery_date_str, "%Y-%m-%d").date()
                        except:
                            surgery_date = datetime.now().date()
                        new_surgery_date = st.date_input("Ngày PT", value=surgery_date)
                    
                    with col3:
                        new_surgeon = st.text_input("Bác sĩ", value=patient_info.get('surgeon_name', ''))
                        new_surgery_type = st.text_input("Phương pháp", 
                                                        value=patient_info.get('surgery_type', ''))
                    
                    new_diagnosis = st.text_area("Chẩn đoán", value=patient_info.get('diagnosis', ''))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        new_complications = st.text_area("Biến chứng", 
                                                        value=patient_info.get('complications', ''))
                    with col2:
                        new_outcome = st.text_area("Ghi chú", value=patient_info.get('outcome_notes', ''))
                    
                    if st.form_submit_button("💾 CẬP NHẬT THÔNG TIN", type="primary"):
                        updated_info = {
                            "patient_id": patient_id, "patient_name": new_name,
                            "patient_age": new_age, "patient_gender": new_gender,
                            "diagnosis": new_diagnosis, 
                            "surgery_date": new_surgery_date.strftime("%Y-%m-%d"),
                            "surgery_type": new_surgery_type, "surgeon_name": new_surgeon,
                            "complications": new_complications, "outcome_notes": new_outcome
                        }
                        add_or_update_patient(patient_id, updated_info)
                        
                        df_all = md_load()
                        if "patient_id" in df_all.columns:
                            mask = df_all["patient_id"] == patient_id
                            df_all.loc[mask, "patient_name"] = new_name
                            df_all.loc[mask, "patient_age"] = str(new_age)
                            df_all.loc[mask, "patient_gender"] = new_gender
                            df_all.loc[mask, "diagnosis"] = new_diagnosis
                            df_all.loc[mask, "surgery_date"] = new_surgery_date.strftime("%Y-%m-%d")
                            df_all.loc[mask, "surgery_type"] = new_surgery_type
                            df_all.loc[mask, "surgeon_name"] = new_surgeon
                            df_all.loc[mask, "complications"] = new_complications
                            df_all.loc[mask, "outcome_notes"] = new_outcome
                            md_save_immediate(df_all)
                        
                        st.success("✅ Đã cập nhật thông tin!")
                        time.sleep(1)
                        st.rerun()
                
                # Phần xóa case
                st.markdown("---")
                st.error("### 🗑️ XÓA CASE LÂM SÀNG")
                st.warning(f"""
                **⚠️ CẢNH BÁO: Thao tác này sẽ:**
                - Xóa vĩnh viễn tất cả thông tin bệnh nhân **{patient_info.get('patient_name', 'N/A')}** (Mã: {patient_id})
                - Xóa **{get_patient_images_count(patient_id)} ảnh** và metadata liên quan
                - Xóa tất cả dữ liệu trong thư mục case
                - **KHÔNG THỂ HOÀN TÁC**
                """)
                
                with st.expander("🔓 Mở khóa chức năng xóa"):
                    st.markdown("Để xác nhận xóa, vui lòng:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        confirm_text = st.text_input(
                            f"Nhập mã bệnh nhân '{patient_id}' để xác nhận:",
                            key="delete_confirm_text"
                        )
                    
                    with col2:
                        confirm_check = st.checkbox(
                            "Tôi hiểu rằng thao tác này không thể hoàn tác",
                            key="delete_confirm_check"
                        )
                    
                    delete_enabled = (confirm_text == patient_id) and confirm_check
                    
                    if st.button(
                        f"🗑️ XÓA VĨNH VIỄN CASE {patient_id}",
                        type="secondary",
                        disabled=not delete_enabled,
                        use_container_width=True
                    ):
                        with st.spinner(f"Đang xóa case {patient_id}..."):
                            # Xóa tất cả ảnh
                            df_all = md_load()
                            if "patient_id" in df_all.columns:
                                patient_images = df_all[df_all["patient_id"] == patient_id]
                                
                                for _, row in patient_images.iterrows():
                                    try:
                                        img_path = DATA_ROOT / row["image_path"]
                                        img_path.unlink(missing_ok=True)
                                        
                                        thumb = row.get("thumb_path")
                                        if thumb:
                                            (DATA_ROOT / thumb).unlink(missing_ok=True)
                                    except Exception as e:
                                        st.warning(f"Lỗi xóa file: {e}")
                                
                                # Xóa metadata
                                df_all = df_all[df_all["patient_id"] != patient_id]
                                md_save_immediate(df_all)
                            
                            # Xóa thư mục case
                            case_dir = CLINICAL_DIR / safe_book_name(f"Patient_{patient_id}")
                            if case_dir.exists():
                                shutil.rmtree(case_dir, ignore_errors=True)
                            
                            # Xóa thông tin bệnh nhân
                            patients_dict = load_clinical_patients()
                            if patient_id in patients_dict:
                                del patients_dict[patient_id]
                                save_clinical_patients(patients_dict)
                            
                            st.success(f"✅ Đã xóa hoàn toàn case {patient_id}")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()

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
                # Dòng quan trọng: Định nghĩa 'search_term' ngay sau khi kiểm tra fkw
                # Biến này sẽ chứa từ khóa đã được xử lý (viết thường, bỏ khoảng trắng)
                search_term = fkw.lower().strip()
                
                # Mặc định, mẫu tìm kiếm (search_pattern) chính là từ khóa gốc đã xử lý
                search_pattern = search_term

                # === ÁP DỤNG LOGIC DỊCH VÀ TÌM KIẾM SONG NGỮ ===
                if use_translation:
                    # Gọi hàm dịch
                    english_kw = translate_vietnamese_to_english(search_term)
                    
                    # Nếu dịch thành công và kết quả khác với từ khóa gốc
                    if english_kw and english_kw.lower() != search_term:
                        st.sidebar.info(f"Đang tìm kiếm cho: **'{search_term}'** OR **'{english_kw}'**")
                        # Cập nhật mẫu tìm kiếm để bao gồm cả hai từ (dùng regex OR `|`)
                        search_pattern = f"{search_term}|{english_kw.lower()}"

                # Logic tìm kiếm cuối cùng
                bag = (view["caption"].fillna("") + " " + view["context"].fillna("")).str.lower()
                               
                # Sử dụng regex=True để biểu thức `|` hoạt động
                view = view[bag.str.contains(search_pattern, na=False, regex=True)]
                # =================================================

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

            render_library_gallery_optimized(view)
            render_quick_edit_dialog()
            # =======================================================================
            
            render_lightbox_enhanced()
            render_comparison_dialog()
            
        
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
            st.success("Đã xoá metadata.")
        
        if c2.button("🧱 Rebuild thumbnails"):
            df = md_load()
            if df.empty:
                st.info("Không có ảnh nào để tạo thumbnail.")
            else:
                st.info("Bắt đầu quá trình tạo lại thumbnails...")
                total_images = len(df)
                progress_bar = st.progress(0.0, f"Chuẩn bị xử lý {total_images} ảnh...")
                ok = 0

                # Dùng df.iterrows() an toàn hơn khi ghi lại DataFrame
                for index, row in df.iterrows():
                    p = DATA_ROOT / str(row["image_path"])
                    if not p.exists():
                        continue
                    
                    t = thumb_path_for(str(row["image_path"]))
                    if make_thumb(p, t):
                        ok += 1
                    
                    # Ghi lại đường dẫn thumbnail mới vào DataFrame
                    df.loc[index, "thumb_path"] = str(t.relative_to(DATA_ROOT)) if t.exists() else ""
                    
                    # Cập nhật thanh tiến trình mỗi 5 ảnh để tối ưu hiệu năng
                    if (index + 1) % 5 == 0:
                        progress_percent = (index + 1) / total_images
                        progress_bar.progress(progress_percent, f"Đang xử lý {index + 1}/{total_images} ảnh...")
                
                # Lưu lại DataFrame với các đường dẫn thumbnail đã được cập nhật
                md_save_immediate(df)
                
                progress_bar.progress(1.0, f"Hoàn tất! Đã xử lý {ok} thumbnails.")
                st.success(f"Đã tạo/cập nhật {ok} thumbnails.")
        if c3.button("🔄 Sắp xếp lại toàn bộ", help="Đổi tên tất cả file ảnh theo thứ tự tuần tự trong sách (Sách -> Trang -> Vị trí)"):
            if st.button("⚠️ Xác nhận Sắp xếp lại", key="confirm_reorganize"):
                progress_bar = st.progress(0.0, "Bắt đầu...")
                status_text = st.empty()
                # Vì tên file đã thay đổi, các lựa chọn cũ không còn hợp lệ
                st.session_state.selected_list = []
                def _progress_callback(percent, msg):
                    progress_bar.progress(percent, msg)
                    status_text.text(msg)

                processed, errors = reorganize_all_images_sequentially(progress_callback=_progress_callback)
                
                progress_bar.progress(1.0, "Hoàn tất!")
                st.success(f"✅ Đã sắp xếp và đổi tên thành công {processed} ảnh.")
                if errors > 0:
                    st.warning(f"⚠️ Có {errors} lỗi xảy ra trong quá trình xử lý.")
                
                # Tự động làm mới để thấy kết quả
                time.sleep(2)
                st.rerun()
        if c4.button("🧰 Data Health Check"):
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
