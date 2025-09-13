import os
import sys
from pathlib import Path
import streamlit as st
import requests
import json
from typing import List, Optional
import pandas as pd
import csv
from io import StringIO

# ====== Ensure we can import from project root (for AppSettings) ======
ROOT_DIR = Path(__file__).resolve().parents[1]  # project root (contains "app/")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.core.settings import AppSettings

    APP_SETTINGS = AppSettings()
    MAP_DIR_DEFAULT = APP_SETTINGS.MAP_KEYFRAME_DIR  # <-- dùng mặc định từ settings
except Exception:
    # fallback nếu không import được (chạy ngoài repo)
    MAP_DIR_DEFAULT = "data/map-keyframes"

# ============== Page configuration ==============
st.set_page_config(
    page_title="Keyframe Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============== Custom CSS ==============
st.markdown(
    """
<style>
    .main > div { padding-top: 2rem; }

    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    /* Small download button + toolbar (right aligned) */
    .top-toolbar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: .5rem;
        margin-top: -1rem;
        margin-bottom: .5rem;
    }
    /* Shrink download button */
    div[data-testid="stDownloadButton"] button {
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
        border-radius: 8px;
        background: #4c72ff;
    }

    .toolbar-input input {
        height: 2rem !important;
        padding: 0 .5rem !important;
        font-size: 0.9rem !important;
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============== Header ==============
st.markdown(
    """
<div class="search-container">
    <h1 style="margin: 0; font-size: 2.5rem;">🔍 Keyframe Search</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Search through video keyframes using semantic similarity
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============== Session State ==============
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"
if "export_fname" not in st.session_state:
    st.session_state.export_fname = None
if "trake_events" not in st.session_state:
    st.session_state.trake_events = ["", "", "", "", ""]
if "trake_params" not in st.session_state:
    st.session_state.trake_params = {
        "beam_width": 50,
        "top_k_per_stage": 50,
        "score_threshold": 0.1,
        "max_kf_gap": 200,
    }

# ============== API Configuration ==============
with st.expander("⚙️ API Configuration", expanded=False):
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_base_url,
        help="Base URL for the keyframe search API",
    )
    if api_url != st.session_state.api_base_url:
        st.session_state.api_base_url = api_url


# ============== Main search interface ==============
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 🎛️ Search Mode")
    search_mode = st.selectbox(
        "Mode",
        options=[
            "Default",
            "Exclude Groups",
            "Include Groups & Videos",
            "TRAKE (1-5 events)",
        ],
        help="Choose how to filter your search results",
        key="mode_select",
    )

with col1:
    # ========== Default/Exclude/Include: show query inputs; hide TRAKE ==========
    if search_mode in ["Default", "Exclude Groups", "Include Groups & Videos"]:
        query = st.text_input(
            "🔍 Search Query",
            placeholder="Enter your search query (e.g., 'person walking in the park')",
            help="Enter 1-1000 characters describing what you're looking for",
            key="default_query",
        )
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            top_k = st.slider(
                "📊 Max Results",
                min_value=1,
                max_value=200,
                value=100,
                key="default_topk",
            )
        with col_param2:
            score_threshold = st.slider(
                "🎯 Min Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="default_score_th",
            )

    # ========== TRAKE: show 5 events (5 rows) + TRAKE params; hide default query ==========
    elif search_mode == "TRAKE (1-5 events)":
        st.markdown("### 🧭 TRAKE – Enter 1 to 5 events")
        # 5 rows, same width as default query box
        for i in range(5):
            st.session_state.trake_events[i] = st.text_input(
                f"Event {i+1}",
                value=st.session_state.trake_events[i],
                placeholder=f"Describe event #{i+1} (optional)",
                key=f"trake_event_row_{i}",
            )

        col_tr1, col_tr2, col_tr3, col_tr4 = st.columns(4)
        with col_tr1:
            st.session_state.trake_params["top_k_per_stage"] = st.slider(
                "📊 Max Results per stage",
                5,
                200,
                st.session_state.trake_params["top_k_per_stage"],
                5,
                key="tr_topk_stage",
            )
        with col_tr2:
            st.session_state.trake_params["beam_width"] = st.slider(
                "🧮 Beam width",
                5,
                200,
                st.session_state.trake_params["beam_width"],
                5,
                key="tr_beam_w",
            )
        with col_tr3:
            st.session_state.trake_params["score_threshold"] = st.slider(
                "🎯 Min Score (TRAKE)",
                0.0,
                1.0,
                st.session_state.trake_params["score_threshold"],
                0.05,
                key="tr_score_th",
            )
        with col_tr4:
            st.session_state.trake_params["max_kf_gap"] = st.number_input(
                "Giới hạn max_kf_gap",
                min_value=1,
                max_value=5000,
                value=st.session_state.trake_params["max_kf_gap"],
                step=1,
                key="tr_max_gap",
            )


def render_download_button(placeholder):
    """Vẽ nút Download vào placeholder nếu đã có export_fname."""
    placeholder.empty()  # clear trước
    fname = st.session_state.get("export_fname")
    if not fname:
        return
    try:
        dl_url = f"{st.session_state.api_base_url}/api/v1/keyframe/download"
        r = requests.get(dl_url, params={"fname": fname}, timeout=30)
        if r.status_code == 200:
            with placeholder:
                st.download_button(
                    "Download CSV",
                    data=r.content,
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv_small",
                )
        else:
            with placeholder:
                st.caption("CSV not ready")
    except Exception:
        with placeholder:
            st.caption("CSV not ready")


# ==== Row: Search + Download (same row) ====
btn_col1, btn_col2 = st.columns([8, 2])

with btn_col1:
    do_search = st.button("🔎 Search", use_container_width=True, key="btn_search_any")

with btn_col2:
    download_ph = st.empty()
    render_download_button(download_ph)

# ============== Mode-specific extra params ==============
if search_mode == "Exclude Groups":
    st.markdown("### 🚫 Exclude Groups")
    exclude_groups_input = st.text_input(
        "Group IDs to exclude",
        placeholder="Enter group IDs separated by commas (e.g., 1, 3, 7)",
        help="Keyframes from these groups will be excluded from results",
        key="exclude_grps",
    )
    exclude_groups = []
    if exclude_groups_input.strip():
        try:
            exclude_groups = [
                int(x.strip()) for x in exclude_groups_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")

elif search_mode == "Include Groups & Videos":
    st.markdown("### ✅ Include Groups & Videos")
    col_inc1, col_inc2 = st.columns(2)
    with col_inc1:
        include_groups_input = st.text_input(
            "Group IDs to include",
            placeholder="e.g., 2, 4, 6",
            help="Only search within these groups",
            key="include_grps",
        )
    with col_inc2:
        include_videos_input = st.text_input(
            "Video IDs to include",
            placeholder="e.g., 101, 102, 203",
            help="Only search within these videos",
            key="include_vids",
        )
    include_groups, include_videos = [], []
    if include_groups_input.strip():
        try:
            include_groups = [
                int(x.strip()) for x in include_groups_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Please enter valid group IDs separated by commas")
    if include_videos_input.strip():
        try:
            include_videos = [
                int(x.strip()) for x in include_videos_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Please enter valid video IDs separated by commas")

# ============== Search buttons & logic ==============
if do_search:
    if search_mode in ["Default", "Exclude Groups", "Include Groups & Videos"]:
        if not query.strip():
            st.error("Please enter a search query")
        elif len(query) > 1000:
            st.error("Query too long. Please keep it under 1000 characters.")
        else:
            with st.spinner("🔍 Searching for keyframes..."):
                try:
                    if search_mode == "Default":
                        endpoint = (
                            f"{st.session_state.api_base_url}/api/v1/keyframe/search"
                        )
                        payload = {
                            "query": query,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                        }
                    elif search_mode == "Exclude Groups":
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/exclude-groups"
                        payload = {
                            "query": query,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            "exclude_groups": exclude_groups,
                        }
                    else:  # Include Groups & Videos
                        endpoint = f"{st.session_state.api_base_url}/api/v1/keyframe/search/selected-groups-videos"
                        payload = {
                            "query": query,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            "include_groups": include_groups,
                            "include_videos": include_videos,
                        }

                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.search_results = data.get("results", [])
                        st.session_state.export_fname = (
                            os.path.basename(data.get("export_csv") or "")
                            if data.get("export_csv")
                            else None
                        )

                        render_download_button(download_ph)
                        st.success(
                            f"✅ Found {len(st.session_state.search_results)} results!"
                        )
                    else:
                        st.error(
                            f"❌ API Error: {response.status_code} - {response.text}"
                        )

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Connection Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")

    elif search_mode == "TRAKE (1-5 events)":
        events = [e.strip() for e in st.session_state.trake_events if e.strip()]
        if len(events) == 0:
            st.error("Please enter at least 1 event.")
        else:
            with st.spinner("🔍 Searching..."):
                try:
                    endpoint = (
                        f"{st.session_state.api_base_url}/api/v1/keyframe/trake_search"
                    )
                    payload = {
                        "events": events,
                        "beam_width": int(st.session_state.trake_params["beam_width"]),
                        "top_k": int(st.session_state.trake_params["top_k_per_stage"]),
                        "score_threshold": float(
                            st.session_state.trake_params["score_threshold"]
                        ),
                        "max_kf_gap": int(st.session_state.trake_params["max_kf_gap"]),
                    }
                    resp = requests.post(endpoint, json=payload, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.search_results = [
                            {"path": item["path"], "score": item["score"]}
                            for item in data.get("results", [])
                        ]

                        st.session_state.export_fname = (
                            os.path.basename(data.get("export_csv") or "")
                            if data.get("export_csv")
                            else None
                        )
                        render_download_button(download_ph)

                        if data.get("results"):
                            video_code = data.get("video_code", "")
                            if not video_code:
                                video_code = f"L{data.get('video_group', 0):02d}_V{data.get('video_num', 0):03d}"
                            st.success(
                                f"✅ Aligned {len(data['results'])} keyframes in video {video_code}."
                            )
                        else:
                            st.warning("No valid sequence found.")
                    else:
                        st.error(f"API Error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ============== Results ==============
if st.session_state.search_results:
    st.markdown("---")
    st.markdown("## 📋 Search Results")

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric("Total Results", len(st.session_state.search_results))
    with col_metric2:
        avg_score = sum(r["score"] for r in st.session_state.search_results) / max(
            1, len(st.session_state.search_results)
        )
        st.metric("Average Score", f"{avg_score:.3f}")
    with col_metric3:
        max_score = max(r["score"] for r in st.session_state.search_results)
        st.metric("Best Score", f"{max_score:.3f}")

    sorted_results = sorted(
        st.session_state.search_results, key=lambda x: x["score"], reverse=True
    )

    for i, result in enumerate(sorted_results):
        with st.container():
            col_img, col_info = st.columns([1, 3])
            with col_img:
                try:
                    st.image(
                        os.path.join("data/keyframes", result["path"]),
                        width=200,
                        caption=f"Keyframe {i+1}",
                    )
                except:
                    st.markdown(
                        f"""
                    <div style="
                        background: #f0f0f0;
                        height: 150px;
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border: 2px dashed #ccc;
                    ">
                        <div style="text-align: center; color: #666;">
                            🖼️<br>Image Preview<br>Not Available
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            with col_info:
                st.markdown(
                    f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #333;">Result #{i+1}</h4>
                        <span class="score-badge">Score: {result['score']:.3f}</span>
                    </div>
                    <p style="margin: 0.5rem 0; color: #666;"><strong>Path:</strong> {result['path']}</p>
                    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; font-family: monospace; font-size: 0.9rem;">
                        {result['path'].split('/')[-1]}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

# ============== Footer ==============
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎥 Keyframe Search Application | Built with Streamlit</p>
</div>
""",
    unsafe_allow_html=True,
)
