"""
app.py
======
Streamlit Dashboard — ML Auto-Tagging untuk file Dataset Komdigi

Navigasi:
  🏠 Overview              → Statistik dataset & distribusi label
  ⚙️ Training & Comparison → Train 5 model, bandingkan, pilih terbaik
  🏷️ Auto-Tag Dataset      → Klasifikasi semua file di dataset/
  📤 Upload & Classify     → Upload file baru, prediksi, simpan ke training data
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Pastikan import module lokal bisa ditemukan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import BASE_DIR, get_all_dataset_files
from trainer import (
    train_all_models, save_comparison, load_comparison,
    load_best_model, MODELS_DIR, COMPARISON_PATH
)
from classifier import (
    predict_file, classify_all_files, save_results,
    save_uploaded_to_training,
    CSV_PATH, DATASET_FOLDER, RESULTS_CSV
)

logging.basicConfig(level=logging.INFO)

# ================================================================
# Konfigurasi Halaman
# ================================================================
st.set_page_config(
    page_title="ML Auto-Tagging Dashboard",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Custom CSS
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebarNavLink"] { border-radius: 8px !important; }

    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: rgba(99,179,237,0.5); }
    .metric-card .number { font-size: 2.4rem; font-weight: 700; color: #63b3ed; }
    .metric-card .label  { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    .best-badge {
        background: linear-gradient(135deg, #38a169, #276749);
        color: white !important;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: opacity 0.2s, transform 0.15s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #63b3ed; margin-bottom: 12px;
        border-bottom: 1px solid rgba(99,179,237,0.2);
        padding-bottom: 8px;
    }
    .tag-group { display:inline-block; background:#2d3748; color:#68d391; padding:2px 10px; border-radius:12px; font-size:0.78rem; margin:2px; }
    .tag-org   { display:inline-block; background:#2d3748; color:#63b3ed; padding:2px 10px; border-radius:12px; font-size:0.78rem; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# Helper Functions
# ================================================================

@st.cache_data(show_spinner=False)
def load_csv():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)

def reload_csv():
    load_csv.clear()
    return load_csv()

def models_trained() -> bool:
    return (
        os.path.exists(os.path.join(MODELS_DIR, "best_group.joblib")) and
        os.path.exists(os.path.join(MODELS_DIR, "best_organization.joblib"))
    )

# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:2.5rem;'>🏷️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#63b3ed;'>ML Auto-Tagger</div>
        <div style='font-size:0.75rem; color:#718096; margin-top:4px;'>Dataset Komdigi Classifier</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1); margin: 10px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["🏠 Overview", "⚙️ Training & Model Comparison", "🏷️ Auto-Tag Dataset", "📤 Upload & Classify"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin: 16px 0;'>", unsafe_allow_html=True)

    df_global = load_csv()
    n_files = len(get_all_dataset_files(DATASET_FOLDER))
    st.markdown(f"<div style='font-size:0.8rem; color:#718096; padding:0 8px;'>📁 {n_files} file di dataset/<br>📋 {len(df_global)} baris training</div>", unsafe_allow_html=True)

    if models_trained():
        st.markdown("<div style='margin-top:10px'><span class='best-badge'>✓ Model Terlatih</span></div>", unsafe_allow_html=True)


# ================================================================
# PAGE 1: OVERVIEW
# ================================================================
if page == "🏠 Overview":
    st.markdown("# 🏠 Overview Dataset")
    df = load_csv()

    if df.empty:
        st.warning("File training_data_enriched.csv tidak ditemukan.")
    else:
        files = get_all_dataset_files(DATASET_FOLDER)
        n_with_link = (df["link_file"].fillna("") != "").sum() if "link_file" in df.columns else 0
        groups       = df["group"].fillna("").replace("", None).dropna().unique()
        orgs         = df["organization"].fillna("").replace("", None).dropna().unique()

        # Metric Cards
        cols = st.columns(4)
        metrics = [
            (str(len(df)),       "Total Entri CSV"),
            (str(len(files)),    "File di Dataset"),
            (str(len(groups)),   "Unique Group"),
            (str(len(orgs)),     "Unique Org"),
        ]
        for col, (num, label) in zip(cols, metrics):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='number'>{num}</div>
                <div class='label'>{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-title'>📊 Distribusi Group</div>", unsafe_allow_html=True)
            grp_counts = df["group"].fillna("(kosong)").replace("", "(kosong)").value_counts().reset_index()
            grp_counts.columns = ["group", "count"]
            fig = px.bar(
                grp_counts, x="count", y="group", orientation="h",
                color="count", color_continuous_scale="blues",
                labels={"count": "Jumlah", "group": "Group"},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", height=380,
                coloraxis_showscale=False, margin=dict(l=0, r=10, t=10, b=0),
                yaxis=dict(categoryorder="total ascending"),
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<div class='section-title'>🏢 Distribusi Organization</div>", unsafe_allow_html=True)
            org_counts = df["organization"].fillna("(kosong)").replace("", "(kosong)").value_counts().reset_index()
            org_counts.columns = ["organization", "count"]
            fig2 = px.pie(
                org_counts, values="count", names="organization",
                hole=0.45, color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
                height=380, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font_size=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-title'>📋 Preview Data Training</div>", unsafe_allow_html=True)
        show_cols = [c for c in ["title", "name", "group", "organization", "link_file"] if c in df.columns]
        st.dataframe(df[show_cols].head(30), use_container_width=True, height=350)


# ================================================================
# PAGE 2: TRAINING & MODEL COMPARISON
# ================================================================
elif page == "⚙️ Training & Model Comparison":
    st.markdown("# ⚙️ Training & Model Comparison")
    st.markdown("Bandingkan **5 model ML** dengan **GridSearchCV** (5-fold stratified CV) untuk dua target label: `group` dan `organization`.")

    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        run_training = st.button("🚀 Mulai Training", type="primary", use_container_width=True)

    if run_training:
        progress_bar = st.progress(0)
        status_text  = st.empty()
        model_names  = ["Logistic Regression", "Random Forest", "Linear SVC", "Gradient Boosting", "Multinomial NB"]

        try:
            # --- Training Group ---
            status_text.markdown("**Training label: `group`...**")

            def cb_group(name, cur, total):
                progress_bar.progress(int((cur - 1) / (total * 2) * 100))
                status_text.markdown(f"**group** → `{name}` ({cur}/{total})")

            group_results = train_all_models("group", progress_callback=cb_group)

            # --- Training Organization ---
            def cb_org(name, cur, total):
                progress_bar.progress(int((total + cur - 1) / (total * 2) * 100))
                status_text.markdown(f"**organization** → `{name}` ({cur}/{total})")

            org_results = train_all_models("organization", progress_callback=cb_org)

            save_comparison(group_results, org_results)
            progress_bar.progress(100)
            status_text.markdown("✅ Training selesai!")
            st.success("Semua model berhasil ditraining dan disimpan!")
            load_csv.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Error saat training: {e}")
            progress_bar.empty()

    # --- Tampilkan Hasil ---
    comparison = load_comparison()
    if comparison:
        st.markdown("<br>", unsafe_allow_html=True)

        for label_key, label_display in [("group", "🏷️ Group"), ("organization", "🏢 Organization")]:
            res = comparison.get(label_key, {})
            if not res:
                continue

            best_model = res.get("_best_model", "")
            trained_at = res.get("_trained_at", "")[:19].replace("T", " ") if res.get("_trained_at") else ""
            n_samples  = res.get("_n_samples", "-")
            n_classes  = res.get("_n_classes", "-")

            with st.expander(f"{label_display} — Best: **{best_model}** | {n_samples} sampel, {n_classes} kelas | {trained_at}", expanded=True):
                model_names_list = [k for k in res if not k.startswith("_")]
                rows = []
                for mn in model_names_list:
                    m  = res[mn]
                    is_best = mn == best_model
                    rows.append({
                        "Model":          f"⭐ {mn}" if is_best else mn,
                        "CV F1 (val)":    m["cv_f1"],
                        "Train Accuracy": m["accuracy"],
                        "Train F1":       m["f1_weighted"],
                        "Best Params":    str(m["best_params"]),
                    })

                df_res = pd.DataFrame(rows).sort_values("CV F1 (val)", ascending=False)

                # Bar chart comparison
                fig = go.Figure()
                colors_cv   = ["#38a169" if r["Model"].startswith("⭐") else "#4299e1" for _, r in df_res.iterrows()]
                fig.add_trace(go.Bar(
                    name="CV F1 (validasi)",
                    x=df_res["Model"], y=df_res["CV F1 (val)"],
                    marker_color=colors_cv, text=df_res["CV F1 (val)"].round(3), textposition="inside",
                ))
                fig.add_trace(go.Bar(
                    name="Train Accuracy",
                    x=df_res["Model"], y=df_res["Train Accuracy"],
                    marker_color="rgba(159,122,234,0.7)", text=df_res["Train Accuracy"].round(3), textposition="inside",
                ))
                fig.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e0e0e0", height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(range=[0, 1.1]),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada hasil training. Klik **🚀 Mulai Training** untuk memulai.")


# ================================================================
# PAGE 3: AUTO-TAG DATASET (satu per satu)
# ================================================================
elif page == "🏷️ Auto-Tag Dataset":
    st.markdown("# 🏷️ Auto-Tag Dataset")
    st.markdown("Pilih satu file dari folder `dataset/`, lihat prediksi label, lalu simpan hasilnya.")

    if not models_trained():
        st.warning("⚠️ Model belum ditraining. Silakan ke halaman **⚙️ Training & Model Comparison** terlebih dahulu.")
    else:
        files = sorted(get_all_dataset_files(DATASET_FOLDER))

        if not files:
            st.info("Folder `dataset/` kosong.")
        else:
            # Load history untuk tahu file mana yg sudah di-tag
            tagged_files = set()
            if os.path.exists(RESULTS_CSV):
                df_hist = pd.read_csv(RESULTS_CSV)
                tagged_files = set(df_hist["filename"].tolist())

            # Pisahkan file yang belum dan sudah di-tag
            untagged = [f for f in files if f not in tagged_files]
            tagged   = [f for f in files if f in tagged_files]

            # Progress indicator
            col_prog1, col_prog2, col_prog3 = st.columns(3)
            col_prog1.markdown(f"""<div class='metric-card'>
                <div class='number'>{len(files)}</div>
                <div class='label'>Total File</div></div>""", unsafe_allow_html=True)
            col_prog2.markdown(f"""<div class='metric-card'>
                <div class='number'>{len(tagged)}</div>
                <div class='label'>Sudah Di-tag</div></div>""", unsafe_allow_html=True)
            col_prog3.markdown(f"""<div class='metric-card'>
                <div class='number'>{len(untagged)}</div>
                <div class='label'>Belum Di-tag</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabs: Pilih & Tag | Riwayat
            tab1, tab2 = st.tabs(["🔍 Pilih & Tag File", "📋 Riwayat Tag"])

            with tab1:
                # Filter opsi tampilan
                col_opt1, col_opt2 = st.columns([1, 3])
                with col_opt1:
                    show_filter = st.radio(
                        "Tampilkan",
                        ["Belum di-tag", "Semua file", "Sudah di-tag"],
                        horizontal=False,
                    )

                if show_filter == "Belum di-tag":
                    file_options = untagged
                elif show_filter == "Sudah di-tag":
                    file_options = tagged
                else:
                    file_options = files

                with col_opt2:
                    search_q = st.text_input("🔎 Cari nama file", placeholder="ketik untuk filter...")
                    if search_q:
                        file_options = [f for f in file_options if search_q.lower() in f.lower()]

                if not file_options:
                    st.info("Tidak ada file yang sesuai filter.")
                else:
                    st.markdown(f"**{len(file_options)} file tersedia** — pilih satu untuk diklasifikasi:")

                    selected_file = st.selectbox(
                        "Pilih file dataset",
                        file_options,
                        format_func=lambda f: (
                            f"✅ {f}" if f in tagged_files else f"🗂️ {f}"
                        ),
                        label_visibility="collapsed",
                    )

                    # Tombol klasifikasi
                    col_tag, col_skip, _ = st.columns([1, 1, 4])
                    with col_tag:
                        do_tag = st.button("🔍 Klasifikasi File Ini", type="primary", use_container_width=True)

                    if do_tag or selected_file:
                        # Prediksi otomatis saat file dipilih
                        pred = predict_file(selected_file)

                        # Hasil prediksi
                        st.markdown("---")
                        st.markdown(f"### 📄 `{selected_file}`")

                        r1, r2 = st.columns(2)
                        with r1:
                            st.markdown("<div class='section-title'>🏷️ Prediksi Group</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:1.5rem; font-weight:700; color:#68d391; padding:12px 0;'>{pred['pred_group']}</div>", unsafe_allow_html=True)

                            # Override group
                            df_csv = load_csv()
                            all_groups = sorted(df_csv["group"].fillna("").replace("", None).dropna().unique().tolist())
                            override_group = st.selectbox(
                                "Koreksi Group (opsional)",
                                ["(gunakan prediksi)"] + all_groups,
                                key="ovr_group",
                            )

                        with r2:
                            st.markdown("<div class='section-title'>🏢 Prediksi Organization</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size:1.5rem; font-weight:700; color:#63b3ed; padding:12px 0;'>{pred['pred_organization']}</div>", unsafe_allow_html=True)

                            all_orgs = sorted(df_csv["organization"].fillna("").replace("", None).dropna().unique().tolist())
                            override_org = st.selectbox(
                                "Koreksi Organization (opsional)",
                                ["(gunakan prediksi)"] + all_orgs,
                                key="ovr_org",
                            )

                        # Tentukan nilai final
                        final_group = pred["pred_group"] if override_group == "(gunakan prediksi)" else override_group
                        final_org   = pred["pred_organization"] if override_org == "(gunakan prediksi)" else override_org

                        st.markdown("<br>", unsafe_allow_html=True)
                        col_save, col_dl, _ = st.columns([1, 1, 4])

                        with col_save:
                            if st.button("💾 Simpan Tag", type="primary", use_container_width=True):
                                row_df = pd.DataFrame([{
                                    "filename":          selected_file,
                                    "pred_group":        final_group,
                                    "pred_organization": final_org,
                                    "classified_at":     pred["classified_at"],
                                    "source":            "manual" if (override_group != "(gunakan prediksi)" or override_org != "(gunakan prediksi)") else "auto",
                                }])
                                save_results(row_df)
                                st.success(f"✅ Tag disimpan ke `classification_results.csv`: **{final_group}** | **{final_org}**")
                                st.rerun()

                        with col_dl:
                            single_csv = pd.DataFrame([{
                                "filename": selected_file,
                                "group": final_group,
                                "organization": final_org,
                            }]).to_csv(index=False).encode()
                            st.download_button(
                                "⬇️ Download",
                                data=single_csv,
                                file_name=f"tag_{selected_file}.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )

            with tab2:
                if os.path.exists(RESULTS_CSV):
                    df_hist_full = pd.read_csv(RESULTS_CSV)
                    st.markdown(f"**{len(df_hist_full)} file sudah di-tag:**")

                    # Search di riwayat
                    search_hist = st.text_input("🔎 Cari di riwayat", placeholder="ketik nama file...", key="hist_search")
                    df_display  = df_hist_full
                    if search_hist:
                        df_display = df_hist_full[df_hist_full["filename"].str.contains(search_hist, case=False, na=False)]

                    st.dataframe(df_display, use_container_width=True, height=450)

                    csv_all = df_hist_full.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Semua Hasil", data=csv_all,
                                       file_name="classification_results.csv",
                                       mime="text/csv")
                else:
                    st.info("Belum ada file yang di-tag. Mulai dari tab 'Pilih & Tag File'.")


# ================================================================
# PAGE 4: UPLOAD & CLASSIFY
# ================================================================
elif page == "📤 Upload & Classify":
    st.markdown("# 📤 Upload & Classify File Baru")
    st.markdown("Upload file baru untuk diklasifikasi secara otomatis. File akan dipindahkan ke folder `dataset/` dan data training akan diperbarui.")

    if not models_trained():
        st.warning("⚠️ Model belum ditraining. Silakan ke halaman **⚙️ Training & Model Comparison** terlebih dahulu.")
    else:
        uploaded_files = st.file_uploader(
            "Pilih file untuk diupload",
            accept_multiple_files=True,
            type=["csv", "xlsx", "xls", "pdf", "json", "zip", "docx", "txt"],
            help="Mendukung: CSV, XLSX, XLS, PDF, JSON, ZIP, DOCX, TXT",
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file dipilih:**")

            os.makedirs(DATASET_FOLDER, exist_ok=True)
            results_upload = []

            for uf in uploaded_files:
                dest_path = os.path.join(DATASET_FOLDER, uf.name)
                # Simpan ke dataset/
                with open(dest_path, "wb") as f:
                    f.write(uf.getbuffer())
                # Prediksi
                pred = predict_file(uf.name)
                results_upload.append(pred)

            df_upload = pd.DataFrame(results_upload)

            # Tampilkan hasil prediksi
            st.markdown("### Hasil Klasifikasi")
            for _, row in df_upload.iterrows():
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 2])
                    c1.markdown(f"📄 **{row['filename']}**")
                    c2.markdown(f"<span class='tag-group'>{row['pred_group']}</span>", unsafe_allow_html=True)
                    c3.markdown(f"<span class='tag-org'>{row['pred_organization']}</span>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_a, col_b, _ = st.columns([1, 1, 3])
            with col_a:
                if st.button("✅ Konfirmasi & Simpan ke Training Data", type="primary", use_container_width=True):
                    added = 0
                    for _, row in df_upload.iterrows():
                        save_uploaded_to_training(
                            filename=row["filename"],
                            pred_group=row["pred_group"],
                            pred_organization=row["pred_organization"],
                        )
                        added += 1
                    save_results(df_upload)
                    load_csv.clear()
                    st.success(f"✅ {added} file ditambahkan ke training data dan `classification_results.csv`!")
                    st.balloons()

            with col_b:
                csv_bytes = df_upload.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Hasil",
                    data=csv_bytes,
                    file_name="upload_classification.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Tampilkan history upload (dari results CSV)
        if os.path.exists(RESULTS_CSV):
            st.markdown("---")
            st.markdown("### 📋 History Klasifikasi Sebelumnya")
            df_hist = pd.read_csv(RESULTS_CSV)
            st.dataframe(df_hist.tail(20), use_container_width=True, height=300)
