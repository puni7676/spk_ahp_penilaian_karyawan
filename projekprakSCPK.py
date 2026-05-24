import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="SPK AHP Penilaian Karyawan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "Sistem Penilaian Kinerja dan Gaji Karyawan.csv"

# =========================================================
# INFORMASI KRITERIA
# =========================================================
CRITERIA_INFO = {
    "Performance Score": {
        "kode": "C1",
        "nama": "Performance Score",
        "jenis": "Benefit",
        "keterangan": "Nilai kinerja karyawan. Semakin tinggi nilainya, semakin baik."
    },
    "Experience": {
        "kode": "C2",
        "nama": "Experience",
        "jenis": "Benefit",
        "keterangan": "Pengalaman kerja karyawan. Semakin tinggi, semakin baik."
    },
    "Salary": {
        "kode": "C3",
        "nama": "Salary",
        "jenis": "Cost",
        "keterangan": "Gaji karyawan. Default cost untuk melihat prioritas kelayakan evaluasi gaji."
    },
    "Masa Kerja (Tahun)": {
        "kode": "C4",
        "nama": "Masa Kerja",
        "jenis": "Benefit",
        "keterangan": "Lama karyawan bekerja sejak tanggal bergabung."
    },
    "Status Score": {
        "kode": "C5",
        "nama": "Status Karyawan",
        "jenis": "Benefit",
        "keterangan": "Status karyawan: Active = 1 dan Inactive = 0."
    },
}

CRITERIA = list(CRITERIA_INFO.keys())

# Random Index AHP berdasarkan jumlah kriteria
RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

# Nilai default dibuat cukup konsisten agar nilai CR memenuhi syarat AHP.
DEFAULT_PAIRWISE = {
    ("Performance Score", "Experience"): 2,
    ("Performance Score", "Salary"): 3,
    ("Performance Score", "Masa Kerja (Tahun)"): 3,
    ("Performance Score", "Status Score"): 4,
    ("Experience", "Salary"): 2,
    ("Experience", "Masa Kerja (Tahun)"): 2,
    ("Experience", "Status Score"): 3,
    ("Salary", "Masa Kerja (Tahun)"): 1,
    ("Salary", "Status Score"): 2,
    ("Masa Kerja (Tahun)", "Status Score"): 2,
}

SAATY_OPTIONS = [
    ("9 - Kriteria kiri mutlak lebih penting", 9.0),
    ("8 - Kriteria kiri sangat kuat lebih penting", 8.0),
    ("7 - Kriteria kiri sangat lebih penting", 7.0),
    ("6 - Kriteria kiri kuat lebih penting", 6.0),
    ("5 - Kriteria kiri jelas lebih penting", 5.0),
    ("4 - Kriteria kiri cukup lebih penting", 4.0),
    ("3 - Kriteria kiri sedikit lebih penting", 3.0),
    ("2 - Kriteria kiri agak lebih penting", 2.0),
    ("1 - Sama penting", 1.0),
    ("1/2 - Kriteria kanan agak lebih penting", 1 / 2),
    ("1/3 - Kriteria kanan sedikit lebih penting", 1 / 3),
    ("1/4 - Kriteria kanan cukup lebih penting", 1 / 4),
    ("1/5 - Kriteria kanan jelas lebih penting", 1 / 5),
    ("1/6 - Kriteria kanan kuat lebih penting", 1 / 6),
    ("1/7 - Kriteria kanan sangat lebih penting", 1 / 7),
    ("1/8 - Kriteria kanan sangat kuat lebih penting", 1 / 8),
    ("1/9 - Kriteria kanan mutlak lebih penting", 1 / 9),
]

LABEL_TO_VALUE = dict(SAATY_OPTIONS)
VALUE_TO_LABEL = {round(v, 6): label for label, v in SAATY_OPTIONS}


# =========================================================
# STYLE UI
# =========================================================
def apply_custom_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .hero-box {
            padding: 24px;
            border-radius: 18px;
            background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 55%, #ffffff 100%);
            border: 1px solid #dce8ff;
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 30px;
            font-weight: 800;
            color: #12325c;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            color: #42526e;
            font-size: 16px;
            line-height: 1.6;
        }
        .small-note {
            font-size: 13px;
            color: #5c667a;
        }
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid #e7eaf0;
            padding: 14px;
            border-radius: 14px;
            box-shadow: 0 3px 10px rgba(16, 24, 40, 0.04);
        }
        .ok-box {
            border-left: 5px solid #2e7d32;
            background: #f1f8f3;
            padding: 12px 14px;
            border-radius: 10px;
        }
        .warning-box {
            border-left: 5px solid #ed6c02;
            background: #fff7ed;
            padding: 12px 14px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FUNGSI DATA DAN PREPROCESSING
# =========================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path(DATA_FILE)
    if not path.exists():
        st.error(f"File dataset `{DATA_FILE}` tidak ditemukan. Pastikan file CSV berada satu folder dengan app.py.")
        st.stop()
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Konversi tanggal bergabung menjadi masa kerja dalam tahun.
    data["Joining Date"] = pd.to_datetime(data["Joining Date"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    data["Masa Kerja (Tahun)"] = ((today - data["Joining Date"]).dt.days / 365).clip(lower=0)

    # Isi nilai kosong Performance Score dengan median agar data tetap dapat dihitung.
    data["Performance Score"] = pd.to_numeric(data["Performance Score"], errors="coerce")
    if data["Performance Score"].isna().all():
        data["Performance Score"] = data["Performance Score"].fillna(0)
    else:
        data["Performance Score"] = data["Performance Score"].fillna(data["Performance Score"].median())

    # Konversi status menjadi numerik.
    data["Status Score"] = data["Status"].astype(str).str.lower().map({"active": 1, "inactive": 0}).fillna(0)

    # Pastikan seluruh kriteria berbentuk numerik.
    for col in CRITERIA:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    return data


# =========================================================
# FUNGSI AHP
# =========================================================
def get_label_for_value(value: float) -> str:
    return VALUE_TO_LABEL.get(round(float(value), 6), "1 - Sama penting")


def build_pairwise_matrix(criteria: list[str], pair_values: dict[tuple[str, str], float]) -> pd.DataFrame:
    n = len(criteria)
    matrix = np.ones((n, n), dtype=float)

    for i, c1 in enumerate(criteria):
        for j, c2 in enumerate(criteria):
            if i < j:
                value = float(pair_values[(c1, c2)])
                matrix[i, j] = value
                matrix[j, i] = 1 / value

    index_labels = [f"{CRITERIA_INFO[c]['kode']} - {CRITERIA_INFO[c]['nama']}" for c in criteria]
    return pd.DataFrame(matrix, index=index_labels, columns=index_labels)


def calculate_ahp(pairwise_df: pd.DataFrame):
    matrix = pairwise_df.values.astype(float)
    n = matrix.shape[0]

    column_sum = matrix.sum(axis=0)
    normalized = matrix / column_sum
    weights = normalized.mean(axis=1)

    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0

    weight_df = pd.DataFrame({
        "Kode": [CRITERIA_INFO[c]["kode"] for c in CRITERIA],
        "Kriteria": CRITERIA,
        "Nama Kriteria": [CRITERIA_INFO[c]["nama"] for c in CRITERIA],
        "Bobot": weights,
        "Bobot (%)": weights * 100,
    }).sort_values("Bobot", ascending=False).reset_index(drop=True)

    normalized_df = pd.DataFrame(normalized, index=pairwise_df.index, columns=pairwise_df.columns)
    return normalized_df, weight_df, lambda_max, ci, cr


def normalize_alternatives(data: pd.DataFrame, criteria_types: dict[str, str]) -> pd.DataFrame:
    norm = pd.DataFrame(index=data.index)

    for criterion, ctype in criteria_types.items():
        values = pd.to_numeric(data[criterion], errors="coerce").fillna(0).astype(float)

        if ctype == "Benefit":
            max_val = values.max()
            norm[criterion] = values / max_val if max_val != 0 else 0
        else:
            safe_values = values.replace(0, np.nan)
            min_val = safe_values.min()
            norm[criterion] = min_val / safe_values
            norm[criterion] = norm[criterion].replace([np.inf, -np.inf], np.nan).fillna(0)

    return norm


def calculate_ranking(data: pd.DataFrame, weight_df: pd.DataFrame, criteria_types: dict[str, str]) -> pd.DataFrame:
    weight_series = weight_df.set_index("Kriteria").loc[CRITERIA, "Bobot"]
    normalized_alt = normalize_alternatives(data, criteria_types)
    score = normalized_alt[CRITERIA].dot(weight_series)

    result = data.copy()
    result["Skor AHP"] = score
    result = result.sort_values("Skor AHP", ascending=False).reset_index(drop=True)
    result.insert(0, "Peringkat", range(1, len(result) + 1))
    return result, normalized_alt


# =========================================================
# SIDEBAR INPUT DINAMIS
# =========================================================
def sidebar_inputs(processed_df: pd.DataFrame):
    st.sidebar.title("📊 SPK AHP Karyawan")
    st.sidebar.caption("Aplikasi Streamlit untuk proyek akhir Praktikum SCPK.")

    st.sidebar.header("📌 Navigasi")
    menu = st.sidebar.radio(
        "Pilih halaman",
        ["Dashboard", "Dataset", "Preprocessing", "Pemodelan AHP", "Perhitungan AHP", "Hasil Ranking", "Profil Kelompok"],
        label_visibility="collapsed"
    )

    st.sidebar.header("🔎 Filter Dataset")
    departments = sorted(processed_df["Department"].dropna().unique().tolist())
    selected_dept = st.sidebar.multiselect("Department", departments, default=departments)

    statuses = ["Semua"] + sorted(processed_df["Status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Status Karyawan", statuses)

    filtered = processed_df.copy()
    if selected_dept:
        filtered = filtered[filtered["Department"].isin(selected_dept)]
    if selected_status != "Semua":
        filtered = filtered[filtered["Status"] == selected_status]

    st.sidebar.header("⚙️ Input Dinamis AHP")
    st.sidebar.caption("Ubah nilai perbandingan kriteria menggunakan skala Saaty.")

    pair_values = {}
    with st.sidebar.expander("Matriks Perbandingan Berpasangan", expanded=False):
        for i in range(len(CRITERIA)):
            for j in range(i + 1, len(CRITERIA)):
                c1, c2 = CRITERIA[i], CRITERIA[j]
                default_value = DEFAULT_PAIRWISE.get((c1, c2), 1)
                default_label = get_label_for_value(default_value)
                left_label = f"{CRITERIA_INFO[c1]['kode']} {CRITERIA_INFO[c1]['nama']}"
                right_label = f"{CRITERIA_INFO[c2]['kode']} {CRITERIA_INFO[c2]['nama']}"
                selected = st.select_slider(
                    f"{left_label} dibanding {right_label}",
                    options=[opt[0] for opt in SAATY_OPTIONS],
                    value=default_label,
                    key=f"pair_{i}_{j}",
                )
                pair_values[(c1, c2)] = LABEL_TO_VALUE[selected]

    salary_type = st.sidebar.radio(
        "Jenis Kriteria Salary",
        options=["Cost", "Benefit"],
        index=0,
        help="Cost: gaji lebih rendah dengan performa baik lebih diprioritaskan. Benefit: gaji tinggi dianggap indikator pencapaian."
    )

    return menu, filtered, pair_values, salary_type


# =========================================================
# KOMPONEN TAMPILAN
# =========================================================
def show_hero():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Sistem Pendukung Keputusan Penilaian Kinerja dan Gaji Karyawan</div>
            <div class="hero-subtitle">
                Metode yang digunakan adalah <b>Analytical Hierarchy Process (AHP)</b>. Sistem ini membantu menentukan ranking karyawan
                berdasarkan beberapa kriteria, menampilkan proses perhitungan, nilai Consistency Ratio (CR), dan grafik hasil akhir.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_metrics(raw_df: pd.DataFrame, filtered_df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", f"{len(raw_df):,}")
    col2.metric("Data Setelah Filter", f"{len(filtered_df):,}")
    col3.metric("Jumlah Kriteria", len(CRITERIA))
    col4.metric("Missing Performance", int(raw_df["Performance Score"].isna().sum()))


def criteria_table(criteria_types: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Kode": CRITERIA_INFO[c]["kode"],
            "Kriteria": CRITERIA_INFO[c]["nama"],
            "Jenis": criteria_types[c],
            "Keterangan": CRITERIA_INFO[c]["keterangan"],
        }
        for c in CRITERIA
    ])


def save_result_to_session(pairwise_df, normalized_df, weight_df, lambda_max, ci, cr, result_df, normalized_alt, criteria_types):
    st.session_state["pairwise_df"] = pairwise_df
    st.session_state["normalized_df"] = normalized_df
    st.session_state["weight_df"] = weight_df
    st.session_state["lambda_max"] = lambda_max
    st.session_state["ci"] = ci
    st.session_state["cr"] = cr
    st.session_state["result_df"] = result_df
    st.session_state["normalized_alt"] = normalized_alt
    st.session_state["criteria_types"] = criteria_types


# =========================================================
# MAIN PROGRAM
# =========================================================
apply_custom_css()
raw_df = load_data()
processed_df = preprocess_data(raw_df)
menu, filtered_df, pair_values, salary_type = sidebar_inputs(processed_df)

criteria_types = {c: CRITERIA_INFO[c]["jenis"] for c in CRITERIA}
criteria_types["Salary"] = salary_type

show_hero()
show_metrics(raw_df, filtered_df)
st.divider()

# =========================================================
# HALAMAN DASHBOARD
# =========================================================
if menu == "Dashboard":
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Ringkasan Proyek")
        st.write(
            "Proyek ini membuat aplikasi Sistem Pendukung Keputusan berbasis Streamlit untuk membantu proses penilaian kinerja dan "
            "kelayakan gaji karyawan. Metode AHP digunakan untuk menentukan bobot tiap kriteria melalui perbandingan berpasangan, "
            "lalu menghasilkan ranking karyawan dari skor tertinggi sampai terendah."
        )

        st.markdown("### Kriteria yang Digunakan")
        st.dataframe(criteria_table(criteria_types), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Kesesuaian dengan Ketentuan Proyek")
        checklist = pd.DataFrame({
            "No": [1, 2, 3, 4, 5, 6, 7],
            "Ketentuan": [
                "GUI menggunakan Streamlit",
                "Navigasi menggunakan Sidebar",
                "Dataset ditampilkan dengan st.dataframe",
                "Minimal 3 input widget dinamis",
                "Perhitungan dipicu tombol st.button",
                "Tabel ranking diurutkan dari tertinggi",
                "AHP menampilkan matriks dan nilai CR",
            ],
            "Status": ["Terpenuhi"] * 7,
        })
        st.dataframe(checklist, use_container_width=True, hide_index=True)
        st.info("Untuk menjalankan perhitungan, buka halaman **Perhitungan AHP** lalu klik tombol **Hitung AHP dan Ranking**.")

# =========================================================
# HALAMAN DATASET
# =========================================================
elif menu == "Dataset":
    st.subheader("Dataset Mentah")
    st.write("Dataset mentah ditampilkan dalam bentuk tabel interaktif menggunakan `st.dataframe`.")

    tab1, tab2, tab3 = st.tabs(["Tabel Dataset", "Informasi Kolom", "Statistik Ringkas"])
    with tab1:
        st.dataframe(raw_df, use_container_width=True)
    with tab2:
        info_df = pd.DataFrame({
            "Kolom": raw_df.columns,
            "Tipe Data": [str(dtype) for dtype in raw_df.dtypes],
            "Jumlah Missing": raw_df.isna().sum().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)

# =========================================================
# HALAMAN PREPROCESSING
# =========================================================
elif menu == "Preprocessing":
    st.subheader("Preprocessing Data")
    st.write(
        "Preprocessing dilakukan agar data siap digunakan dalam perhitungan AHP. Proses ini mencakup pembersihan nilai kosong, "
        "konversi tanggal menjadi masa kerja, dan konversi status karyawan menjadi nilai numerik."
    )

    steps = pd.DataFrame({
        "No": [1, 2, 3, 4],
        "Tahap": [
            "Cek missing value",
            "Isi nilai kosong Performance Score",
            "Konversi Joining Date",
            "Konversi Status",
        ],
        "Keterangan": [
            f"Performance Score kosong sebanyak {raw_df['Performance Score'].isna().sum()} data.",
            "Nilai kosong diisi menggunakan median.",
            "Joining Date diubah menjadi Masa Kerja (Tahun).",
            "Active = 1 dan Inactive = 0.",
        ],
    })
    st.dataframe(steps, use_container_width=True, hide_index=True)

    st.markdown("### Data Setelah Preprocessing dan Filter")
    st.dataframe(filtered_df, use_container_width=True)

# =========================================================
# HALAMAN PEMODELAN AHP
# =========================================================
elif menu == "Pemodelan AHP":
    st.subheader("Pemodelan Sistem Pendukung Keputusan")

    st.markdown("### Struktur Hierarki AHP")
    hierarchy = pd.DataFrame({
        "Level": ["Level 1", "Level 2", "Level 3"],
        "Komponen": ["Tujuan", "Kriteria", "Alternatif"],
        "Isi": [
            "Menentukan ranking penilaian kinerja dan kelayakan gaji karyawan",
            "Performance Score, Experience, Salary, Masa Kerja, dan Status Karyawan",
            "Seluruh karyawan dalam dataset",
        ],
    })
    st.dataframe(hierarchy, use_container_width=True, hide_index=True)

    st.markdown("### Skala Penilaian AHP")
    scale_df = pd.DataFrame({
        "Nilai": [1, 3, 5, 7, 9, "2, 4, 6, 8"],
        "Keterangan": [
            "Sama penting",
            "Sedikit lebih penting",
            "Lebih penting",
            "Sangat lebih penting",
            "Mutlak lebih penting",
            "Nilai antara dua tingkat kepentingan",
        ],
    })
    st.dataframe(scale_df, use_container_width=True, hide_index=True)

    st.markdown("### Kriteria")
    st.dataframe(criteria_table(criteria_types), use_container_width=True, hide_index=True)

# =========================================================
# HALAMAN PERHITUNGAN AHP
# =========================================================
elif menu == "Perhitungan AHP":
    st.subheader("Perhitungan AHP")
    st.write(
        "Halaman ini menampilkan proses utama metode AHP, mulai dari matriks perbandingan berpasangan, normalisasi matriks, "
        "bobot prioritas, hingga nilai Consistency Ratio (CR)."
    )

    col_btn, col_note = st.columns([0.35, 0.65])
    with col_btn:
        run_button = st.button("Hitung AHP dan Ranking", type="primary", use_container_width=True)
    with col_note:
        st.caption("Perhitungan tidak berjalan otomatis. Klik tombol untuk memenuhi ketentuan proyek.")

    if run_button:
        if len(filtered_df) == 0:
            st.error("Data kosong setelah difilter. Ubah filter data terlebih dahulu.")
            st.stop()

        pairwise_df = build_pairwise_matrix(CRITERIA, pair_values)
        normalized_df, weight_df, lambda_max, ci, cr = calculate_ahp(pairwise_df)
        result_df, normalized_alt = calculate_ranking(filtered_df, weight_df, criteria_types)
        save_result_to_session(pairwise_df, normalized_df, weight_df, lambda_max, ci, cr, result_df, normalized_alt, criteria_types)
        st.success("Perhitungan AHP berhasil dijalankan.")

    if "pairwise_df" not in st.session_state:
        st.warning("Perhitungan belum dijalankan. Klik tombol **Hitung AHP dan Ranking** terlebih dahulu.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Matriks", "Normalisasi", "Bobot Kriteria", "Uji Konsistensi"])

        with tab1:
            st.markdown("### Matriks Perbandingan Berpasangan")
            st.dataframe(st.session_state["pairwise_df"].round(4), use_container_width=True)

        with tab2:
            st.markdown("### Normalisasi Matriks")
            st.dataframe(st.session_state["normalized_df"].round(4), use_container_width=True)

        with tab3:
            st.markdown("### Bobot Prioritas Kriteria")
            st.dataframe(st.session_state["weight_df"].round(4), use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            plot_df = st.session_state["weight_df"].sort_values("Bobot", ascending=True)
            ax.barh(plot_df["Nama Kriteria"], plot_df["Bobot"])
            ax.set_xlabel("Bobot")
            ax.set_title("Bobot Prioritas Kriteria AHP")
            st.pyplot(fig)

        with tab4:
            st.markdown("### Uji Konsistensi")
            col1, col2, col3 = st.columns(3)
            col1.metric("λ Maks", f"{st.session_state['lambda_max']:.4f}")
            col2.metric("CI", f"{st.session_state['ci']:.4f}")
            col3.metric("CR", f"{st.session_state['cr']:.4f}")

            if st.session_state["cr"] <= 0.1:
                st.markdown("<div class='ok-box'><b>Status:</b> Nilai CR ≤ 0,10 sehingga matriks dinyatakan konsisten.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'><b>Status:</b> Nilai CR > 0,10 sehingga matriks belum konsisten. Ubah input perbandingan pada sidebar.</div>", unsafe_allow_html=True)

# =========================================================
# HALAMAN HASIL RANKING
# =========================================================
elif menu == "Hasil Ranking":
    st.subheader("Hasil Perangkingan Karyawan")

    if "result_df" not in st.session_state:
        st.warning("Belum ada hasil ranking. Buka halaman **Perhitungan AHP**, lalu klik tombol perhitungan.")
    else:
        result_df = st.session_state["result_df"]
        selected_cols = [
            "Peringkat", "ID", "Name", "Department", "Status", "Salary", "Performance Score",
            "Experience", "Masa Kerja (Tahun)", "Skor AHP"
        ]

        st.markdown("### Tabel Hasil Ranking")
        st.dataframe(result_df[selected_cols].round(4), use_container_width=True, hide_index=True)

        csv_data = result_df[selected_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Hasil Ranking CSV",
            data=csv_data,
            file_name="hasil_ranking_ahp.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("### Grafik Top 10 Karyawan")
        top10 = result_df.head(10).sort_values("Skor AHP", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top10["Name"], top10["Skor AHP"])
        ax.set_xlabel("Skor AHP")
        ax.set_ylabel("Nama Karyawan")
        ax.set_title("Top 10 Karyawan Berdasarkan Skor AHP")
        st.pyplot(fig)

        st.markdown("### Interpretasi")
        best = result_df.iloc[0]
        st.success(
            f"Peringkat pertama adalah {best['Name']} dengan skor AHP {best['Skor AHP']:.4f}. "
            "Karyawan dengan skor tertinggi dapat menjadi rekomendasi utama berdasarkan kriteria yang digunakan."
        )

# =========================================================
# HALAMAN PROFIL KELOMPOK
# =========================================================
elif menu == "Profil Kelompok":
    st.subheader("Profil Kelompok")
    st.write("Silakan ganti nama dan NIM sesuai anggota kelompok masing-masing sebelum dikumpulkan.")

    profil = pd.DataFrame({
        "Nama": ["Adelia Nurrahmawati", "Sri Arwati", "Puput Septiani"],
        "NIM": ["123240048", "123240255", "123240263"],
        "Kelas": ["IF-F", "IF-F", "IF-F"],
        "Peran": ["Analis Sistem dan Penyunting Laporan", "Pengolah Data dan Perhitungan AHP", "Pengembang Aplikasi dan Penyusun Laporan"],
    })
    st.dataframe(profil, use_container_width=True, hide_index=True)

    st.markdown("### Tentang Aplikasi")
    st.write(
        "Aplikasi ini dibuat untuk memenuhi proyek akhir Praktikum Sistem Cerdas Pendukung Keputusan. "
        "Metode yang digunakan adalah Analytical Hierarchy Process (AHP) dengan studi kasus penilaian kinerja dan kelayakan gaji karyawan."
    )
