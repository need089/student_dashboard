import streamlit as st
import pandas as pd
import plotly.express as px

from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="แดชบอร์ดแสดงผลสัมฤทธิ์ทางการเรียนของผู้เรียน",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# LOAD DATA
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("xAPI-Edu-Data.csv")

df = load_data()

# ------------------------
# SIDEBAR NAVIGATION & FILTERS
# ------------------------
with st.sidebar:
    # 🎨 Custom CSS สำหรับ Sidebar และ ตัวกรองแบบใหม่ (ธีมน้ำเงินเข้ม)
    # --- Custom CSS (รวมสไตล์ทั้งหมด) ---
    st.markdown("""
        <style>
            /* 1. เปลี่ยนสีพื้นหลังของ Sidebar */
            [data-testid="stSidebar"] {
                background-color: #0A2540 !important;
                border-right: 1px solid #1E293B;
            }
            div[data-testid="stSidebarHeader"] button,
            button[data-testid="stSidebarCollapseButton"],
            button[aria-label="Close sidebar"] {
                opacity: 1 !important; /* ปิดความจาง */
                color: #FFFFFF !important;
            }   

            /* ย้อมสีองค์ประกอบ SVG/Path ข้างในปุ่มให้เป็นสีขาวสว่าง 100% */
            div[data-testid="stSidebarHeader"] button *,
            button[data-testid="stSidebarCollapseButton"] *,
            button[aria-label="Close sidebar"] * {
                fill: #FFFFFF !important;
                stroke: #FFFFFF !important;
                color: #FFFFFF !important;
                opacity: 1 !important;
                filter: drop-shadow(0px 0px 1px #FFFFFF) !important; /* เพิ่มความคมชัด */
            }

            /* 2. สไตล์ข้อความ Label หัวข้อตัวกรอง (สีขาวสว่าง) */
            .filter-label {
                color: #FFFFFF !important;
                font-size: 1rem !important;
                font-weight: bold !important;
                margin-top: 10px !important;
                margin-bottom: 6px !important;
                display: block !important;
            }

            /* 3. ปรับแต่งปุ่มกด option_menu */
            .nav-link {
                font-size: 14px !important;
                color: #FFFFFF !important;
                text-align: left !important;
                padding: 10px 14px !important;
                margin: 4px 0px !important;
                border-radius: 20px !important;
            }
            .nav-link:hover {
                background-color: rgba(255, 255, 255, 0.1) !important;
            }
            .nav-link-selected {
                background-color: #8ECAE6 !important;
                color: #000000 !important;
                font-weight: bold !important;
            }

            /* 4. ปรับแต่งช่อง Selectbox ให้ขอบสีขาว มนทรงแคปซูล */
            div[data-baseweb="select"] > div {
                background-color: transparent !important;
                border: 2px solid #FFFFFF !important;
                border-radius: 25px !important;
                color: #FFFFFF !important;
            }
            div[data-baseweb="select"] div {
                color: #FFFFFF !important;
            }
            div[data-baseweb="select"] svg { 
                fill: #FFFFFF !important; 
                color: #FFFFFF !important;
            }

            /* 5. ปุ่มล้างตัวกรอง */
            div[data-testid="stSidebar"] button {
                background-color: transparent !important;
                border: 2px solid #FFFFFF !important;
                border-radius: 25px !important;
                color: #FFFFFF !important;
                width: 100% !important;
                font-size: 1rem !important;
                font-weight: bold !important;
                margin-top: 15px !important;
                transition: all 0.3s ease;
            }
            div[data-testid="stSidebar"] button:hover {
                background-color: rgba(255, 255, 255, 0.15) !important;
                border-color: #FFFFFF !important;
            }

            /* 6. กล่องสรุปภาพรวมชุดข้อมูล (ด้านล่างสุด) */
            .summary-card {
                border: 2px solid #FFFFFF;
                border-radius: 35px;
                padding: 18px 10px;
                text-align: center;
                color: #FFFFFF;
                margin-top: 20px;
                background-color: transparent;
            }
            .summary-title {
                font-size: 1.1rem;
                font-weight: bold;
                margin-bottom: 4px;
            }
            .summary-subtitle {
                font-size: 0.95rem;
                margin-bottom: 8px;
            }
            .summary-count {
                font-size: 1.3rem;
                font-weight: bold;
            }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # --- 1. ส่วน Header ด้านบน ---
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; padding: 5px 0px 15px 5px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="48" style="filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.2));">
            <div style="font-size: 18px; font-weight: 700; color: #FFFFFF; line-height: 1.25; letter-spacing: -0.3px;">
                การวิเคราะห์<br>ข้อมูลนักเรียน
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 5px 0 15px 0; border-color: rgba(255,255,255,0.15);'>", unsafe_allow_html=True)

    # --- 2. เมนูนำทางแบบปุ่มกด (Option Menu) ---
    menu = option_menu(
        menu_title=None,
        options=[
            "ภาพรวม",
            "ผลสัมฤทธิ์ทางการเรียน",
            "พฤติกรรมการเรียนรู้",
            "การเข้าเรียน",
            "การมีส่วนร่วมของผู้ปกครอง",
            "การวิเคราะห์ความสัมพันธ์ของข้อมูล",
            "การทำนายผลการเรียนรู้"
        ],
        icons=[
            "house-fill",
            "mortarboard-fill",
            "display",
            "card-checklist",
            "people",
            "search-heart",
            "diagram-3"
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"font-size": "16px"}, 
        }
    )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- 3. ส่วนหัวตัวกรอง ---
    st.markdown("<h3 style='color: white; font-weight: bold; margin-bottom: 15px;'>🔍 กรองข้อมูล</h3>", unsafe_allow_html=True)

    gender_map = {
        "M": "ชาย", 
        "F": "หญิง", 
        "ทั้งหมด": "ทั้งหมด"
    }
    stage_map = {
        "lowerlevel": "ประถมศึกษา", 
        "MiddleSchool": "มัธยมศึกษาตอนต้น", 
        "HighSchool": "มัธยมศึกษาตอนปลาย", 
        "ทั้งหมด": "ทั้งหมด"
    }
    semester_map = {
        "F": "ภาคเรียนที่ 1", 
        "S": "ภาคเรียนที่ 2", 
        "ทั้งหมด": "ทั้งหมด"
        }
    # เตรียม List รายการที่มีตัวเลือก "ทั้งหมด" อยู่ด้วย
    gender_options = ["ทั้งหมด"] + list(df["gender"].unique())
    stage_options = ["ทั้งหมด"] + list(df["StageID"].unique())
    semester_options = ["ทั้งหมด"] + list(df["Semester"].unique())

    # --- 4. ตัวกรองข้อมูล (แสดงข้อความสีขาวด้วย .filter-label + ซ่อน label ดั้งเดิม) ---
    st.markdown("<span class='filter-label'>เพศ</span>", unsafe_allow_html=True)
    selected_gender = st.selectbox("เพศ", options=gender_options, index=0, label_visibility="collapsed",
    format_func=lambda x: gender_map.get(x, x)
    )

    st.markdown("<span class='filter-label'>ระดับชั้น</span>", unsafe_allow_html=True)
    selected_stage = st.selectbox("ระดับชั้น", options=stage_options, index=0, label_visibility="collapsed",
    format_func=lambda x: stage_map.get(x, x)
    )

    st.markdown("<span class='filter-label'>ภาคเรียน</span>", unsafe_allow_html=True)
    selected_semester = st.selectbox("ภาคเรียน", options=semester_options, index=0, label_visibility="collapsed",
    format_func=lambda x: semester_map.get(x, x)
    )

    # --- 5. ปุ่มล้างตัวกรอง ---
    if st.button("🔄  ล้างตัวกรอง", use_container_width=True):
        st.rerun()

    # --- 6. การกรองข้อมูล DataFrame ตามตัวเลือก ---
    filtered_df = df.copy()
    if selected_gender != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["gender"] == selected_gender]
    if selected_stage != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["StageID"] == selected_stage]
    if selected_semester != "ทั้งหมด":
        filtered_df = filtered_df[filtered_df["Semester"] == selected_semester]

    # --- 7. กล่องภาพรวมชุดข้อมูล ---
    st.markdown(
        f"""
        <div class='summary-card'>
            <div class='summary-title'>ภาพรวมชุดข้อมูล</div>
            <div class='summary-subtitle'>จำนวนนักเรียนทั้งหมด</div>
            <div class='summary-count'>{len(filtered_df):,} คน</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# =====================================================
# OVERVIEW (ภาพรวม)
# =====================================================
if menu == "ภาพรวม":

    st.title("📌 Overview")
    st.caption("ภาพรวมทั้งหมดของผู้เรียน")
    st.markdown("---")

    # ==========================================
    # 📊 METRIC CARDS (KPIs ทรงแคปซูล)
    # ==========================================
    total_students = len(filtered_df)
    male_count = len(filtered_df[filtered_df["gender"] == "M"])
    female_count = len(filtered_df[filtered_df["gender"] == "F"])

    above7_count = len(filtered_df[filtered_df["StudentAbsenceDays"] == "Above-7"])
    absence_avg_pct = (above7_count / total_students * 100) if total_students > 0 else 0

    st.markdown("""
        <style>
            .kpi-container {
                display: flex;
                justify-content: space-between;
                gap: 15px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }
            .kpi-card {
                flex: 1;
                min-width: 200px;
                background-color: #FFFFFF;
                border: 2px solid #03254C;
                border-radius: 50px;
                padding: 12px 20px;
                display: flex;
                align-items: center;
                gap: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .kpi-icon-circle {
                width: 55px;
                height: 55px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 26px;
                flex-shrink: 0;
            }
            .kpi-info {
                display: flex;
                flex-direction: column;
            }
            .kpi-title {
                color: #2D3748;
                font-size: 14px;
                font-weight: 600;
                margin: 0;
                line-height: 1.2;
            }
            .kpi-value {
                color: #2B6CB0;
                font-size: 20px;
                font-weight: 700;
                margin-top: 2px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon-circle" style="background-color: #DCFCE7;">👥</div>
                <div class="kpi-info">
                    <div class="kpi-title">นักเรียนทั้งหมด</div>
                    <div class="kpi-value">{total_students:,} คน</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-circle" style="background-color: #BAE6FD;">👦</div>
                <div class="kpi-info">
                    <div class="kpi-title">นักเรียนชาย</div>
                    <div class="kpi-value">{male_count:,} คน</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-circle" style="background-color: #FBCFE8;">👧</div>
                <div class="kpi-info">
                    <div class="kpi-title">นักเรียนหญิง</div>
                    <div class="kpi-value">{female_count:,} คน</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    # ----------------------------------------------------------------
    # 📦 ROW 1: GENDER DISTRIBUTION (แบ่งแถวด้วย Container มีเส้นขอบ)
    # ----------------------------------------------------------------
    with st.container(border=True):
        chart1, chart2 = st.columns(2)
        
        with chart1:
            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700;'>👤 สัดส่วนของผู้เรียน</h3>
                    <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>ผู้เรียนจำแนกตามเพศ</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            gender_count = (
                filtered_df["gender"]
                .value_counts()
                .reset_index()
            )
            gender_count.columns = ["Gender", "Count"]

            gender_count["Gender"] = gender_count["Gender"].replace({
                "M": "เพศชาย",
                "F": "เพศหญิง"
            })

            total_gender_students = gender_count["Count"].sum()

            fig_gender = px.pie(
                gender_count,
                values="Count",
                names="Gender",
                hole=0.55,
                color="Gender",
                color_discrete_map={
                    "เพศชาย": "#4F8EF7",
                    "เพศหญิง": "#FF6B9A"
                }
            )

            fig_gender.update_traces(
                textposition="inside",
                texttemplate="<b>%{percent}</b><br>%{value:,} คน",
                textfont=dict(size=14, color="white", family="Arial"),
                hovertemplate="<b>%{label}</b><br>จำนวน : %{value:,}<br>สัดส่วน : %{percent}<extra></extra>",
                marker=dict(line=dict(color="white", width=3))
            )

            fig_gender.update_layout(
                annotations=[
                    dict(
                        x=0.5, y=0.5,
                        text=f"<span style='color:#64748B; font-size:12px;'>จำนวนนักเรียนทั้งหมด</span><br><br>"
                             f"<b style='font-size:22px; color:#1E293B;'>{total_gender_students:,} คน</b>",
                        showarrow=False
                    )
                ],
                showlegend=True,
                legend=dict(
                    orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=13)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=30),
                height=380
            )

            st.plotly_chart(fig_gender, use_container_width=True, config={"displayModeBar": False})

        with chart2:
            males = len(filtered_df[filtered_df["gender"] == "M"])
            females = len(filtered_df[filtered_df["gender"] == "F"])
            total_curr_gender = males + females
            male_pct = (males / total_curr_gender * 100) if total_curr_gender > 0 else 0
            female_pct = (females / total_curr_gender * 100) if total_curr_gender > 0 else 0

            st.markdown(
                f"""
                <div style='background-color: #DBEAFE; color: #1E40AF; text-align: center; 
                            padding: 6px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; font-size: 14px;'>
                    สรุปภาพรวมเพศ
                </div>
                
                <div style='background-color: #EFF6FF; border-radius: 10px; padding: 10px; margin-bottom: 8px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: #4F8EF7; color: white; width: 38px; height: 38px;
                                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                    font-size: 18px; margin-right: 10px;'>👨</div>
                        <div>
                            <div style='color: #1E293B; font-weight: bold; font-size: 14px;'>เพศชาย</div>
                            <div style='color: #4F8EF7; font-size: 20px; font-weight: bold;'>{males:,}
                              <span style='font-size: 13px; font-weight: normal; color: #475569;'>คน ({male_pct:.1f}%)</span></div>
                        </div>
                    </div>
                </div>

                <div style='background-color: #FFF1F2; border-radius: 10px; padding: 10px; margin-bottom: 8px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: #FF6B9A; color: white; width: 38px; height: 38px;
                                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                    font-size: 18px; margin-right: 10px;'>👩</div>
                        <div>
                            <div style='color: #1E293B; font-weight: bold; font-size: 14px;'>เพศหญิง</div>
                            <div style='color: #FF6B9A; font-size: 20px; font-weight: bold;'>{females:,} 
                                <span style='font-size: 13px; font-weight: normal; color: #475569;'>คน ({female_pct:.1f}%)</span></div>
                        </div>
                    </div>
                </div>

                <div style='background-color: #DCFCE7; border: 1px solid #BBF7D0; border-radius: 10px; padding: 10px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: #064E3B; width: 38px; height: 38px; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; 
                                    margin-right: 10px;'>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M16 21V19C16 17.9391 15.5786 16.9217 14.8284 16.1716C14.0783 15.4214 13.0609 15 12 15C10.9391 15 9.92172 15.4214 9.17157 16.1716C8.42143 16.9217 8 17.9391 8 19V21" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div>
                            <div style='color: #1E293B; font-weight: bold; font-size: 14px;'>รวมทั้งหมด</div>
                            <div style='color: #059669; font-size: 20px; font-weight: bold;'>{total_curr_gender:,} 
                                <span style='font-size: 13px; font-weight: normal; color: #475569;'>คน (100.0%)</span></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ------------------------------------------------------------------------
    # 📦 ROW 2: SEMESTER DISTRIBUTION (แบ่งแถวด้วย Container มีเส้นขอบ)
    # ------------------------------------------------------------------------
    with st.container(border=True):
        chart3, chart4 = st.columns(2)

        with chart3:
            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700;'>📅 จำนวนนักเรียน</h3>
                    <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>จำแนกตามภาคเรียน</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            semester_count = (
                filtered_df["Semester"]
                .value_counts()
                .reset_index()
            )
            semester_count.columns = ["Semester", "Count"]

            semester_count["Semester"] = semester_count["Semester"].replace({
                "F": "ภาคเรียนที่ 1",
                "S": "ภาคเรียนที่ 2"
            })

            total_semester_students = semester_count["Count"].sum()

            semester1 = semester_count.loc[
                semester_count["Semester"] == "ภาคเรียนที่ 1", "Count"
            ].sum()

            semester2 = semester_count.loc[
                semester_count["Semester"] == "ภาคเรียนที่ 2", "Count"
            ].sum()

            percent1 = (semester1 / total_semester_students * 100) if total_semester_students > 0 else 0
            percent2 = (semester2 / total_semester_students * 100) if total_semester_students > 0 else 0

            fig_semester = px.pie(
                semester_count,
                values="Count",
                names="Semester",
                hole=0.68,
                color="Semester",
                color_discrete_map={
                    "ภาคเรียนที่ 1": "#3B82F6",
                    "ภาคเรียนที่ 2": "#8B5CF6"
                }
            )

            fig_semester.update_traces(
                textposition="inside",
                texttemplate="<b>%{percent}</b>",
                textfont=dict(size=16, color="white"),
                marker=dict(line=dict(color="white", width=3)),
                hovertemplate="<b>%{label}</b><br>จำนวน : %{value:,} คน<br>คิดเป็น : %{percent}<extra></extra>"
            )

            fig_semester.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=20),
                legend=dict(
                    orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=13)
                ),
                annotations=[
                    dict(
                        x=0.5, y=0.5,
                        text=f"<span style='color:#64748B; font-size:12px;'>รวมทั้งหมด</span><br><br>"
                             f"<b style='font-size:22px; color:#1E293B;'>{total_semester_students:,} คน</b>",
                        showarrow=False
                    )
                ]
            )

            st.plotly_chart(fig_semester, use_container_width=True, config={"displayModeBar": False})

        with chart4:
            major_semester = "ภาคเรียนที่ 1" if semester1 >= semester2 else "ภาคเรียนที่ 2"
            major_percent = max(percent1, percent2)

            st.markdown(
                f"""
                <div style='display: flex; flex-direction: column; justify-content: center; height: 100%; gap: 8px;'>
                    <div style='background-color: #EFF6FF; border-radius: 10px; padding: 12px;'>
                        <div style='display: flex; align-items: center;'>
                            <div style='background-color: #3B82F6; color: white; width: 38px; height: 38px; 
                                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                        font-size: 18px; margin-right: 10px;'>📘</div>
                            <div>
                                <div style='color: #1E293B; font-weight: bold; font-size: 14px;'>ภาคเรียนที่ 1</div>
                                <div style='color: #2563EB; font-size: 20px; font-weight: bold;'>{semester1:,} <span style='font-size: 13px; 
                                            font-weight: normal; color: #475569;'>คน ({percent1:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    <div style='background-color: #F5F3FF; border-radius: 10px; padding: 12px;'>
                        <div style='display: flex; align-items: center;'>
                            <div style='background-color: #8B5CF6; color: white; width: 38px; height: 38px; border-radius: 50%; 
                                        display: flex; align-items: center; justify-content: center; font-size: 18px; margin-right: 10px;'>📙</div>
                            <div>
                                <div style='color: #1E293B; font-weight: bold; font-size: 14px;'>ภาคเรียนที่ 2</div>
                                <div style='color: #7C3AED; font-size: 20px; font-weight: bold;'>{semester2:,} <span style='font-size: 13px; 
                                            font-weight: normal; color: #475569;'>คน ({percent2:.1f}%)</span></div>
                            </div>
                        </div>
                    </div>
                    <div style='position: relative; top: -15px; margin-bottom: -15px;
                                background-color:#F8FAFC; border:1px solid #BFDBFE; border-left:4px solid #3B82F6;
                                padding:10px 12px; border-radius:8px; font-size:13px; color:#334155;'>
                        <b>💡 ข้อสังเกต:</b> ผู้เรียนส่วนใหญ่อยู่ใน <b>{major_semester}</b> คิดเป็น <b>{major_percent:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    # ------------------------------------------------------------------------
    # 📦 ROW 3: STAGE DISTRIBUTION (แบ่งแถวด้วย Container มีเส้นขอบ)
    # ------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700;'>🏫 นักเรียนตามระดับชั้น</h3>
                <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>แสดงจำนวนผู้เรียนในแต่ละชั้น</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        stage_count = (
            filtered_df["StageID"]
            .value_counts()
            .reset_index()
        )

        stage_count.columns = ["Stage", "Count"]
        stage_count["Stage"] = stage_count["Stage"].replace({
            "lowerlevel": "ประถมศึกษา",
            "MiddleSchool": "มัธยมศึกษาตอนต้น",
            "HighSchool": "มัธยมศึกษาตอนปลาย"
        })
        
        fig_stage = px.bar(
            stage_count,
            x="Stage",
            y="Count",
            color="Stage",
            text="Count",
            color_discrete_map={
                "ประถมศึกษา": "#BFDBFE",
                "มัธยมศึกษาตอนต้น": "#A7F3D0",
                "มัธยมศึกษาตอนปลาย": "#FDE68A"
            },
        )

        max_stage_val = stage_count["Count"].max() if not stage_count.empty else 100
        
        fig_stage.update_layout(
                    height=380,
                    xaxis_title="ระดับการศึกษา",
                    yaxis_title="จำนวนนักเรียน",
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(
                        gridcolor="#E2E8F0", 
                        zeroline=False,
                        range=[0, max_stage_val * 1.25]  # ✨ ขยายเพดานแกน Y เพิ่มขึ้น 25% เพื่อเว้นที่ให้ตัวเลข
                    ),
                    xaxis=dict(linecolor="#CBD5E1"),
                    margin=dict(l=10, r=10, t=40, b=10) # ✨ เพิ่ม margin ขอบบนเป็น 40
        )

        st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})

        if not stage_count.empty:
            most_stage = stage_count.iloc[0]["Stage"]
            most_count = stage_count.iloc[0]["Count"]

            st.markdown(
                f"""
                <div style='position: relative; top: -15px; margin-bottom: -15px;
                            background-color:#F8FAFC; border:1px solid #BFDBFE; border-left:4px solid #3B82F6;
                            padding:10px 12px; border-radius:8px; font-size:13px; color:#334155;'>
                    <b>💡 ข้อสังเกต:</b> ระดับการศึกษาที่มีนักเรียนมากที่สุดคือ <b>{most_stage}</b> จำนวน <b>{most_count:,}</b> คน
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # ------------------------------------------------------------------------
    # 📦 ROW 5: TOP NATIONALITY (แบ่งแถวด้วย Container มีเส้นขอบ)
    # ------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 8px;'>
                    🌍 สัญชาติของผู้เรียนทั้งหมด
                </h3>
                <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>
                    วิเคราะห์สัดส่วนการกระจายตัวของนักเรียนจำแนกตามประเทศสัญชาติทั้งหมด
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        nationality_count = (
            filtered_df["NationalITy"]
            .value_counts()
            .reset_index()
        )
        nationality_count.columns = ["Nationality", "Count"]
        
        nationality_count["Percent"] = (nationality_count["Count"] / total_students * 100) if total_students > 0 else 0

        ctrl_col1, ctrl_col2 = st.columns([2, 3])

        with ctrl_col1:
            sort_order = st.selectbox(
                "↕️ การเรียงลำดับ:",
                options=["มากไปน้อย", "น้อยไปมาก"],
                index=0,
                key="nat_sort"
            )

        with ctrl_col2:
            total_countries = len(nationality_count)
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
                            border-radius: 10px; padding: 6px 12px; text-align: center;'>
                    <span style='color: #1E40AF; font-size: 12px; font-weight: 600;'>ความหลากหลายทางสัญชาติ</span> | 
                    <b style='color: #1D4ED8; font-size: 15px;'>{total_countries:,} ประเทศ</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        is_ascending = True if sort_order == "น้อยไปมาก" else False
        category_order_setting = "total ascending" if not is_ascending else "total descending"

        dynamic_height = max(380, len(nationality_count) * 36)

        fig_nat = px.bar(
            nationality_count,
            x="Count",
            y="Nationality",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Cividis",
            custom_data=["Percent"]
        )

        fig_nat.update_traces(
            texttemplate="<b>%{x:,} คน</b> (%{customdata[0]:.1f}%)",
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            marker=dict(line=dict(width=0), cornerradius=6),
            hovertemplate="<b>🌐 ประเทศ:</b> %{y}<br>" +
                          "<b>👥 จำนวน:</b> %{x:,} คน<br>" +
                          "<b>📊 คิดเป็น:</b> %{customdata[0]:.2f}% ของทั้งหมด<extra></extra>"
        )

        max_count = nationality_count["Count"].max() if not nationality_count.empty else 100

        fig_nat.update_layout(
            template="plotly_white",
            height=dynamic_height,
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=80, t=10, b=10),
            yaxis=dict(
                categoryorder=category_order_setting, 
                title="",
                tickfont=dict(size=13, color="#1E293B", family="Arial"),
                showline=False
            ),
            xaxis=dict(
                title="จำนวนนักเรียน (คน)",
                title_font=dict(size=12, color="#64748B"),
                gridcolor="#F1F5F9",
                zeroline=False,
                range=[0, max_count * 1.28]
            )
        )

        st.plotly_chart(fig_nat, use_container_width=True, config={"displayModeBar": False})

        if not nationality_count.empty:
            top_nat = nationality_count.iloc[0]["Nationality"]
            top_count = nationality_count.iloc[0]["Count"]
            top_pct = nationality_count.iloc[0]["Percent"]

            st.markdown(
                f"""
                <div style='position: relative; top: -15px; margin-bottom: -15px;
                            background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6;
                            border-radius: 8px; padding: 10px 12px; font-size: 13px; color: #334155;'>
                    💡 <b>ข้อสังเกตเชิงสถิติ:</b> ผู้เรียนส่วนใหญ่เป็นสัญชาติ <b>{top_nat}</b> 
                    ครองสัดส่วนสูงสุดถึง <b>{top_count:,} คน ({top_pct:.1f}%)</b> จากทั้งหมด {total_students:,} คน
                </div>
                """,
                unsafe_allow_html=True
            )
    # ------------------------------------------------------------------------
    # 📦 ROW 6: SEARCH & DATA TABLE (แบ่งแถวด้วย Container มีเส้นขอบ)
    # ------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(
        """
        <div style='position: relative; top: -8px; margin-bottom: -5px;
                    background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #3B82F6;
                    border-radius: 8px; padding: 10px 14px; font-size: 16px; font-weight: 600; color: #1E40AF;'>
            🔍 ค้นหาและดูข้อมูลนักเรียน
        </div>
        """,
        unsafe_allow_html=True
        )
    
        # 2. ช่องค้นหาข้อมูล
        keyword = st.text_input("ระบุคำค้นหา (ค้นหาได้ทุกคอลัมน์):", placeholder="พิมพ์คำค้นหา...")

        # 3. การกรองข้อมูล DataFrame
        if keyword:
            display_df = filtered_df[
                filtered_df.astype(str)
                .apply(lambda x: x.str.contains(keyword, case=False))
                .any(axis=1)
            ]
        else:
            display_df = filtered_df

        # 4. แสดงผลตาราง
        st.dataframe(display_df, use_container_width=True, height=350)

        # 5. เตรียมไฟล์และปุ่มดาวน์โหลด
        csv = display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ ดาวน์โหลดรายงาน CSV",
            data=csv,
            file_name="student_dashboard_report.csv",
            mime="text/csv",
            use_container_width=True # (Optional) ปรับให้ปุ่มยาวเต็มความกว้างขอบเพื่อความเรียบร้อย
        )

# =====================================================
# ผลสัมฤทธิ์ทางการเรียน
# =====================================================
elif menu == "ผลสัมฤทธิ์ทางการเรียน":

    # --- CSS ตกแต่งการ์ด KPI และปรับแต่งขอบ Container ของ Streamlit ---
    st.markdown("""
        <style>
            /* สไตล์การ์ด KPI ด้านบน */
            .kpi-card-custom {
                background-color: #FFFFFF;
                border: 1.5px solid #1E3A5F;
                border-radius: 30px;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
            }
            .kpi-icon-bg {
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                flex-shrink: 0;
            }
            .kpi-text-title {
                font-size: 0.85rem;
                color: #1E3A5F;
                font-weight: 600;
                line-height: 1.2;
            }
            .kpi-text-value {
                font-size: 1.1rem;
                font-weight: bold;
                color: #0A2540;
                margin-top: 2px;
            }
            /* บังคับขอบของ st.container ให้เป็นเส้นสีน้ำเงินเข้ม ขอบมน และมีเงา */
            div[data-testid="stVerticalBlockBorderWrapper"] > div {
                border: 1.5px solid #1E3A5F !important;
                border-radius: 20px !important;
                background-color: #FFFFFF !important;
                box-shadow: 0px 2px 5px rgba(0,0,0,0.02) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- ส่วน Header ---
    st.markdown("<h1 style='color: #0A2540; margin-bottom: 0px; font-weight: bold;'>Academic performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569; font-size: 1.1rem; margin-top: 0px; margin-bottom: 20px;'>ผลสัมฤทธิ์ทางการเรียน</p>", unsafe_allow_html=True)

    # --- คำนวณค่าตัวเลขสำหรับ KPI ---
    total_students = len(filtered_df)
    class_counts = filtered_df["Class"].value_counts()
    high_count = class_counts.get("H", 0)
    mid_count = class_counts.get("M", 0)
    low_count = class_counts.get("L", 0)

    # --- 1. KPI Cards ด้านบน (4 การ์ด) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #DCFCE7;'>👨‍👩‍👧‍👦</div>
                <div>
                    <div class='kpi-text-title'>นักเรียนทั้งหมด</div>
                    <div class='kpi-text-value'>{total_students:,} คน</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #E0F2FE;'>🏃</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียน<br>ระดับสูง</div>
                    <div class='kpi-text-value'>{high_count:,} คน</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #FCE7F3;'>🙋‍♀️</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียน<br>ระดับปานกลาง</div>
                    <div class='kpi-text-value'>{mid_count:,} คน</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #FEF3C7;'>🙋‍♂️</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียน<br>ระดับต่ำ</div>
                    <div class='kpi-text-value'>{low_count:,} คน</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- 2. โซนกราฟกลาง (Donut Chart & Bar Chart) ---
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>การกระจายผลสัมฤทธิ์ทางการเรียน</h3>", unsafe_allow_html=True)
            
            fig_donut = px.pie(
                filtered_df,
                names="Class",
                hole=0.55,
                color="Class",
                category_orders={"Class": ["H", "M", "L"]},
                color_discrete_map={"H": "#86EFAC", "M": "#FDE047", "L": "#F87171"}
            )
            
            fig_donut.update_traces(
                textposition="inside",
                textinfo="percent"
            )

            fig_donut.for_each_trace(lambda t: t.update(labels=[
                "ระดับสูง" if label == "H" else "ระดับปานกลาง" if label == "M" else "ระดับต่ำ" for label in t.labels
            ]))

            fig_donut.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[{
                    "text": f"<b>นักเรียนทั้งหมด</b><br>{total_students:,} คน",
                    "x": 0.5, "y": 0.5,
                    "font_size": 13,
                    "font_color": "#1E293B",
                    "showarrow": False
                }]
            )

            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with col_right:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>ผลการเรียนรายภาคการศึกษา</h3>", unsafe_allow_html=True)
            
            if "Semester" in filtered_df.columns:
                sem_df = filtered_df.groupby(["Semester", "Class"]).size().reset_index(name="Count")
                sem_df["Semester_Label"] = sem_df["Semester"].map({"F": "ภาคเรียนที่ 1", "S": "ภาคเรียนที่ 2"}).fillna(sem_df["Semester"])
                
                fig_bar = px.bar(
                    sem_df,
                    x="Semester_Label",
                    y="Count",
                    color="Class",
                    barmode="group",
                    text_auto=True,
                    category_orders={"Class": ["H", "M", "L"]},
                    color_discrete_map={"H": "#86EFAC", "M": "#FDE047", "L": "#F87171"}
                )
                
                # ✨ แก้ไขตรงนี้: ปรับ cliponaxis=False ไม่ให้ตัวเลขโดนขอบตัด
                fig_bar.update_traces(
                    textposition="outside",
                    cliponaxis=False
                )
                
                newnames = {'H': 'ระดับสูง', 'M': 'ระดับปานกลาง', 'L': 'ระดับต่ำ'}
                fig_bar.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))

                fig_bar.update_layout(
                    height=290,  # ขยับความสูงขึ้นเล็กน้อย
                    margin=dict(l=10, r=10, t=40, b=10),  # ✨ แก้ไขตรงนี้: เพิ่ม t=40 เพิ่มพื้นที่ด้านบน
                    xaxis_title="",
                    yaxis_title="",
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9')
                )

                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    # --- 3. โซนล่าง: คะแนนเฉลี่ยตามรายวิชา (Topic) - ข้อความอยู่กึ่งกลางคอลัมน์ ---
    with st.container(border=True):
        st.markdown("<h3 style='font-size: 1.2rem; font-weight: bold; color: #1E293B; margin-bottom: 20px;'>คะแนนเฉลี่ยตามรายวิชา</h3>", unsafe_allow_html=True)
        
        if "Topic" in filtered_df.columns and not filtered_df.empty:
            activity_cols = [c for c in ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"] if c in filtered_df.columns]
            
            if activity_cols:
                topic_df = filtered_df.groupby("Topic")[activity_cols].mean()
                topic_df["Avg_Score"] = topic_df.mean(axis=1)
                topic_df = topic_df.reset_index().sort_values(by="Avg_Score", ascending=False)
                
                html_table = "<table style='width:100%; border-collapse:collapse; color:#1E3A5F; font-size:0.9rem; margin-top:10px;'>"
                html_table += "<thead><tr style='font-weight:600; color:#1E293B; border-bottom: 1px solid #E2E8F0;'><th style='padding:12px 8px; width:20%; text-align:center; vertical-align:middle;'>รายวิชา</th><th style='padding:12px 8px; width:65%; text-align:center; vertical-align:middle;'>คะแนนเฉลี่ย</th><th style='padding:12px 8px; width:15%; text-align:center; vertical-align:middle; white-space:nowrap;'>นักเรียน</th></tr></thead><tbody>"
                
                for _, row in topic_df.iterrows():
                    score_pct = min(max(row['Avg_Score'], 0), 100)
                    student_cnt = len(filtered_df[filtered_df['Topic'] == row['Topic']])
                    
                    html_table += f"<tr><td style='padding:10px 8px; text-align:center; font-weight:500; color:#1E293B; vertical-align:middle;'>{row['Topic']}</td><td style='padding:10px 8px; vertical-align:middle;'><div style='background-color:#003366; border-radius:15px; width:100%; height:18px; overflow:hidden;'><div style='background-color:#00A3E0; width:{score_pct:.0f}%; height:100%; border-radius:15px;'></div></div></td><td style='padding:10px 8px; text-align:center; font-weight:500; color:#1E293B; vertical-align:middle; white-space:nowrap;'>{student_cnt} คน</td></tr>"
                
                html_table += "</tbody></table>"
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.warning("ไม่พบคอลัมน์ข้อมูลกิจกรรมการเรียนรู้")
        else:
            st.info("ไม่พบข้อมูลรายวิชาสำหรับแสดงผล")

# =====================================================
# พฤติกรรมการเรียนรู้
# =====================================================
elif menu == "พฤติกรรมการเรียนรู้":

    st.title("📚 Learning Bahavior")
    st.caption("พฤติกรรมการเรียนรู้")
    st.markdown("---")

    with st.container(border=True):
        fig = px.scatter(
            filtered_df,
            x="raisedhands",
            y="VisITedResources",
            color="Class",
            size="Discussion",
            hover_data=["AnnouncementsView"],
            title="การยกมือตอบคำถาม vs การเข้าดูสื่อการเรียน (ขนาดจุด = การอภิปราย)",
            labels={
                "raisedhands": "Raised Hands (ยกมือตอบคำถาม)",
                "VisITedResources": "Visited Resources (เข้าดูสื่อการเรียน)",
                "Class": "ระดับผลการเรียน"
            },
            category_orders={"Class": ["L", "M", "H"]},
            color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
            template="plotly_white"
        )

        fig.update_layout(
            height=500,
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =====================================================
# การเข้าเรียน
# =====================================================
elif menu == "การเข้าเรียน":

    st.title("🗓️ Attendance")
    st.caption("เปรียบเทียบผลกระทบของวันขาดเรียน (น้อยกว่า 7 วัน vs มากกว่า 7 วัน) ต่อระดับผลการเรียน")
    st.markdown("---")
    with st.container(border=True):
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 12px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 8px;'>
                    🚨 การขาดเรียน
                </h3>
                <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>
                    สัดส่วนการขาดเรียนของนักเรียนจำแนกตามเกณฑ์เสี่ยง (น้อยกว่า 7 วัน vs ตั้งแต่ 7 วันขึ้นไป)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        absence_counts = filtered_df["StudentAbsenceDays"].value_counts().reset_index()
        absence_counts.columns = ["AbsenceCategory", "Count"]
    
        category_map = {
            "Under-7": "ขาดเรียนน้อย ",
            "Above-7": "ขาดเรียนมาก "
        }
        absence_counts["Absence_TH"] = absence_counts["AbsenceCategory"].map(category_map)
            
        total_filtered = len(filtered_df)
        absence_counts["Percent"] = (absence_counts["Count"] / total_filtered * 100) if total_filtered > 0 else 0
    
        under_7_row = absence_counts[absence_counts["AbsenceCategory"] == "Under-7"]
        above_7_row = absence_counts[absence_counts["AbsenceCategory"] == "Above-7"]
    
        count_under7 = under_7_row["Count"].values[0] if not under_7_row.empty else 0
        pct_under7 = under_7_row["Percent"].values[0] if not under_7_row.empty else 0
    
        count_above7 = above_7_row["Count"].values[0] if not above_7_row.empty else 0
        pct_above7 = above_7_row["Percent"].values[0] if not above_7_row.empty else 0
    
        kpi_col1, kpi_col2 = st.columns(2)
    
        with kpi_col1:
            st.markdown(
                f"""
                <div style='background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px; text-align: center;'>
                    <span style='color: #166534; font-size: 13px; font-weight: 600;'>🟢 ขาดเรียนน้อย </span>
                    <div style='color: #15803D; font-size: 22px; font-weight: 800; margin-top: 2px;'>
                        {count_under7:,} คน <span style='font-size: 14px; font-weight: 500;'>({pct_under7:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        with kpi_col2:
            st.markdown(
                f"""
                <div style='background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 10px; padding: 12px; text-align: center;'>
                    <span style='color: #991B1B; font-size: 13px; font-weight: 600;'>🔴 ขาดเรียนมาก </span>
                    <div style='color: #DC2626; font-size: 22px; font-weight: 800; margin-top: 2px;'>
                        {count_above7:,} คน <span style='font-size: 14px; font-weight: 500;'>({pct_above7:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        fig_absence = px.pie(
            absence_counts,
            names="Absence_TH",
            values="Count",
            hole=0.55,
            color="AbsenceCategory",
            color_discrete_map={
                "Under-7": "#22C55E",
                "Above-7": "#EF4444"
            },
            custom_data=["Percent"]
        )
    
        fig_absence.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextfont=dict(size=13, color="#FFFFFF"),
            hovertemplate="<b>เกณฑ์การขาดเรียน:</b> %{label}<br>" +
                          "<b>จำนวน:</b> %{value:,} คน<br>" +
                          "<b>คิดเป็น:</b> %{customdata[0]:.2f}%<extra></extra>",
            marker=dict(line=dict(color="#FFFFFF", width=2))
        )
    
        fig_absence.update_layout(
            template="plotly_white",
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=13, color="#334155")
            ),
            margin=dict(l=10, r=10, t=10, b=30),
            annotations=[{
                "text": f"<b>รวมทั้งหมด</b><br>{total_filtered:,} คน",
                "x": 0.5, "y": 0.5,
                "font_size": 13,
                "font_color": "#1E293B",
                "showarrow": False
            }]
        )
    
        st.plotly_chart(fig_absence, use_container_width=True, config={"displayModeBar": False})
    
        if count_above7 > 0:
            st.markdown(
                f"""
                <div style='position: relative; top: -15px; margin-bottom: -15px;
                            background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B;
                            border-radius: 8px; padding: 10px 12px; font-size: 13px; color: #92400E; '>
                ⚠️ <b>ข้อสังเกตพฤติกรรม:</b> มีนักเรียนอยู่ในกลุ่มเสี่ยงขาดเรียนบ่อย (ตั้งแต่ 7 วันขึ้นไป) 
                    จำนวน <b>{count_above7:,} คน ({pct_above7:.1f}%)</b> ซึ่งอาจส่งผลกระทบโดยตรงต่อผลสัมฤทธิ์ทางการเรียน
                </div>
                """,
                unsafe_allow_html=True
            )
# =====================================================
# การมีส่วนร่วมของผู้ปกครอง
# =====================================================
elif menu == "การมีส่วนร่วมของผู้ปกครอง":

    st.title("👨‍👩‍👧 การวิเคราะห์การมีส่วนร่วมของผู้ปกครอง")
    st.caption("วิเคราะห์สัดส่วนความร่วมมือของผู้ดูแลหลัก ความพึงพอใจต่อโรงเรียน และผลการเรียนของผู้เรียน")
    st.markdown("---")

    with st.container(border=True):
        col_chart, col_info = st.columns([2, 1])

        with col_chart:
            plot_df = filtered_df.copy()
            plot_df["Class"] = plot_df["Class"].astype(str)
            plot_df["Relation"] = plot_df["Relation"].astype(str)
            plot_df["ParentschoolSatisfaction"] = plot_df["ParentschoolSatisfaction"].astype(str)

            fig = px.sunburst(
                plot_df,
                path=["Relation", "ParentschoolSatisfaction", "Class"],
                title="โครงสร้างสายสัมพันธ์ผู้ปกครอง ความพึงพอใจ และผลการเรียน",
                color="Class",
                color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"}
            )

            fig.update_layout(
                margin=dict(t=40, l=0, r=0, b=0),
                height=500,
                font=dict(family="Plus Jakarta Sans, sans-serif")
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_info:
            st.markdown("### 💡 คำแนะนำการอ่านแผนภูมิ")
            st.info("""
    **ลำดับชั้นของแผนภูมิ Sunburst:**

    🟢 **วงในสุด (Inner):**
    ผู้ปกครองผู้ดูแลหลัก (Father / Mother)

    🟡 **วงกลาง (Middle):**
    ความพึงพอใจต่อโรงเรียน (Good / Bad)

    🔴 **วงนอกสุด (Outer):**
    ระดับผลการเรียนของนักเรียน (H / M / L)

    📌 *สามารถคลิกที่วงกลมแต่ละส่วนเพื่อเจาะลึกดูสัดส่วนย่อยได้*
    """)

# =====================================================
# การวิเคราะห์ความสัมพันธ์ของข้อมูล
# =====================================================
elif menu == "การวิเคราะห์ความสัมพันธ์ของข้อมูล":

    st.title("🔗 Relationship Analysis")
    st.caption("เมทริกซ์ความสัมพันธ์เชิงตัวเลขระหว่างพฤติกรรมการเรียนรู้ด้านต่าง ๆ ของผู้เรียน")
    st.markdown("---")

    with st.container(border=True):
        corr_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
        corr_matrix = filtered_df[corr_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="เมทริกซ์ความสัมพันธ์ของพฤติกรรมการเรียน (Correlation Heatmap)",
            labels=dict(color="ค่าความสัมพันธ์"),
            x=["การยกมือ", "การเข้าดูสื่อ", "การดูประกาศ", "การอภิปราย"],
            y=["การยกมือ", "การเข้าดูสื่อ", "การดูประกาศ", "การอภิปราย"]
        )

        fig.update_layout(
            height=450,
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("🔍 คลิกเพื่อดูแผนภูมิ Scatter Matrix เพิ่มเติม"):
            fig_matrix = px.scatter_matrix(
                filtered_df[corr_cols],
                dimensions=corr_cols,
                color=filtered_df["Class"],
                color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
                template="plotly_white"
            )
            fig_matrix.update_traces(diagonal_visible=False)
            st.plotly_chart(fig_matrix, use_container_width=True)

# =====================================================
# การทำนายผลการเรียนรู้
# =====================================================
elif menu == "การทำนายผลการเรียนรู้":

    st.title("🤖 โมเดลทำนายผลการเรียนรู้ (Random Forest)")
    st.caption("ประเมินประสิทธิภาพโมเดล และทำนายระดับผลการเรียนสำหรับผู้เรียนรายบุคคล")
    st.markdown("---")

    # =========================================================
    # 🛠️ 1. เตรียมข้อมูล และ Encode แบบปลอดภัย
    # =========================================================
    model_df = df.copy()
    encoders = {}
    
    # ลบคอลัมน์ที่ไม่เกี่ยวหรือเป็น ID ออก
    drop_cols = ["Class", "Student_ID", "Name", "StudentName", "id"]
    feature_cols = [c for c in model_df.columns if c not in drop_cols]

    # Encode เฉพาะคอลัมน์ที่เป็นข้อความ
    for col in model_df.columns:
        if model_df[col].dtype == "object" or model_df[col].dtype.name == "category":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str).fillna("Unknown"))
            encoders[col] = le

    X = model_df[feature_cols]
    y = model_df["Class"]

    # บังคับให้เป็นตัวเลข และเติมค่าว่างด้วย 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Split ข้อมูล
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    # =========================================================
    # 📊 2. สร้าง Tabs
    # =========================================================
    tab1, tab2 = st.tabs(["📊 ประสิทธิภาพโมเดล (Model Performance)", "🔮 ทำนายผลการเรียนผู้เรียนใหม่ (Live Prediction)"])

    with tab1:
        with st.container(border=True):
            col_acc, col_cm = st.columns([1, 2])
        
            with col_acc:
                st.metric("🎯 ความแม่นยำโมเดล (Accuracy)", f"{accuracy:.2%}")
                st.info("""
                **คำอธิบายโมเดล:**
                โมเดลประมวลผลข้อมูลจากปัจจัย 16 ตัวแปร เพื่อจำแนกผลการเรียนออกเป็น:
                * **H (High):** ผลการเรียนสูง
                * **M (Medium):** ผลการเรียนปานกลาง
                * **L (Low):** ผลการเรียนต่ำ
                """)

            with col_cm:
                cm = confusion_matrix(y_test, pred)
                class_names = encoders["Class"].classes_ if "Class" in encoders else ["L", "M", "H"]
            
                fig_cm = px.imshow(
                    cm,
                    x=class_names,
                    y=class_names,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Confusion Matrix Heatmap",
                    labels=dict(x="ผลคาดการณ์ (Predicted)", y="ผลจริง (Actual)")
                )
                fig_cm.update_layout(
                    height=320,
                    font=dict(family="Plus Jakarta Sans, sans-serif"),
                    template="plotly_white"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with st.container(border=True):
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig_imp = px.bar(
                importance.tail(10),
                x="Importance",
                y="Feature",
                orientation="h",
                title="10 ปัจจัยสำคัญที่สุดที่มีผลต่อเกรดของนักเรียน (Top Feature Importance)",
                color="Importance",
                color_continuous_scale="Blugrn",
                template="plotly_white"
            )
            fig_imp.update_layout(
                height=380,
                font=dict(family="Plus Jakarta Sans, sans-serif")
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        with st.container(border=True):
            st.markdown("### 📝 ป้อนข้อมูลพฤติกรรมของนักเรียนใหม่")
            st.caption("กรอกข้อมูลการเรียนและการมีส่วนร่วมด้านล่างเพื่อประเมินเกรดคาดการณ์")

            with st.form("prediction_form"):
                col_f1, col_f2, col_f3 = st.columns(3)

                with col_f1:
                    st.markdown("**👤 ข้อมูลทั่วไป**")
                    input_gender = st.selectbox("Gender (เพศ)", df["gender"].unique() if "gender" in df else ["M", "F"])
                    input_national = st.selectbox("NationalITy (สัญชาติ)", df["NationalITy"].unique() if "NationalITy" in df else ["KW"])
                    input_place = st.selectbox("PlaceofBirth (สถานที่เกิด)", df["PlaceofBirth"].unique() if "PlaceofBirth" in df else ["KuwaIT"])
                    input_stage = st.selectbox("StageID (ระดับชั้น)", df["StageID"].unique() if "StageID" in df else ["lowerlevel"])
                    input_grade = st.selectbox("GradeID (ชั้นปี)", df["GradeID"].unique() if "GradeID" in df else ["G-04"])
                    input_section = st.selectbox("SectionID (ห้องเรียน)", df["SectionID"].unique() if "SectionID" in df else ["A"])

                with col_f2:
                    st.markdown("**📚 พฤติกรรมการเรียน**")
                    input_hands = st.number_input("Raised Hands (จำนวนครั้งที่ยกมือ)", min_value=0, max_value=100, value=25)
                    input_resources = st.number_input("Visited Resources (การเข้าดูสื่อ)", min_value=0, max_value=100, value=50)
                    input_announcements = st.number_input("Announcements View (การดูประกาศ)", min_value=0, max_value=100, value=20)
                    input_discussion = st.number_input("Discussion (การพูดคุยแลกเปลี่ยน)", min_value=0, max_value=100, value=15)
                    input_topic = st.selectbox("Topic (วิชาเรียน)", df["Topic"].unique() if "Topic" in df else ["IT"])

                with col_f3:
                    st.markdown("**👨‍👩‍👧 ข้อมูลการขาดเรียน & ผู้ปกครอง**")
                    input_semester = st.selectbox("Semester (ภาคเรียน)", df["Semester"].unique() if "Semester" in df else ["F"])
                    input_relation = st.selectbox("Relation (ผู้ดูแลหลัก)", df["Relation"].unique() if "Relation" in df else ["Father"])
                    input_absence = st.selectbox("StudentAbsenceDays (วันขาดเรียน)", df["StudentAbsenceDays"].unique() if "StudentAbsenceDays" in df else ["Under-7"])
                    input_parent_sat = st.selectbox("ParentschoolSatisfaction (ความพึงพอใจผู้ปกครอง)", df["ParentschoolSatisfaction"].unique() if "ParentschoolSatisfaction" in df else ["Good"])
                    input_parent_survey = st.selectbox("ParentAnsweringSurvey (การตอบแบบสอบถาม)", df["ParentAnsweringSurvey"].unique() if "ParentAnsweringSurvey" in df else ["Yes"])

                submit_button = st.form_submit_button("🔮 ประมวลผลการทำนาย (Predict Performance)", use_container_width=True)

            if submit_button:
                input_dict = {
                    "gender": input_gender,
                    "NationalITy": input_national,
                    "PlaceofBirth": input_place,
                    "StageID": input_stage,
                    "GradeID": input_grade,
                    "SectionID": input_section,
                    "Topic": input_topic,
                    "Semester": input_semester,
                    "Relation": input_relation,
                    "raisedhands": input_hands,
                    "VisITedResources": input_resources,
                    "AnnouncementsView": input_announcements,
                    "Discussion": input_discussion,
                    "ParentAnsweringSurvey": input_parent_survey,
                    "ParentschoolSatisfaction": input_parent_sat,
                    "StudentAbsenceDays": input_absence
                }

                input_data = pd.DataFrame([input_dict])
                
                # กรองเอาเฉพาะคอลัมน์ที่มีใน feature_cols
                for col in feature_cols:
                    if col not in input_data.columns:
                        input_data[col] = 0

                input_data = input_data[feature_cols]

                # Encode ข้อมูลอินพุตใหม่
                for col in input_data.columns:
                    if col in encoders:
                        try:
                            input_data[col] = encoders[col].transform(input_data[col].astype(str))
                        except Exception:
                            input_data[col] = 0

                input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0)

                prediction_encoded = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                if "Class" in encoders:
                    predicted_class = encoders["Class"].inverse_transform([prediction_encoded])[0]
                else:
                    predicted_class = str(prediction_encoded)

                st.markdown("---")
                st.markdown("### 🎯 ผลการทำนาย (Prediction Result)")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    if predicted_class in ["H", 2]:
                        st.success("🎉 **ระดับผลการเรียนคาดการณ์: High (H)**")
                    elif predicted_class in ["M", 1]:
                        st.warning("⚡ **ระดับผลการเรียนคาดการณ์: Medium (M)**")
                    else:
                        st.error("🚨 **ระดับผลการเรียนคาดการณ์: Low (L)**")

                with res_col2:
                    class_labels = encoders["Class"].classes_ if "Class" in encoders else ["L", "M", "H"]
                    prob_df = pd.DataFrame({
                        "Class": class_labels,
                        "Probability": probabilities
                    })
                    fig_prob = px.bar(
                        prob_df,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        title="ความน่าจะเป็นของผลทำนาย (Confidence Rate)",
                        text_auto=".1%",
                        color="Class",
                        color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
                        template="plotly_white"
                    )
                    fig_prob.update_layout(height=180, showlegend=False)
                    st.plotly_chart(fig_prob, use_container_width=True)