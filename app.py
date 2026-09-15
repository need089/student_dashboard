import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


from plotly.subplots import make_subplots
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
            "การมีส่วนร่วมของผู้ปกครอง",
            "การวิเคราะห์ความสัมพันธ์ของตัวแปร",
            "การทำนายผลการเรียนของนักเรียน"
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
        "lowerlevel": "ประถม", 
        "MiddleSchool": "มัธยมต้น", 
        "HighSchool": "มัธยมปลาย", 
        "ทั้งหมด": "ทั้งหมด"
    }
    grade_map = {
        "G-01": "ป1",
        "G-02": "ป.2",
        "G-03": "ป.3",
        "G-04": "ป.4",
        "G-05": "ป.5",
        "G-06": "ป.6",
        "G-07": "ม.1",
        "G-08": "ม.2",
        "G-09": "ม.3",
        "G-10": "ม.4",
        "G-11": "ม.5",
        "G-12": "ม.6",
        "ทั้งหมด": "ทั้งหมด"
    }
    semester_map = {
        "F": "ภาคการศึกษาที่ 1", 
        "S": "ภาคการศึกษาที่ 2", 
        "ทั้งหมด": "ทั้งหมด"
        }
    # เตรียม List รายการที่มีตัวเลือก "ทั้งหมด" อยู่ด้วย
    gender_options = ["ทั้งหมด"] + list(df["gender"].unique())
    stage_options = ["ทั้งหมด"] + list(df["StageID"].unique())
    grade_options = [
            "ทั้งหมด",
            "G-01",
            "G-02",
            "G-03",
            "G-04",
            "G-05",
            "G-06",
            "G-07",
            "G-08",
            "G-09",
            "G-10",
            "G-11",
            "G-12"
    ]
    semester_options = ["ทั้งหมด"] + list(df["Semester"].unique())

    # --- 4. ตัวกรองข้อมูล (แสดงข้อความสีขาวด้วย .filter-label + ซ่อน label ดั้งเดิม) ---
    st.markdown("<span class='filter-label'>เพศ</span>", unsafe_allow_html=True)
    selected_gender = st.selectbox("เพศ", options=gender_options, index=0, label_visibility="collapsed",
    format_func=lambda x: gender_map.get(x, x)
    )

    st.markdown("<span class='filter-label'>ช่วงชั้น</span>", unsafe_allow_html=True)
    selected_stage = st.selectbox("ช่วงชั้น", options=stage_options, index=0, label_visibility="collapsed",
    format_func=lambda x: stage_map.get(x, x)
    )

    st.markdown("<span class='filter-label'>ระดับชั้น</span>", unsafe_allow_html=True)
    selected_grade = st.selectbox("ระดับชั้น", options=grade_options, index=0, label_visibility="collapsed",
    format_func=lambda x: grade_map.get(x, x)
    )

    st.markdown("<span class='filter-label'>ภาคการศึกษา</span>", unsafe_allow_html=True)
    selected_semester = st.selectbox("ภาคการศึกษา", options=semester_options, index=0, label_visibility="collapsed",
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
    if selected_grade != "ทั้งหมด":
         filtered_df = filtered_df[filtered_df["GradeID"] == selected_grade]
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
    st.markdown(
        """
        <div style="
            background-color: #F0F9FF;
            border: 1px solid #BAE6FD;
            border-left: 5px solid #0284C7;
            border-radius: 8px;
            padding: 15px 18px;
            margin-top: 10px;
            margin-bottom: 20px;
            color: #0C4A6E;
            line-height: 1.6;
            font-size: 14.5px;
        ">
            🎓 แดชบอร์ดนี้เป็นระบบสารสนเทศเพื่อวิเคราะห์และติดตาม <strong>ผลสัมฤทธิ์ทางการเรียนและพฤติกรรมการเรียนรู้ของผู้เรียน (xAPI Education Data Analysis)</strong><br> 
            จัดทำขึ้นเพื่อให้ผู้สอน ผู้บริหาร และผู้ที่เกี่ยวข้องสามารถมองเห็นภาพรวมสถิติต่าง ๆ ของผู้เรียน ได้แก่ สัดส่วนเพศ สัญชาติ ระดับชั้นการศึกษา 
            ตลอดจนการเชื่อมโยง<br>ข้อมูลพฤติกรรมในห้องเรียน การเข้าเรียน และการมีส่วนร่วมของผู้ปกครอง เพื่อนำไปสู่การวางแผนพัฒนาและช่วยเหลือผู้เรียนได้อย่างมีประสิทธิภาพ
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ==========================================
    # 📊 METRIC CARDS (KPIs ทรงแคปซูล)
    # ==========================================
    total_students = len(filtered_df)
    male_count = len(filtered_df[filtered_df["gender"] == "M"])
    female_count = len(filtered_df[filtered_df["gender"] == "F"])

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
    # 📦 ROW 1: GENDER DISTRIBUTION / NATIONALITY
    # ----------------------------------------------------------------

    chart1, chart2 = st.columns(2, gap="medium")
    # ================================================================
    # 📦 กรอบที่ 1 : สัดส่วนของผู้เรียน
    # ================================================================
    with chart1:
        with st.container(border=True):
            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='color: #1E293B;margin: 0;font-size: 20px;font-weight: 700;'>
                    👤 สัดส่วนของผู้เรียน</h3>
                    <p style='color: #64748B;font-size: 14px;margin: 2px 0 0 0;'>
                        ผู้เรียนจำแนกตามเพศ</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ========================================================
            # 📊 นับจำนวนผู้เรียนตามเพศ
            # ========================================================

            gender_count = (
                filtered_df["gender"]
                .value_counts()
                .reset_index()
            )
            gender_count.columns = [
                "Gender",
                "Count"
            ]
            gender_count["Gender"] = gender_count["Gender"].replace({
                "M": "เพศชาย",
                "F": "เพศหญิง"
            })
            total_gender_students = gender_count["Count"].sum()
            # ========================================================
            # 🥧 สร้างกราฟวงกลม
            # ========================================================
            fig_gender = px.pie(
                gender_count,
                values="Count",
                names="Gender",
                hole=0.55,
                color="Gender",
                color_discrete_map={
                    "เพศชาย": "#4A90E2",
                    "เพศหญิง": "#E22E82"
                }
            )
            # ========================================================
            # 🎨 ปรับแต่งกราฟวงกลม
            # ========================================================
            fig_gender.update_traces(
                rotation=180,
                direction="clockwise",
                textposition="inside",
                texttemplate="<b>%{percent}</b><br>%{value:,} คน",
                textfont=dict(
                    size=14,
                    color="white",
                    family="Arial"
                ),
                hovertemplate=
                    "<b>%{label}</b><br>" +
                    "จำนวน : %{value:,}<br>" +
                    "สัดส่วน : %{percent}" +
                    "<extra></extra>",
                    marker=dict(
                        line=dict(
                        color="white",
                        width=3
                    )
                )
            )
            # ========================================================
            # ⚙️ ตั้งค่ารูปแบบกราฟ
            # ========================================================
            fig_gender.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=0.5,
                        text=
                            f"<span style='color:#64748B; font-size:12px;'>"
                            f"จำนวนนักเรียนทั้งหมด"
                            f"</span><br><br>"
                            f"<b style='font-size:22px; color:#1E293B;'>"
                            f"{total_gender_students:,} คน"
                            f"</b>",
                        showarrow=False
                    )
                ],
                showlegend=True,
                legend=dict(
                    orientation="h",
                    y=-0.05,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=13)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=30
                ),
                height=525
            )
            # ========================================================
            # 📈 แสดงกราฟ
            # ========================================================
            st.plotly_chart(
                fig_gender,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
            # ========================================================
            # 📊 คำนวณข้อมูลสำหรับข้อสังเกตเชิงสถิติ
            # ========================================================
            male_count = gender_count.loc[
                gender_count["Gender"] == "เพศชาย",
                "Count"
            ].sum()
            female_count = gender_count.loc[
                gender_count["Gender"] == "เพศหญิง",
                "Count"
            ].sum()
            male_percent = (
                male_count / total_gender_students * 100
                if total_gender_students > 0
                else 0
            )
            female_percent = (
                female_count / total_gender_students * 100
                if total_gender_students > 0
                else 0
            )
            # ========================================================
            # 🏆 หาเพศที่มีจำนวนผู้เรียนมากที่สุด
            # ========================================================
            if male_count >= female_count:
                main_gender = "เพศชาย"
                main_gender_percent = male_percent
            else:
                main_gender = "เพศหญิง"
                main_gender_percent = female_percent
            # ========================================================
            # 💡 ข้อสังเกตเชิงสถิติ
            # ========================================================
            if total_gender_students > 0:
                st.markdown(
                    f"""
                    <div style="
                        position: relative;
                        top: -15px;
                        margin-bottom: -15px;
                        background-color: #F8FAFC;
                        border: 1px solid #E2E8F0;
                        border-left: 4px solid #3B82F6;
                        border-radius: 8px;
                        padding: 10px 12px;
                        font-size: 13px;
                        color: #334155;
                        line-height: 1.6;
                    ">
                        💡 <strong>ข้อสังเกตเชิงสถิติ:</strong>
                        • นักเรียนเพศชายมีจำนวน {male_count} คน คิดเป็น {male_percent:.1f}%<br>
                        • นักเรียนเพศหญิงมีจำนวน {female_count} คน คิดเป็น {female_percent:.1f}%<br>
                        • โครงสร้างข้อมูลมีสัดส่วนเพศชายมากกว่าเพศหญิงประมาณ {(male_count/female_count):.1f} เท่า<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ================================================================
    # 📦 กรอบที่ 2 : สัญชาติของผู้เรียนทั้งหมด
    # ================================================================
    with chart2:
        with st.container(border=True):

            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 8px;'>
                        🌍 สัญชาติของผู้เรียนทั้งหมด
                    </h3>
                    <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>
                        วิเคราะห์สัดส่วนการกระจายตัวของนักเรียนจำแนกตามประเทศสัญชาติทั้งหมด</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ========================================================
            # นับจำนวนผู้เรียนแต่ละสัญชาติ
            # ========================================================
            nationality_count = (
                filtered_df["NationalITy"]
                .value_counts()
                .reset_index()
            )

            nationality_count.columns = [
                "Nationality",
                "Count"
            ]
            nationality_labels = {
                "KW": "คูเวต",
                "lebanon": "เลบานอน",
                "Egypt": "อียิปต์",
                "SaudiArabia": "ซาอุดีอาระเบีย",
                "USA": "สหรัฐอเมริกา",
                "Jordan": "จอร์แดน",
                "Iran": "อิหร่าน",
                "Tunis": "ตูนิเซีย",
                "Morocco": "โมร็อกโก",
                "Iraq": "อิรัก",
                "Syria": "ซีเรีย",
                "Palestine": "ปาเลสไตน์",
                "Lybia": "ลิเบีย",
                "venzuela":"เวเนซุเอลา"
            }
            nationality_count["Nationality"] = (
                nationality_count["Nationality"]
                .map(nationality_labels)
                .fillna(nationality_count["Nationality"])
            )

            # ========================================================
            # คำนวณเปอร์เซ็นต์
            # ========================================================
            nationality_count["Percent"] = (
                nationality_count["Count"] / total_students * 100
                if total_students > 0
                else 0
            )
            # ========================================================
            # เรียงจากมากไปน้อย
            # ========================================================
            nationality_count = (
                nationality_count
                .sort_values(
                    by="Count",
                    ascending=True
                )
                .reset_index(drop=True)
            )
            # ========================================================
            # จำนวนประเทศทั้งหมด
            # ========================================================
            total_countries = len(nationality_count)

            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
                            border-radius: 10px;
                            padding: 6px 12px;
                            text-align: center;
                            margin-bottom: 10px;'>
                    <span style='color: #1E40AF;
                                 font-size: 12px;
                                font-weight: 600;'>
                            ความหลากหลายทางสัญชาติ</span>|
                    <b style='color: #1D4ED8;
                              font-size: 15px;'>
                        {total_countries:,} ประเทศ</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ========================================================
            # กำหนดความสูงของกราฟ
            # ========================================================
            dynamic_height = 500
            # ========================================================
            # สร้างกราฟแท่งแนวนอน
            # ========================================================
            fig_nat = px.bar(
                nationality_count,
                x="Count",
                y="Nationality",
                orientation="h",
                text="Count",
                color="Count",
                color_continuous_scale=[
                    [0.00, "#B8B3D0"],
                    [0.25, "#9A95BB"],
                    [0.50, "#7A75A2"],
                    [0.75, "#59558A"],
                    [1.00, "#35305F"]
                ],
                custom_data=["Percent"]

            )
            # ========================================================
            # ปรับรูปแบบแท่งกราฟ
            # ========================================================
            fig_nat.update_traces(
                texttemplate="<b>%{x:,} คน</b> (%{customdata[0]:.1f}%)",
                textposition="outside",
                cliponaxis=False,
                textfont=dict(
                    size=12,
                    color="#334155"
                ),

                marker=dict(
                    line=dict(width=0),
                    cornerradius=6
                ),

                hovertemplate=
                    "<b>🌐 ประเทศ:</b> %{y}<br>" +
                    "<b>👥 จำนวน:</b> %{x:,} คน<br>" +
                    "<b>📊 คิดเป็น:</b> %{customdata[0]:.2f}% ของทั้งหมด" +
                    "<extra></extra>"
            )
            # ========================================================
            # หาค่าสูงสุด
            # ========================================================
            max_count = (
                nationality_count["Count"].max()
                if not nationality_count.empty
                else 100
            )
            # ========================================================
            # ตั้งค่ารูปแบบกราฟ
            # ========================================================
            fig_nat.update_layout(
                template="plotly_white",
                height=dynamic_height,
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=120,
                    t=10,
                    b=10
                ),
                # ----------------------------------------------------
                # มากที่สุดอยู่ด้านบน
                # ----------------------------------------------------
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=nationality_count["Nationality"].tolist()[::-1],
                    title="",
                    tickfont=dict(
                        size=13,
                        color="#1E293B",
                        family="Arial"
                    ),
                    showline=False
                ),
                # ----------------------------------------------------
                # แกน X
                # ----------------------------------------------------
                xaxis=dict(
                    title="จำนวนนักเรียน (คน)",
                    title_font=dict(
                        size=12,
                        color="#64748B"
                    ),
                    gridcolor="#F1F5F9",
                    zeroline=False,
                    range=[
                        0,
                        max_count * 1.40
                    ]
                )
            )   
            # ========================================================
            # แสดงกราฟ
            # ========================================================
            st.plotly_chart(
                fig_nat,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
            # ========================================================
            # ข้อสังเกตเชิงสถิติ
            # ========================================================
            if not nationality_count.empty:
                top_row = nationality_count.loc[nationality_count["Count"].idxmax()]
                top_nat = top_row["Nationality"]
                top_count = top_row["Count"]
                top_pct = top_row["Percent"]

                st.markdown(
                    f"""
                    <div style="
                        position: relative;
                        top: -15px;
                        margin-bottom: -15px;
                        background-color: #FAF5FF;
                        border: 1px solid #E9D5FF;
                        border-left: 4px solid #9333A8;
                        border-radius: 8px;
                        padding: 10px 12px;
                        font-size: 13px;
                        color: #334155;
                        line-height: 1.6;
                    ">
                    💡 <strong>ข้อสังเกตเชิงสถิติ:</strong>
                        ผู้เรียนส่วนใหญ่เป็นสัญชาติ
                        <strong>{top_nat}</strong>
                        ครองสัดส่วนสูงสุดถึง
                        <strong>{top_count:,} คน ({top_pct:.1f}%)</strong>
                        จากทั้งหมด {total_students:,} คน
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ------------------------------------------------------------------------
    # 📦 ROW 2: STUDENT DISTRIBUTION BY STAGE / GRADE
    # ------------------------------------------------------------------------
    chart3, chart4 = st.columns(2, gap="medium")

    # ========================================================================
    # 📦 กรอบที่ 1 : จำนวนนักเรียนตามช่วงชั้น
    # ========================================================================
    with chart3:
        with st.container(border=True):
            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='
                        color: #1E293B;
                        margin: 0;
                        font-size: 20px;
                        font-weight: 700;
                    '>
                        🏫 จำนวนนักเรียนตามช่วงชั้น</h3>
                    <p style='
                        color: #64748B;
                        font-size: 14px;
                        margin: 2px 0 0 0;
                    '>
                        จำแนกตามช่วงชั้นการศึกษา
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ================================================================
            # 📊 นับจำนวนตามช่วงชั้น
            # ================================================================
            stage_count = (
                filtered_df["StageID"]
                .value_counts()
                .reset_index()
            )
            stage_count.columns = [
                "Stage",
                "Count"
            ]
            # ================================================================
            # 🔄 เปลี่ยนชื่อช่วงชั้น
            # ================================================================
            stage_count["Stage"] = stage_count["Stage"].replace({
                "lowerlevel": "ประถม",
                "MiddleSchool": "มัธยมต้น",
                "HighSchool": "มัธยมปลาย"
            })
            # ================================================================
            # 🔢 เรียงช่วงชั้นตามจำนวนจริงจากน้อย → มาก
            # ================================================================
            stage_count = stage_count.sort_values(
                "Count",
                ascending=True
                ).reset_index(drop=True)

            stage_order = stage_count["Stage"].tolist()
            # ================================================================
            # 📊 จำนวนผู้เรียนทั้งหมด
            # ================================================================
            total_stage_students = stage_count["Count"].sum()

            # ================================================================
            # 📊 คำนวณเปอร์เซ็นต์
            # ================================================================
            stage_count["Percent"] = (
                stage_count["Count"]
                / total_stage_students
                * 100
                if total_stage_students > 0
                else 0
            )
            # ================================================================
            # 📈 สร้างกราฟ
            # ================================================================
            fig_stage = px.bar(
                stage_count,
                x="Stage",
                y="Count",
                text="Count",
                color="Count",
                color_continuous_scale=[
                    [0.0, "#FFEDD5"],   # อ่อนสุด 
                    [0.5, "#FB923C"],   # กลาง
                    [1.0, "#9A3412"]    # เข้มสุด
                ],
                custom_data=["Percent"],
                category_orders={
                    "Stage": stage_order
                }
            )
            # ================================================================
            # 🎨 ปรับกราฟ
            # ================================================================
            fig_stage.update_traces(
                texttemplate=
                    "<b>%{y:,} คน</b><br>"
                    "(%{customdata[0]:.1f}%)",
                textposition="outside",
                textfont=dict(
                    size=12,
                    color="#334155"
                ),
                marker=dict(
                    line=dict(width=0),
                    cornerradius=6
                ),
                hovertemplate=
                    "<b>🏫 ช่วงชั้น:</b> %{x}<br>" +
                    "<b>👥 จำนวน:</b> %{y:,} คน<br>" +
                    "<b>📊 คิดเป็น:</b> " +
                    "%{customdata[0]:.2f}% ของทั้งหมด" +
                    "<extra></extra>"
            )
            max_stage_count = (
                stage_count["Count"].max()
                if not stage_count.empty
                else 100
            )
            # ================================================================
            # ⚙️ ตั้งค่ากราฟ
            # ================================================================
            fig_stage.update_layout(
                height=400,
                template="plotly_white",
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=20,
                    r=50,
                    t=20,
                    b=70
                ),
                xaxis=dict(
                    title="ช่วงชั้น",
                    title_font=dict(
                        size=13,
                        color="#64748B"
                    ),
                    tickfont=dict(
                        size=12,
                        color="#1E293B"
                    ),
                    showgrid=False
                ),
                yaxis=dict(
                    title="จำนวนนักเรียน (คน)",
                    title_font=dict(
                        size=13,
                        color="#64748B"
                    ),
                    gridcolor="#F1F5F9",
                    zeroline=False,
                    range=[
                        0,
                        max_stage_count * 1.30
                    ]
                )
            )
            st.plotly_chart(
                fig_stage,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
            # ================================================================
            # 💡 ข้อสังเกต
            # ================================================================
            if not stage_count.empty:
                top_stage_row = stage_count.loc[
                    stage_count["Count"].idxmax()
                ]
                top_stage = top_stage_row["Stage"]
                top_stage_count = top_stage_row["Count"]
                top_stage_percent = top_stage_row["Percent"]
            else:
                top_stage = "-"
                top_stage_count = 0
                top_stage_percent = 0

            st.markdown(
                f"""
                <div style="
                    position: relative;
                    margin-top: -10px;
                    margin-bottom: 15px;
                    background-color: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-left: 4px solid #3B82F6;
                    border-radius: 8px;
                    padding: 10px 12px;
                    font-size: 13px;
                    color: #334155;
                    line-height: 1.6;
                ">
                    💡 <strong>ข้อสังเกตเชิงสถิติ:</strong>
                    ผู้เรียนส่วนใหญ่อยู่ใน<strong>{top_stage}</strong>
                    จำนวน<strong> {top_stage_count:,} คน ({top_stage_percent:.1f}%)</strong>
                    จากผู้เรียนทั้งหมด<strong> {total_stage_students:,} คน</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ========================================================================
    # 📦 กรอบที่ 2 : จำนวนนักเรียนตามระดับชั้น
    # ========================================================================
    with chart4:
        with st.container(border=True):
            st.markdown(
                """
                <div style='text-align: left; margin-bottom: 10px;'>
                    <h3 style='
                        color: #1E293B;
                        margin: 0;
                        font-size: 20px;
                        font-weight: 700;
                    '>
                        🎓 จำนวนนักเรียนตามระดับชั้น</h3>
                    <p style='
                        color: #64748B;
                        font-size: 14px;
                        margin: 2px 0 0 0;
                    '>
                        วิเคราะห์จำนวนผู้เรียนจำแนกตามระดับชั้น</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ================================================================
            # 📊 นับจำนวนตามระดับชั้น
            # ================================================================
            grade_count = (
                filtered_df["GradeID"]
                .value_counts()
                .reset_index()
            )
            grade_count.columns = [
                "Grade",
                "Count"
            ]
            # ================================================================
            # 🔽 เรียงระดับชั้นตามจำนวนจริงจากน้อย → มาก
            # ================================================================
            grade_count = (
                grade_count
                .sort_values(
                    "Count",
                    ascending=True
                )
                .reset_index(drop=True)
            )
            # ================================================================
            # 📊 จำนวนทั้งหมด
            # ================================================================
            total_grade_students = grade_count["Count"].sum()

            grade_count["Percent"] = (
                grade_count["Count"]
                / total_grade_students
                * 100
                if total_grade_students > 0
                else 0
            )
            # ================================================================
            # 🏷️ ชื่อระดับชั้น
            # ================================================================
            grade_labels= {
                "G-01": "ป.1",
                "G-02": "ป.2",
                "G-03": "ป.3",
                "G-04": "ป.4",
                "G-05": "ป.5",
                "G-06": "ป.6",
                "G-07": "ม.1",
                "G-08": "ม.2",
                "G-09": "ม.3",
                "G-10": "ม.4",
                "G-11": "ม.5",
                "G-12": "ม.6"
            }
            grade_count["GradeLabel"] = (
                grade_count["Grade"]
                .astype(str)
                .map(grade_labels)
            )
            # ================================================================
            # 📈 สร้างกราฟ
            # ================================================================
            fig_grade = px.bar(
                grade_count,
                x="GradeLabel",
                y="Count",
                text="Count",
                color="GradeLabel",
                custom_data=["Percent"],
                category_orders={
                    "GradeLabel": grade_count["GradeLabel"].tolist()
                },
                color_discrete_sequence=[
                    "#B7E4D8",  # เขียวอ่อน ชัดเจน
                    "#8FD3C1",  # เขียวมิ้นต์อ่อน
                    "#68C2AA",  # เขียวมิ้นต์
                    "#4CAF8A",  # เขียวหยกอ่อน
                    "#41AE76",      # เขียวมรกตสด
                    "#238B45",  # เขียวมรกตกลาง
                    "#006D2C",  # เขียวมรกตเข้ม
                    "#00441B",  # เขียวไพน์เข้ม
                    "#003615",  # เขียวเข้ม
                    "#00280F",  # เขียวเข้มลึก
                    "#001F0B",  # เขียวเกือบดำ
                    "#001407"   # เขียวเข้มสุด
                ]   
            )
            fig_grade.update_traces(
                texttemplate=
                    "<b>%{y:,} คน</b><br>"
                    "(%{customdata[0]:.1f}%)",
                textposition="outside",
                textfont=dict(
                    size=11,
                    color="#334155"
                ),
                marker=dict(
                    line=dict(width=0),
                    cornerradius=6
                ),
                hovertemplate=
                    "<b>🎓 ระดับชั้น:</b> %{x}<br>" +
                    "<b>👥 จำนวน:</b> %{y:,} คน<br>" +
                    "<b>📊 คิดเป็น:</b> " +
                    "%{customdata[0]:.2f}% ของทั้งหมด" +
                    "<extra></extra>"
            )
            max_grade_count = (
                grade_count["Count"].max()
                if not grade_count.empty
                else 100
            )
            fig_grade.update_layout(
                height=400,
                template="plotly_white",
                showlegend=False,
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=20,
                    r=50,
                    t=20,
                    b=70
                ),
                xaxis=dict(
                    title="ระดับชั้น",
                    title_font=dict(
                        size=13,
                        color="#64748B"
                    ),
                    tickfont=dict(
                        size=11,
                        color="#1E293B"
                    ),
                    showgrid=False
                ),
                yaxis=dict(
                    title="จำนวนนักเรียน (คน)",
                    title_font=dict(
                        size=13,
                        color="#64748B"
                    ),
                    gridcolor="#F1F5F9",
                    zeroline=False,
                    range=[
                        0,
                        max_grade_count * 1.30
                    ]
                )
            )
            st.plotly_chart(
                fig_grade,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
            # ================================================================
            # 💡 ข้อสังเกต
            # ================================================================
            if not grade_count.empty:
                top_grade_row = grade_count.loc[
                    grade_count["Count"].idxmax()
                ]
                top_grade = top_grade_row["GradeLabel"]
                top_grade_count = top_grade_row["Count"]
                top_grade_percent = top_grade_row["Percent"]
            else:
                top_grade = "-"
                top_grade_count = 0
                top_grade_percent = 0
            st.markdown(
                f"""
                <div style="
                    position: relative;
                    margin-top: -10px;
                    margin-bottom: 15px;
                    background-color: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-left: 4px solid #3B82F6;
                    border-radius: 8px;
                    padding: 10px 12px;
                    font-size: 13px;
                    color: #334155;
                    line-height: 1.6;
                ">
                    💡 <strong>ข้อสังเกตเชิงสถิติ:</strong>
                    ผู้เรียนส่วนใหญ่อยู่ใน<strong>{top_grade}</strong>
                    จำนวน<strong>{top_grade_count:,} คน({top_grade_percent:.1f}%)</strong>
                    จากผู้เรียนทั้งหมด<strong>{total_grade_students:,} คน</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ------------------------------------------------------------------------
    # 📦 ROW 3: SEMESTER DISTRIBUTION
    # ------------------------------------------------------------------------
    with st.container(border=True, height=600):
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='
                    color: #1E293B;
                    margin: 0;
                    font-size: 20px;
                    font-weight: 700;
                '>
                    📅 จำนวนนักเรียนตามภาคการศึกษา</h3>
                <p style='
                    color: #64748B;
                    font-size: 14px;
                    margin: 2px 0 0 0;
                '>
                    การกระจายตัวของผู้เรียนจำแนกตามภาคการศึกษา
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        # ================================================================
        # 📊 เตรียมข้อมูลภาคเรียน
        # ================================================================
        semester_count = (
            filtered_df["Semester"]
            .value_counts()
            .reset_index()
        )
        semester_count.columns = [
            "Semester",
            "Count"
        ]
        # ================================================================
        # 🔄 แปลงรหัสภาคเรียน
        # ================================================================
        semester_count["Semester"] = semester_count["Semester"].replace({
            "F": "ภาคการศึกษาที่ 1",
            "S": "ภาคการศึกษาที่ 2"
        })
        semester_count["Semester"] = pd.Categorical(
                        semester_count["Semester"],
                        categories=[
                            "ภาคการศึกษาที่ 1",
                            "ภาคการศึกษาที่ 2"
                        ],
                        ordered=True
                    )
        semester_count = semester_count.sort_values("Semester")
        # ================================================================
        # 🔢 ป้องกันข้อมูลผิดประเภท / NaN
        # ================================================================
        semester_count["Count"] = pd.to_numeric(
            semester_count["Count"],
            errors="coerce"
        ).fillna(0)
        # ================================================================
        # 📊 จำนวนนักเรียนทั้งหมด
        # ================================================================
        total_semester_students = semester_count["Count"].sum()
        # ================================================================
        # 📘 จำนวนนักเรียนภาคเรียนที่ 1
        # ================================================================
        semester1_count = semester_count.loc[
            semester_count["Semester"] == "ภาคการศึกษาที่ 1",
            "Count"
        ].sum()
        # ================================================================
        # 📙 จำนวนนักเรียนภาคเรียนที่ 2
        # ================================================================
        semester2_count = semester_count.loc[
            semester_count["Semester"] == "ภาคการศึกษาที่ 2",
            "Count"
        ].sum()
        # ================================================================
        # 📈 คำนวณเปอร์เซ็นต์
        # ================================================================
        if total_semester_students > 0:
            semester1_percent = (
                semester1_count
                / total_semester_students
                * 100
            )
            semester2_percent = (
                semester2_count
                / total_semester_students
                * 100
            )
        else:
            semester1_percent = 0
            semester2_percent = 0
        # ================================================================
        # 📊 เพิ่มเปอร์เซ็นต์ลงใน DataFrame
        # ================================================================
        if total_semester_students > 0:
            semester_count["Percent"] = (
                semester_count["Count"]
                / total_semester_students
                * 100
            ).fillna(0)
        else:
            semester_count["Percent"] = 0
        # ป้องกันค่า NaN / Inf ใน Percent
        semester_count["Percent"] = pd.to_numeric(
            semester_count["Percent"],
            errors="coerce"
        ).fillna(0)
        # ================================================================
        # 🏆 หาภาคเรียนที่มีนักเรียนมากที่สุด
        # ================================================================
        if semester1_count >= semester2_count:
            top_semester = "ภาคการศึกษาที่ 1"
            top_semester_count = semester1_count
            top_semester_percent = semester1_percent
        else:
            top_semester = "ภาคการศึกษาที่ 2"
            top_semester_count = semester2_count
            top_semester_percent = semester2_percent
        # ================================================================
        # 📦 แบ่งพื้นที่ : กราฟ / สรุปข้อมูล
        # ================================================================
        semester_chart_col, semester_summary_col = st.columns(
            [2, 1],
            gap="large"
        )
        # ====================================================================
        # 📊 ฝั่งซ้าย : กราฟวงกลม
        # ====================================================================
        with semester_chart_col:
            fig_semester = px.pie(
                semester_count,
                values="Count",
                names="Semester",
                hole=0.60,
                color="Semester",
                color_discrete_map={
                    "ภาคการศึกษาที่ 1": "#35A6DB",
                    "ภาคการศึกษาที่ 2": "#383AB5"
                },
            )
            # ================================================================
            # 🎨 รูปแบบกราฟ
            # ================================================================
            fig_semester.update_traces(
                textposition="inside",
                texttemplate=
                    "<b>%{percent}</b><br>"
                    "%{value:,} คน",
                textfont=dict(
                    size=15,
                    color="white"
                ),
                marker=dict(
                    line=dict(
                        color="white",
                        width=3
                    )
                ),
                sort=False,          # เพิ่ม
                rotation=180,        # เพิ่ม
                hovertemplate=
                    "<b>📅 %{label}</b><br>" +
                    "<b>👥 จำนวน:</b> %{value:,} คน<br>" +
                    "<b>📊 คิดเป็น:</b> %{percent}<br>" +
                    "<extra></extra>"
            )
            # ================================================================
            # ⚙️ ตั้งค่ากราฟ
            # ================================================================
            fig_semester.update_layout(        
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=40
                ),
                legend=dict(
                    orientation="h",
                    y=-0.05,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=13)
                ),
                annotations=[
                    dict(
                        x=0.5,
                        y=0.5,
                        text=
                            f"<span style='"
                            f"color:#64748B;"
                            f"font-size:12px;'>"
                            f"นักเรียนทั้งหมด"
                            f"</span><br><br>"
                            f"<b style='"
                            f"font-size:24px;"
                            f"color:#1E293B;'>"
                            f"{total_semester_students:,} คน"
                            f"</b>",
                        showarrow=False
                    )
                ]
            )
            # ================================================================
            # 📊 แสดงกราฟ
            # ================================================================

            st.plotly_chart(
                fig_semester,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
        # ====================================================================
        # 📋 ฝั่งขวา : สรุปข้อมูล
        # ====================================================================
        with semester_summary_col:

            # ------------------------------------------------------------
            # 📦 กรอบหลักฝั่งขวา ให้เต็มพื้นที่
            # ------------------------------------------------------------
            with st.container(border=True, height=445):

                # ============================================================
                # 📊 หัวข้อสรุปข้อมูล
                # ============================================================
                st.markdown(
                    """
                    <div style="
                        width:100%;
                        box-sizing:border-box;
                        background:linear-gradient(
                            135deg,
                            #EFF6FF 0%,
                            #DBEAFE 100%
                        );
                        border:1px solid #BFDBFE;
                        border-radius:12px;
                        padding:10px;
                        text-align:center;
                        margin-bottom:10px;
                    ">
                        <span style="
                            color:#1E40AF;
                            font-size:16px;
                            font-weight:700;
                        ">
                            📊 สรุปข้อมูล
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ============================================================
                # 📘 ภาคการศึกษาที่ 1
                # ============================================================
                st.markdown(
                    f"""
                    <div style="
                        width:100%;
                        box-sizing:border-box;
                        background:#EFF6FF;
                        border-radius:10px;
                        padding:10px;
                        margin-bottom:8px;
                    ">
                    <div style="
                        color:#1E293B;
                        font-size:13px;
                        font-weight:600;
                    ">
                        📘 ภาคการศึกษาที่ 1
                    </div>

                    <div style="
                        color:#3FADE0;
                        font-size:22px;
                        font-weight:700;
                        margin-top:3px;
                    ">
                        {semester1_count:,.0f} คน
                    </div>

                    <div style="35A6DB;
                        color:#64748B;
                        font-size:12px;
                    ">
                        คิดเป็น {semester1_percent:.1f}%
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ============================================================
                # 📙 ภาคการศึกษาที่ 2
                # ============================================================
                st.markdown(
                    f"""    
                    <div style="width:100%; box-sizing:border-box; 
                        background:#F5F3FF; border-radius:10px;
                        padding:10px; margin-bottom:8px;">
                    <div style="
                        color:#1E293B;
                        font-size:13px;
                        font-weight:600;
                        ">
                        📙 ภาคการศึกษาที่ 2
                    </div>

                    <div style="
                        color:#383AB5;
                        font-size:22px;
                        font-weight:700;
                        margin-top:3px;
                    ">
                        {semester2_count:,.0f} คน
                    </div>

                    <div style="
                        color:#64748B;
                        font-size:12px;
                    ">
                        คิดเป็น {semester2_percent:.1f}%
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ============================================================
                # 💡 ข้อสังเกตเชิงสถิติ
                # ============================================================
                st.markdown(
                    f"""
                    <div style="
                        width:100%;
                        box-sizing:border-box;
                        background-color:#F8FAFC;
                        border:1px solid #E2E8F0;
                        border-left:4px solid #3B82F6;
                        border-radius:8px;
                        padding:10px;
                        font-size:13px;
                        color:#334155;
                        line-height:1.7;
                        margin-top:8px;
                    ">
                        💡 <strong>ข้อสังเกตเชิงสถิติ:</strong>
                        ผู้เรียนส่วนใหญ่อยู่ใน
                        <strong>{top_semester}</strong>
                        จำนวน
                        <strong>
                            {top_semester_count:,.0f} คน
                            ({top_semester_percent:.1f}%)
                        </strong>
                        จากผู้เรียนทั้งหมด
                        <strong>
                            {total_semester_students:,.0f} คน
                        </strong>
                    </div>
                    """,
                    unsafe_allow_html=True
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
            .kpi-text-sub {
                font-size: 0.75rem;
                color: #64748B;
                font-weight: 500;
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
    st.markdown("<h1 style='color: #0A2540; margin-bottom: 0px; font-weight: bold;'>Academic Performance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #475569; font-size: 1.1rem; margin-top: 0px; margin-bottom: 20px;'>ผลสัมฤทธิ์ทางการเรียนของนักเรียน</p>", unsafe_allow_html=True)

    # --- คำนวณค่าตัวเลขและเปอร์เซ็นต์สำหรับ KPI ---
    total_students = len(filtered_df)
    class_counts = filtered_df["Class"].value_counts() if not filtered_df.empty else {}
    
    high_count = class_counts.get("H", 0)
    mid_count = class_counts.get("M", 0)
    low_count = class_counts.get("L", 0)

    # คำนวณ % สัดส่วนเพื่อบอกบริบทที่ชัดเจน
    high_pct = (high_count / total_students * 100) if total_students > 0 else 0
    mid_pct = (mid_count / total_students * 100) if total_students > 0 else 0
    low_pct = (low_count / total_students * 100) if total_students > 0 else 0

    # --- ดึงบริบทจาก Filter ที่เลือกจริงใน Sidebar ---
    stage_disp = stage_map.get(selected_stage, selected_stage)
    sem_disp = semester_map.get(selected_semester, selected_semester)
    gen_disp = gender_map.get(selected_gender, selected_gender)

    context_str = f"ระดับชั้น: {stage_disp} | ภาคเรียน: {sem_disp} | เพศ: {gen_disp}"

    # --- 1. KPI Cards ด้านบน (4 การ์ด) ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #DCFCE7;'>👨‍👩‍👧‍👦</div>
                <div>
                    <div class='kpi-text-title'>นักเรียนทั้งหมด</div>
                    <div class='kpi-text-value'>{total_students:,} คน</div>
                    <div class='kpi-text-sub'>กลุ่มเป้าหมายการวิเคราะห์</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #E0F2FE;'>🏃</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียนระดับสูง (High)</div>
                    <div class='kpi-text-value'>{high_count:,} คน</div>
                    <div class='kpi-text-sub'>คิดเป็น <b>{high_pct:.1f}%</b> ของทั้งหมด</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #FCE7F3;'>🙋‍♀️</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียนปานกลาง (Medium)</div>
                    <div class='kpi-text-value'>{mid_count:,} คน</div>
                    <div class='kpi-text-sub'>คิดเป็น <b>{mid_pct:.1f}%</b> ของทั้งหมด</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='kpi-card-custom'>
                <div class='kpi-icon-bg' style='background-color: #FEF3C7;'>🙋‍♂️</div>
                <div>
                    <div class='kpi-text-title'>ผลการเรียนระดับต่ำ (Low)</div>
                    <div class='kpi-text-value'>{low_count:,} คน</div>
                    <div class='kpi-text-sub'>คิดเป็น <b>{low_pct:.1f}%</b> ของทั้งหมด</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # =====================================================
    # 2. โซนกลาง (Grouped Bar Chart ตามช่วงชั้น)
    # =====================================================
    with st.container(border=True):

        st.markdown(
            """
            <h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>
            การกระจายผลสัมฤทธิ์ทางการเรียนตามช่วงชั้น
            </h3>
            <p style='color: #64748B; font-size: 0.8rem; margin-top: 2px; margin-bottom: 10px;'>
                แสดงจำนวนนักเรียนจำแนกตามระดับผลสัมฤทธิ์ทางการเรียนและช่วงชั้น •
                <i>H = ระดับสูง, M = ระดับปานกลาง, L = ระดับต่ำ</i>
            </p>
            """,
            unsafe_allow_html=True
        )
        if "Class" in filtered_df.columns and not filtered_df.empty:
            # ============================================================
            # 🏫 ช่วงชั้น
            # ============================================================
            stage_map = {
                "lowerlevel": "ประถม",
                "MiddleSchool": "มัธยมต้น",
                "HighSchool": "มัธยมปลาย"
            }
            if "StageID" in filtered_df.columns:
                filtered_df["Stage_TH"] = (
                    filtered_df["StageID"]
                    .map(stage_map)
                    .fillna(filtered_df["StageID"])
                )
            # ============================================================
            # 🎓 ระดับผลสัมฤทธิ์
            # ============================================================
            class_label_map = {
                "H": "ระดับสูง",
                "M": "ระดับปานกลาง",
                "L": "ระดับต่ำ"
            }   
            filtered_df["Class_TH"] = (
                filtered_df["Class"]
                .map(class_label_map)
                .fillna(filtered_df["Class"])
            )
            # ============================================================
            # 🎨 สีระดับผลสัมฤทธิ์
            # ============================================================
            color_class_map = {
                "ระดับต่ำ": "#b0120a",
                "ระดับปานกลาง": "#f57f17",
                "ระดับสูง": "#33691e"
            }
            # ============================================================
            # 1️⃣ กราฟการกระจายผลสัมฤทธิ์ตามช่วงชั้น
            # ============================================================
            stage_class_df = (
                filtered_df
                .groupby(
                    [
                        "Stage_TH",
                        "Class",
                        "Class_TH"
                    ],
                    as_index=False
                )
                .size()
                .rename(
                    columns={
                        "size": "Count"
                    }
                )
            )
            # ------------------------------------------------------------
            # คำนวณร้อยละภายในแต่ละช่วงชั้น
            # ------------------------------------------------------------
            total_stage = (
                stage_class_df
                .groupby("Stage_TH")["Count"]
                .transform("sum")
            )
            stage_class_df["Percent"] = (
                stage_class_df["Count"]
                / total_stage
                * 100
            )
            # ------------------------------------------------------------
            # ลำดับช่วงชั้น
            # ------------------------------------------------------------
            stage_order = [
                "ประถม",
                "มัธยมต้น",
                "มัธยมปลาย"
            ]
            available_stage_order = [
                stage
                for stage in stage_order
                if stage in stage_class_df["Stage_TH"].values
            ]
            # ------------------------------------------------------------
            # สร้างตำแหน่งแท่ง
            # เรียงจำนวนผู้เรียนจริง น้อย → มาก
            # ------------------------------------------------------------
            bar_x = []
            bar_y = []
            bar_class = []
            bar_percent = []
            tick_positions = []
            tick_labels = []
            current_x = 0
            for stage in available_stage_order:
                stage_data = (
                    stage_class_df[
                        stage_class_df["Stage_TH"] == stage
                    ]
                    .sort_values(
                        "Count",
                        ascending=True
                    )
                    .reset_index(drop=True)
                )
                if stage_data.empty:
                    continue
                stage_positions = []
                for _, row in stage_data.iterrows():
                    bar_x.append(current_x)
                    bar_y.append(row["Percent"])
                    bar_class.append(row["Class_TH"])
                    bar_percent.append(row["Percent"])
                    stage_positions.append(current_x)
                    current_x += 1
                if stage_positions:
                    tick_positions.append(
                        sum(stage_positions)
                        / len(stage_positions)
                    )
                    tick_labels.append(stage)
                # เว้นระยะระหว่างช่วงชั้น
                current_x += 1
            # ------------------------------------------------------------
            # สร้าง Figure
            # ------------------------------------------------------------
            fig_bar = go.Figure()
            class_display_order = [
                "ระดับต่ำ",
                "ระดับปานกลาง",
                "ระดับสูง"
            ]
            for class_name in class_display_order:
                x_data = []
                y_data = []
                text_data = []
                customdata = []

                for x, y, cls, pct in zip(
                    bar_x,
                    bar_y,
                    bar_class,
                    bar_percent
                ):

                    if cls == class_name:
                        x_data.append(x)
                        y_data.append(y)
                        text_data.append(
                            f"{y:.1f}%"
                        )
                if not x_data:
                    continue
                fig_bar.add_trace(
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        name=class_name,
                        text=text_data,
                        textposition="outside",
                        marker_color=color_class_map.get(
                            class_name,
                            "#64748B"
                        ),
                        hovertemplate=(
                            "<b>ระดับผลสัมฤทธิ์:</b> "
                            + class_name
                            + "<br>"
                            "<b>สัดส่วน:</b> "
                            + "%{y:.1f}%"
                            + "<br>"
                            "<b>จำนวนนักเรียน:</b> "
                            + "%{customdata:,} คน"
                            + "<extra></extra>"
                        )
                    )
                )
            # ------------------------------------------------------------
            # รูปแบบกราฟ
            # ------------------------------------------------------------
            max_count = (
                max(bar_y)
                if bar_y
                else 0
            )
            fig_bar.update_layout(
                height=380,
                barmode="group",
                margin=dict(
                    l=10,
                    r=10,
                    t=35,
                    b=50
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_text=""
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig_bar.update_xaxes(
                title_text="ช่วงชั้น",
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels,
                tickfont=dict(size=11),
                showgrid=False
            )
            fig_bar.update_yaxes(
                title_text="สัดส่วน (%)",
                tickfont=dict(size=10),
                gridcolor="#E2E8F0",
                rangemode="tozero",
                range=[0, 100]
            )
            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
                key="chart_bar_stage_academic"
            )
            # ============================================================
            # 🔍 วิเคราะห์ข้อมูลกราฟการกระจายผลสัมฤทธิ์
            # ============================================================
            overall_class_count = (
                stage_class_df
                .groupby("Class_TH")["Count"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )
            stage_count = (
                stage_class_df
                .groupby("Stage_TH")["Count"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )
            # ============================================================
            # 📊 เตรียมคะแนนการมีส่วนร่วม
            # ============================================================
            participation_cols = [
                "raisedhands",
                "VisITedResources",
                "AnnouncementsView",
                "Discussion"
            ]
            available_participation_cols = [
                col
                for col in participation_cols
                if col in filtered_df.columns
            ]
            score_df = filtered_df.copy()
            for col in available_participation_cols:
                score_df[col] = pd.to_numeric(
                    score_df[col],
                    errors="coerce"
                )
            if available_participation_cols:
                score_df["_ParticipationAvg"] = (
                    score_df[
                        available_participation_cols
                    ]
                    .mean(axis=1)
                )
            else:
                score_df["_ParticipationAvg"] = 0
            # ------------------------------------------------------------
            # คะแนนเฉลี่ยรวมของแต่ละระดับ
            # น้อย → มาก
            # ------------------------------------------------------------
            class_score_df = (
                score_df
                .groupby("Class_TH")["_ParticipationAvg"]
                .mean()
                .sort_values(
                    ascending=True
                )
            )
            if not overall_class_count.empty:
                highest_class = (
                    overall_class_count.index[0]
                )
                highest_class_count = int(
                    overall_class_count.iloc[0]
                )
                lowest_class = (
                    overall_class_count.index[-1]
                )
                lowest_class_count = int(
                    overall_class_count.iloc[-1]
                )
            else:
                highest_class = "-"
                highest_class_count = 0
                lowest_class = "-"
                lowest_class_count = 0

            if not stage_count.empty:
                highest_stage = (
                    stage_count.index[0]
                )
                highest_stage_count = int(
                    stage_count.iloc[0]
                )
            else:
                highest_stage = "-"
                highest_stage_count = 0

            # =====================================================
            # 🔍 ดึงค่าเปอร์เซ็นต์จากข้อมูลที่ใช้สร้างแท่งกราฟโดยตรง
            # =====================================================
            if bar_y:
                max_index = bar_y.index(max(bar_y))
                min_index = bar_y.index(min(bar_y))
                highest_bar_class = bar_class[max_index]
                highest_bar_percent = bar_y[max_index]
                lowest_bar_class = bar_class[min_index]
                lowest_bar_percent = bar_y[min_index]
            else:
                highest_bar_class = "-"
                highest_bar_percent = 0
                lowest_bar_class = "-"
                lowest_bar_percent = 0
            st.markdown(
                f"""
                <div style="
                    background-color:#F8FAFC;
                    border:1px solid #E2E8F0;
                    border-left:4px solid #3B82F6;
                    border-radius:12px;
                    padding:14px 16px;
                    margin-top:10px;
                    margin-bottom:10px;
                    line-height:1.8;
                ">
                <div style="
                    font-weight:700;
                    color:#1E3A5F;
                    margin-bottom:8px;
                ">🔍 จากข้อมูล</div>
                <div style="
                    color:#334155;
                    font-size:14px;
                ">
                    • สัดส่วนที่สูงที่สุดจากแท่งกราฟ คือ
                    <b>{highest_bar_class}</b>
                    <b>{highest_bar_percent:.1f}%</b>
                    ขณะที่สัดส่วนที่ต่ำที่สุด คือ
                    <b>{lowest_bar_class}</b>
                    <b>{lowest_bar_percent:.1f}%</b>
                    <br>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ============================================================
            # 📊 เตรียมข้อมูลพฤติกรรมการเรียนรู้
            # ============================================================
            total_students = len(filtered_df)
            behavior_df = filtered_df.copy()
            behavior_cols = [
                "raisedhands",
                "VisITedResources",
                "AnnouncementsView",
                "Discussion"
            ]
            # ------------------------------------------------------------
            # แปลงพฤติกรรมเป็นตัวเลข
            # ------------------------------------------------------------
            for col in behavior_cols:
                if col in behavior_df.columns:
                    behavior_df[col] = pd.to_numeric(
                        behavior_df[col],
                        errors="coerce"
                    )
            # ------------------------------------------------------------
            # คอลัมน์พฤติกรรมที่มีอยู่จริง
            # ------------------------------------------------------------
            available_behavior_cols = [
                col
                for col in behavior_cols
                if col in behavior_df.columns
            ]
            # ------------------------------------------------------------
            # คะแนนการมีส่วนร่วมรวม
            # ------------------------------------------------------------
            if available_behavior_cols:
                behavior_df["ParticipationScore"] = (
                    behavior_df[
                        available_behavior_cols
                    ]
                    .sum(axis=1)
                )
            else:
                behavior_df["ParticipationScore"] = 0
            # ============================================================
            # 🟦🟪 ภาคเรียน
            # ============================================================
            if "Semester" in behavior_df.columns:
                behavior_df["Semester_Label"] = (
                    behavior_df["Semester"]
                    .map(
                        {
                            "F": "ภาคเรียนที่ 1",
                            "S": "ภาคเรียนที่ 2"
                        }
                    )
                    .fillna(
                        behavior_df["Semester"]
                    )
                )
            else:
                behavior_df["Semester_Label"] = "-"
            # ============================================================
            # 🏫 ช่วงชั้น
            # ============================================================
            if "StageID" in behavior_df.columns:
                behavior_df["Stage_Name"] = (
                    behavior_df["StageID"]
                    .map(stage_map)
                    .fillna(
                        behavior_df["StageID"]
                    )
                )
            else:
                behavior_df["Stage_Name"] = "-"
            # ============================================================
            # 🎓 ระดับผลสัมฤทธิ์
            # ============================================================
            behavior_df["Class_Label"] = (
                behavior_df["Class"]
                .map(
                    {
                        "L": "ระดับต่ำ",
                        "M": "ระดับปานกลาง",
                        "H": "ระดับสูง"
                    }
                )
                .fillna(
                    behavior_df["Class"]
                )
            )
            # ============================================================
            # 🎨 สีภาคเรียน
            # ============================================================
            semester_color_map = {
                "ภาคเรียนที่ 1": "#3B82F6",
                "ภาคเรียนที่ 2": "#8B5CF6"
            }
            # ============================================================
            # ⭐ ลำดับระดับผลสัมฤทธิ์
            # คะแนนการมีส่วนร่วมรวมเฉลี่ยจริง
            # น้อย → มาก
            # ============================================================
            class_participation_avg = (
                behavior_df
                .groupby("Class_Label")[
                    "ParticipationScore"
                ]
                .mean()
                .sort_values(
                    ascending=True
                )
            )
            class_value_order = (
                class_participation_avg
                .index
                .tolist()
            )
            # ------------------------------------------------------------
            # เพิ่มระดับที่มีอยู่จริงแต่ไม่มีค่าเฉลี่ย
            # ------------------------------------------------------------
            existing_classes = (
                behavior_df["Class_Label"]
                .dropna()
                .unique()
                .tolist()
            )
            for class_name in existing_classes:
                if class_name not in class_value_order:
                    class_value_order.append(
                        class_name
                    )
            # ============================================================
            # 🏫 ลำดับช่วงชั้น
            # ============================================================
            fixed_stage_order = [
                "ประถม",
                "มัธยมต้น",
                "มัธยมปลาย"
            ]
            available_stage_order = [
                stage
                for stage in fixed_stage_order
                if stage in behavior_df["Stage_Name"].values
            ]
            # ============================================================
            # 🧠 Function สร้างกราฟพฤติกรรม
            # ============================================================
            def create_behavior_effect_chart(
                df,
                y_col,
                title_text,
                y_label
            ):
                if (
                    df.empty
                    or y_col not in df.columns
                ):
                    return (
                        px.bar(),
                        pd.DataFrame()
                    )
                # ========================================================
                # ค่าเฉลี่ยของตัวแปรที่กำลังแสดง
                # แยกตามภาคเรียน + ระดับผลสัมฤทธิ์
                # ========================================================
                avg_df = (
                    df
                    .groupby(
                        [
                            "Semester_Label",
                            "Class_Label"
                        ],
                        as_index=False
                    )[y_col]
                    .mean()
                )
                if avg_df.empty:
                    return (
                        px.bar(),
                        avg_df
                    )
                # ========================================================
                # จำนวนผู้เรียนจริง
                # ใช้สำหรับ "จัดลำดับ" L/M/H
                # ไม่ได้ใช้เป็นค่าความสูงของแท่ง
                # ========================================================
                count_df = (
                    df
                    .groupby(
                        [
                            "Semester_Label",
                            "Class_Label"
                        ],
                        as_index=False
                    )
                    .size()
                    .rename(
                        columns={
                            "size": "StudentCount"
                        }
                    )
                )
                # ========================================================
                # รวมจำนวนผู้เรียนเข้ากับค่าเฉลี่ย
                # ========================================================
                avg_df = avg_df.merge(
                    count_df,
                    on=[
                        "Semester_Label",
                        "Class_Label"
                    ],
                    how="left"
                )
                # ========================================================
                # สีระดับผลสัมฤทธิ์
                # ========================================================
                class_color_map = {
                    "ระดับต่ำ": "#f57f17",
                    "ระดับปานกลาง": "#d0120a",
                    "ระดับสูง": "#33691e"
                }
                # ========================================================
                # สีภาคเรียน
                # ========================================================
                semester_color_map = {
                    "ภาคเรียนที่ 1": "#3B82F6",
                    "ภาคเรียนที่ 2": "#8B5CF6"
                }
                # ========================================================
                # สร้าง 2 กราฟ
                # ========================================================
                semester_order = [
                    "ภาคเรียนที่ 1",
                    "ภาคเรียนที่ 2"
                ]
                available_semesters = [
                    semester
                    for semester in semester_order
                    if semester in avg_df["Semester_Label"].values
                ]
                if not available_semesters:
                    return (
                        px.bar(),
                        avg_df
                    )
                fig = make_subplots(
                    rows=1,
                    cols=len(available_semesters),
                    horizontal_spacing=0.10
                )
                # ========================================================
                # สร้างกราฟทีละภาคเรียน
                # ========================================================
                for col_index, semester in enumerate(
                    available_semesters,
                    start=1
                ):
                    semester_df = (
                        avg_df[
                            avg_df["Semester_Label"]
                            == semester
                        ]
                        .copy()
                    )               
                    if semester_df.empty:
                        continue
                    # ====================================================
                    # ⭐ เรียงระดับผลสัมฤทธิ์ตาม "จำนวนผู้เรียนจริง"
                    # น้อย → มาก
                    # ====================================================
                    class_order_df = (
                        semester_df[
                            [
                                "Class_Label",
                                y_col
                            ]
                        ]
                        .drop_duplicates()
                        .sort_values(
                            y_col,
                            ascending=True
                        )
                    )
                    class_order = (
                        class_order_df[
                            "Class_Label"
                        ]
                        .tolist()
                    )
                    # ====================================================
                    # ⭐ กำหนดลำดับแกน X ตามจำนวนผู้เรียนจริง
                    # น้อย → มาก
                    # ====================================================
                    short_class_map = { 
                        "ระดับต่ำ": "L",
                        "ระดับปานกลาง": "M",
                        "ระดับสูง": "H"
                    }
                    short_order = [
                        short_class_map.get(
                        class_name,
                        class_name
                        )
                        for class_name in class_order
                    ]
                    # ====================================================
                    # สร้างแท่งตามลำดับจำนวนจริง
                    # ====================================================
                    for class_name in class_order:
                        row = semester_df[
                            semester_df["Class_Label"]
                            == class_name
                        ]
                        if row.empty:
                            continue
                        row = row.iloc[0]
                        avg_value = row[y_col]
                        student_count = int(
                            row["StudentCount"]
                        )
                        # -----------------------------------------------
                        # สีตามระดับผลสัมฤทธิ์
                        # -----------------------------------------------
                        bar_color = class_color_map.get(
                            class_name,
                            "#64748B"
                        )
                        # -----------------------------------------------
                        # ชื่อย่อ L/M/H สำหรับแกน X
                        # -----------------------------------------------
                        short_class_map = {
                            "ระดับต่ำ": "L",
                            "ระดับปานกลาง": "M",
                            "ระดับสูง": "H"
                        }
                        short_class = short_class_map.get(
                            class_name,
                            class_name
                        )
                        # -----------------------------------------------
                        # เพิ่มแท่ง
                        # -----------------------------------------------
                        fig.add_trace(
                            go.Bar(
                                x=[short_class],
                                y=[avg_value],
                                name=class_name,
                                marker_color=bar_color,
                                text=[
                                    f"{avg_value:.2f}"
                                ],
                                textposition="outside",
                                customdata=[
                                    [
                                        student_count,
                                        class_name,
                                        semester
                                    ]
                                ],
                                hovertemplate=(
                                    "<b>ภาคเรียน:</b> "
                                    "%{customdata[2]}"
                                    "<br>"
                                    "<b>ระดับผลสัมฤทธิ์:</b> "
                                    "%{customdata[1]}"
                                    "<br>"
                                    "<b>จำนวนนักเรียน:</b> "
                                    "%{customdata[0]:,} คน"
                                    "<br>"
                                    "<b>"
                                    + y_label
                                    + ":</b> "
                                    "%{y:.2f}"
                                    "<extra></extra>"
                                ),
                                showlegend=False
                            ),
                            row=1,
                            col=col_index
                        )
                    # ====================================================
                    # แกน X
                    # ใช้ลำดับที่คำนวณจากจำนวนจริง
                    # ====================================================
                    fig.update_xaxes(
                        title_text= "",
                        categoryorder="array",
                        categoryarray=short_order,
                        showgrid=False,
                        showticklabels=False,
                        row=1,
                        col=col_index
                    )
                    # ====================================================
                    # แกน Y
                    # ====================================================
                    max_value = (
                        avg_df[y_col].max()
                        if not avg_df.empty
                        else 0
                    )
                    y_max = (
                        max_value * 1.25
                        if max_value > 0
                        else 1
                    )
                    fig.update_yaxes(
                        title_text=(
                            y_label
                            if col_index == 1
                            else ""
                        ),
                        range=[
                            0,
                            y_max
                        ],
                        rangemode="tozero",
                        gridcolor="#E2E8F0",
                        showgrid=True,
                        row=1,
                        col=col_index
                    )
                # ========================================================
                # 📝 ป้ายภาคเรียน + ชื่อแกน X
                # แสดงด้านล่างกราฟ
                # ========================================================
                # ชื่อระดับผลสัมฤทธิ์ แสดงเพียงครั้งเดียวตรงกลาง
                fig.add_annotation(
                    x=0.5,
                    y=-0.16,
                    xref="paper",
                    yref="paper",
                    text="ระดับผลสัมฤทธิ์ทางการเรียน",
                    showarrow=False,
                    font=dict(
                        size=13,
                        color="#64748B"
                    ),
                    xanchor="center",
                    yanchor="middle"
                )
                # ชื่อภาคเรียน
                for col_index, semester in enumerate(
                    available_semesters,
                    start=1
                ):
                    # หาตำแหน่งกึ่งกลางของแต่ละกราฟ
                    axis_name = (
                        "xaxis"
                        if col_index == 1
                        else f"xaxis{col_index}"
                    )
                    x_domain = fig.layout[
                    axis_name
                    ].domain
                    x_center = (
                        x_domain[0]
                        + x_domain[1]
                    ) / 2
                    fig.add_annotation(
                        x=x_center,
                        y=-0.09,
                        xref="paper",
                        yref="paper",
                        text=semester,
                        showarrow=False,
                        font=dict(
                            size=13,
                            color="#64748B"
                        ),
                        xanchor="center",
                        yanchor="middle"
                    )
                # ========================================================
                # Legend
                # แสดง L/M/H พร้อมสีเดิม
                # ========================================================
                for class_name, class_color in class_color_map.items():
                    fig.add_trace(
                        go.Bar(
                            x=[None],
                            y=[None],
                            name=class_name,
                            marker_color=class_color,
                            showlegend=True
                        )           
                    )
                # ========================================================
                # รูปแบบกราฟ
                # ========================================================
                fig.update_layout(
                    height=390,
                    barmode="group",
                    margin=dict(
                        l=40,
                        r=20,
                        t=75,
                        b=80
                    ),
                    legend=dict(
                        orientation="h",
                        y=-0.25,
                        x=0.5,
                        xanchor="center",
                        title_text=""
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return (
                    fig,
                    avg_df
                )
            # ============================================================
            # 🔍 Function วิเคราะห์ข้อมูลแบบ Dynamic
            # ============================================================
            def generate_insight_html(
                    avg_df,
                    col_name,
                    display_name
            ):
                if (
                    avg_df is None
                    or avg_df.empty
                    or col_name not in avg_df.columns
                ):
                    return ""

                # ========================================================
                # ค่าเฉลี่ยแต่ละระดับผลสัมฤทธิ์
                # ========================================================
                class_avg = (
                    avg_df
                    .groupby("Class_Label")[col_name]
                    .mean()
                    .sort_values(
                        ascending=True
                    )
                )
                # ========================================================
                # ระดับที่มีค่าสูงสุด / ต่ำสุด
                # ========================================================
                if not class_avg.empty:
                    highest_class = class_avg.idxmax()
                    highest_value = class_avg.max()

                    lowest_class = class_avg.idxmin()
                    lowest_value = class_avg.min()
                else:
                    highest_class = "-"
                    highest_value = 0
                    lowest_class = "-"
                    lowest_value = 0
                # ========================================================
                # ค่าเฉลี่ยแยกตามภาคเรียน
                # ========================================================
                semester_avg = (
                    avg_df
                    .groupby("Semester_Label")[col_name]
                    .mean()
                    .sort_values(
                        ascending=False
                    )
                )
                # ========================================================
                # ภาคเรียนที่มีค่าเฉลี่ยสูงสุด
                # ========================================================
                if not semester_avg.empty:

                    max_semester = semester_avg.index[0]
                    max_semester_val = semester_avg.iloc[0]
                else:
                    max_semester = "-"
                    max_semester_val = 0
                # ========================================================
                # ส่วนต่างระหว่างระดับผลสัมฤทธิ์สูงสุดกับต่ำสุด
                # ========================================================
                diff_value = (
                    highest_value
                    - lowest_value
                )
                # ========================================================
                # สร้างข้อความวิเคราะห์
                # ========================================================
                return f"""
                <div style="
                    background-color:#F8FAFC;
                    border:1px solid #E2E8F0;
                    border-left:4px solid #3B82F6;
                    border-radius:12px;
                    padding:12px 16px;
                    margin-top:8px;
                    margin-bottom:10px;
                    line-height:1.8;
                ">
                <div style="
                    font-weight:700;
                        color:#1E3A5F;
                        margin-bottom:6px;
                ">🔍 จากข้อมูล</div>
                <div style="
                    color:#334155;
                    font-size:14px;
                ">
                    • ระดับผลสัมฤทธิ์ที่มีค่าเฉลี่ย
                    <b>{display_name}</b>
                    สูงที่สุด คือ
                    <b>{highest_class}</b>
                    มีค่าเฉลี่ย
                    <b>{highest_value:.2f}</b>
                    <br>
                    • ระดับผลสัมฤทธิ์ที่มีค่าเฉลี่ย
                    <b>{display_name}</b>
                    ต่ำที่สุด คือ
                    <b>{lowest_class}</b>
                    มีค่าเฉลี่ย
                    <b>{lowest_value:.2f}</b>
                    <br>
                    • ค่าเฉลี่ยแตกต่างกัน
                    <b>{diff_value:.2f}</b>
                    คะแนน
                    <br>
                    • ภาคเรียนที่มีค่าเฉลี่ย
                    <b>{display_name}</b>
                    สูงที่สุด คือ
                    <b>{max_semester}</b>
                    มีค่าเฉลี่ย
                    <b>{max_semester_val:.2f}</b>
                </div>
                </div>
                """
            # ============================================================
            # 1️⃣ อัตราการขาดเรียน
            # ============================================================
            with st.container(border=True):
                st.markdown(
                    "### 1️⃣ อัตราการขาดเรียนที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )
                absence_map = {
                    "Under-7": "ขาดน้อยกว่า 7 วัน",
                    "Above-7": "ขาดมากกว่า 7 วัน"
                }
                behavior_df["Absence_Label"] = (
                    behavior_df["StudentAbsenceDays"]
                    .map(absence_map)
                    .fillna(
                        behavior_df["StudentAbsenceDays"]
                    )
                )
                # ========================================================
                # จำนวนผู้เรียน
                # แยกตาม ภาคเรียน + ระดับผลสัมฤทธิ์ + การขาดเรียน
                # ========================================================
                absence_df = (
                    behavior_df
                    .groupby(
                        [
                            "Semester_Label",
                            "Class",
                            "Absence_Label"
                        ]
                    )
                    .size()
                    .reset_index(
                        name="StudentCount"
                    )
                )
                # ========================================================
                # ⭐ จำนวนผู้เรียนรวมของแต่ละ L/M/H
                # ใช้เป็นฐานในการคำนวณเปอร์เซ็นต์
                # ========================================================
                class_count_df = (
                    behavior_df
                    .groupby(
                        [
                            "Semester_Label",
                            "Class"
                        ]
                    )
                    .size()
                    .reset_index(
                        name="TotalStudentCount"
                    )
                )
                absence_df = absence_df.merge(
                    class_count_df,
                    on=[
                        "Semester_Label",
                        "Class"
                    ],
                    how="left"
                )
                # ========================================================
                # ⭐ คำนวณเปอร์เซ็นต์
                # ========================================================
                absence_df["Percentage"] = (
                    absence_df["StudentCount"]
                    / absence_df["TotalStudentCount"]
                    * 100
                )
                # ========================================================
                # สีการขาดเรียน
                # ========================================================
                absence_color_map = {
                    "ขาดน้อยกว่า 7 วัน": "#3c6542",
                    "ขาดมากกว่า 7 วัน": "#db3707"
                }           
                # ========================================================
                # ลำดับภาคเรียน
                # ========================================================
                semester_order = [
                    "ภาคเรียนที่ 1",
                    "ภาคเรียนที่ 2"
                ]
                available_semesters = [
                    semester
                    for semester in semester_order
                    if semester in absence_df["Semester_Label"].values
                ]
                if available_semesters:
                    # ====================================================
                    # ⭐ ลำดับระดับผลสัมฤทธิ์
                    # L → M → H
                    # ====================================================
                    class_order = [
                        "L",
                        "M",
                        "H"
                    ]
                    # ====================================================
                    # สร้างกราฟแยกตามภาคเรียน
                    # ====================================================
                    fig1 = make_subplots(
                        rows=1,
                        cols=len(available_semesters),
                        horizontal_spacing=0.10
                    )
                    # ====================================================
                    # ⭐ ใช้สำหรับสร้าง Legend เพียงครั้งเดียว
                    # ====================================================
                    legend_shown = set()
                    # ====================================================
                    # สร้างกราฟทีละภาคเรียน
                    # ====================================================
                    for col_index, semester in enumerate(
                        available_semesters,
                        start=1
                    ):
                        semester_df = (
                            absence_df[
                                absence_df["Semester_Label"]
                                == semester
                            ]
                            .copy()
                        )
                        # ==================================================
                        # สร้างข้อมูลสำหรับกราฟ
                        # ==================================================
                        plot_rows = []
                        for class_name in class_order:
                            class_df = semester_df[
                                semester_df["Class"] == class_name
                            ].copy()
                            # ----------------------------------------------
                            # ⭐ เรียงจากเปอร์เซ็นต์น้อย → มาก
                            # ----------------------------------------------
                            class_df = (
                                class_df
                                .sort_values(
                                    "Percentage",
                                    ascending=True
                                )
                            )
                            for _, row in class_df.iterrows():
                                plot_rows.append({
                                    "Class": class_name,
                                    "Absence_Label": row[
                                        "Absence_Label"
                                    ],
                                    "StudentCount": int(
                                        row["StudentCount"]
                                    ),
                                    "Percentage": float(
                                        row["Percentage"]
                                    )
                                })
                        # ==================================================
                        # ตำแหน่งแท่ง
                        # ==================================================
                        x_positions = list(
                            range(
                                len(plot_rows)
                            )
                        )
                        # ==================================================
                        # ⭐ เพิ่มแท่งกราฟ
                        # ==================================================
                        for index, row in enumerate(
                            plot_rows
                        ):
                            class_name = row["Class"]
                            absence_type = row[
                                "Absence_Label"
                            ]
                            student_count = row[
                                "StudentCount"
                            ]
                            percentage = row[
                                "Percentage"
                            ]
                            # ----------------------------------------------
                            # Legend แสดงเพียงครั้งเดียว
                            # ----------------------------------------------
                            show_legend = (
                                absence_type
                                not in legend_shown
                            )
                            legend_shown.add(
                                absence_type
                            )
                            fig1.add_trace(
                                go.Bar(
                                    x=[index],
                                    # ⭐ ความสูงของแท่ง = เปอร์เซ็นต์
                                    y=[percentage],
                                    name=absence_type,
                                    marker_color=absence_color_map.get(
                                        absence_type,
                                        "#64748B"
                                    ),
                                    # ⭐ แสดงเปอร์เซ็นต์บนแท่ง
                                    text=[
                                        f"{percentage:.1f}%"
                                    ],
                                    textposition="outside",
                                    # ⭐ ป้องกันตัวเลขถูกตัด
                                    cliponaxis=False,
                                    # ==================================================
                                    # Hover แสดงทั้งเปอร์เซ็นต์ + จำนวนคน
                                    # ==================================================
                                    hovertemplate=(
                                        "<b>ระดับผลสัมฤทธิ์:</b> "
                                        + class_name
                                        + "<br>"
                                        "<b>การขาดเรียน:</b> "
                                        + absence_type
                                        + "<br>"
                                        "<b>ร้อยละ:</b> "
                                        + f"{percentage:.1f}%"
                                        + "<br>"
                                        "<b>จำนวนนักเรียน:</b> "
                                        + f"{student_count:,} คน"
                                        + "<extra></extra>"
                                    ),
                                    showlegend=show_legend
                                ),
                                row=1,
                                col=col_index
                            )
                        # ==================================================
                        # ⭐ แกน X
                        # แสดง L / M / H
                        # โดยอยู่กึ่งกลางของแท่ง 2 แท่ง
                        # ==================================================
                        fig1.update_xaxes(
                            tickmode="array",
                            tickvals=[
                                0.5,
                                2.5,
                                4.5
                            ],
                            ticktext=[
                                "L",
                                "M",
                                "H"
                            ],
                            tickfont=dict(
                                size=13,
                                color="#64748B"
                            ),
                            showticklabels=True,
                            title_text="",
                            showgrid=False,
                            zeroline=False,
                            row=1,
                            col=col_index
                        )
                        # ==================================================
                        # ⭐ แกน Y
                        # แสดงเป็นร้อยละ
                        # ==================================================
                        max_percentage = (
                            absence_df["Percentage"]
                            .max()
                        )
                        y_max = max(
                            100,
                            max_percentage * 1.15
                        )
                        fig1.update_yaxes(
                            title_text=(
                                "ร้อยละ (%)"
                                if col_index == 1
                                else ""
                            ),
                            gridcolor="#E2E8F0",
                            rangemode="tozero",
                            range=[
                                0,
                                y_max
                            ],
                            showgrid=True,
                            ticksuffix="%",
                            tickformat=".0f",
                            row=1,
                            col=col_index
                        )
                    # ========================================================
                    # 📝 ชื่อภาคเรียน
                    # อยู่เหนือ "การขาดเรียน"
                    # ========================================================
                    for col_index, semester in enumerate(
                        available_semesters,
                        start=1
                    ):
                        axis_name = (
                            "xaxis"
                            if col_index == 1
                            else f"xaxis{col_index}"
                        )
                        x_domain = fig1.layout[
                            axis_name
                        ].domain
                        x_center = (
                            x_domain[0]
                            + x_domain[1]
                        ) / 2
                        fig1.add_annotation(
                            x=x_center,
                            y=-0.18,
                            xref="paper",
                            yref="paper",
                            text=semester,
                            showarrow=False,
                            font=dict(
                                size=13,
                                color="#64748B"
                            ),
                            xanchor="center",
                            yanchor="middle"
                        )
                    # ========================================================
                    # 📝 "การขาดเรียน"
                    # แสดงเพียง 1 ครั้ง ตรงกลาง
                    # ========================================================
                    fig1.add_annotation(
                        x=0.5,    
                        y=-0.25,
                        xref="paper",
                        yref="paper",
                        text="การขาดเรียน",
                        showarrow=False,
                        font=dict(
                            size=13,
                            color="#64748B"
                        ),
                        xanchor="center",
                        yanchor="middle"
                    )
                    # ========================================================
                    # 🎨 รูปแบบกราฟ
                    # ========================================================
                    fig1.update_layout(
                        height=390,
                        barmode="group",
                        margin=dict(
                            l=55,
                            r=20,
                            t=55,
                            b=125
                        ),
                        legend=dict(
                            orientation="h",
                            y=-0.34,
                            x=0.5,
                            xanchor="center",
                            yanchor="top",
                            title_text=""
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    # ========================================================
                    # แสดงกราฟ
                    # ========================================================
                    st.plotly_chart(
                        fig1,
                        use_container_width=True,
                        config={
                            "displayModeBar": False
                        },
                        key="chart_absence_academic"
                    )
                    # ============================================================
                    # 📊 กล่องสรุปกราฟที่ 1 : แสดงเฉพาะระดับต่ำและระดับสูง
                    # ============================================================
                    if not absence_df.empty:
                        insight_parts = []
                        # แสดงเฉพาะระดับต่ำและระดับสูง
                        class_order = ["L", "H"]
                        class_name = {
                            "L": "ระดับต่ำ (L)",
                            "H": "ระดับสูง (H)"
                        }
                        # --------------------------------------------------------
                        # สร้างข้อความจากข้อมูลเดียวกับกราฟ
                        # --------------------------------------------------------
                        for semester in ["ภาคเรียนที่ 1", "ภาคเรียนที่ 2"]:
                            semester_df = absence_df[
                                absence_df["Semester_Label"] == semester
                            ].copy()
                            if semester_df.empty:
                                continue
                            semester_text = []
                            for class_code in class_order:
                                class_df = semester_df[
                                    semester_df["Class"] == class_code
                                ].copy()
                                if class_df.empty:
                                    continue
                                # เรียงเปอร์เซ็นต์จากน้อยไปมาก
                                class_df = class_df.sort_values(
                                    "Percentage",
                                    ascending=True
                                )
                                absence_items = []
                                for _, row in class_df.iterrows():
                                    absence_label = str(row["Absence_Label"])
                                    percentage = float(row["Percentage"])
                                    absence_items.append(
                                        f"{absence_label} <b>{percentage:.1f}%</b> "
                                    )
                                if absence_items:
                                    # ใช้ HTML เป็นบรรทัดเดียว
                                    text = (
                                        f'<div style="margin-bottom:8px; '
                                        f'padding:8px 10px; '
                                        f'background:#FFFFFF; '
                                        f'border-radius:8px;">'
                                        f'<b style="color:#1E3A5F;">'
                                        f'{class_name[class_code]}'
                                        f'</b>'
                                        f'<span style="color:#475569;">'
                                        f'{absence_items[0]}<br>'
                                        f'{absence_items[1] if len(absence_items) > 1 else ""}'
                                        f'</span>'
                                        f'</div>'
                                    )
                                    semester_text.append(text)

                            if semester_text:
                                # HTML เป็นบรรทัดเดียวเช่นกัน
                                semester_block = (
                                    f'<div style="'
                                    f'flex:1; '
                                    f'min-width:0; '
                                    f'padding:10px 12px; '
                                    f'background:#FFFFFF; '
                                    f'border:1px solid #E2E8F0; '
                                    f'border-radius:10px;'
                                    f'">'
                                    f'<div style="'
                                    f'font-weight:700; '
                                    f'color:#0A2540; '
                                    f'font-size:15px; '
                                    f'margin-bottom:8px;'
                                    f'">'
                                    f'{semester}'
                                    f'</div>'
                                    f'{"".join(semester_text)}'
                                    f'</div>'
                                )
                                insight_parts.append(semester_block)

                    # --------------------------------------------------------
                    # แสดงกล่องคำอธิบาย
                    # --------------------------------------------------------
                    st.markdown(
                        f'''
                        <div style="
                            background-color:#F8FAFC;
                            border:1px solid #CBD5E1;
                            border-left:5px solid #1E3A5F;
                            border-radius:14px;
                            padding:16px 18px;
                            margin-top:14px;
                            margin-bottom:10px;
                            box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                            font-weight:700;
                            color:#0A2540;
                            font-size:16px; 
                            margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                            display:flex;
                            gap:16px;
                            align-items:stretch;
                            color:#334155;
                            font-size:14px;
                            line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                            margin-top:10px;
                            padding-top:8px;
                            border-top:1px solid #E2E8F0;
                            color:#64748B;
                            font-size:13px;
                            line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าร้อยละแสดงสัดส่วนของผู้เรียนภายใน
                        แต่ละระดับผลสัมฤทธิ์ทางการเรียนของแต่ละภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            # ============================================================
            # 2️⃣ การยกมือตอบคำถาม
            # ============================================================
            with st.container(border=True):

                st.markdown(
                    "### 2️⃣ การยกมือตอบคำถามที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )

                fig2, df2 = (
                    create_behavior_effect_chart(
                        behavior_df,
                        "raisedhands",
                        "ค่าเฉลี่ยการยกมือตอบคำถาม",
                        "จำนวนครั้ง"
                    )
                )
                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                    key="chart_raisedhands"
                )
                # ========================================================
                # 📊 กล่องสรุปกราฟที่ 2 : การยกมือตอบคำถาม
                # ========================================================
                if not df2.empty:
                    insight_parts = []
                    # ----------------------------------------------------
                    # แสดงเฉพาะระดับต่ำและระดับสูง
                    # ----------------------------------------------------
                    class_order = [
                        "ระดับต่ำ",
                        "ระดับสูง"
                    ]
                    class_name = {
                        "ระดับต่ำ": "ระดับต่ำ",
                        "ระดับสูง": "ระดับสูง"
                    }
                    # ----------------------------------------------------
                    # สร้างข้อมูลแยกตามภาคเรียน
                    # ----------------------------------------------------
                    for semester in [
                        "ภาคเรียนที่ 1",
                        "ภาคเรียนที่ 2"
                    ]:
                        semester_df = df2[
                            df2["Semester_Label"] == semester
                        ].copy()
                        if semester_df.empty:
                            continue
                        semester_text = []
                        # ------------------------------------------------
                        # ระดับผลสัมฤทธิ์ L และ H
                        # ------------------------------------------------
                        for class_code in class_order:
                            class_df = semester_df[
                                semester_df["Class_Label"] == class_code
                            ].copy()

                            if class_df.empty:
                                continue
                            row = class_df.iloc[0]

                            avg_value = float(
                                row["raisedhands"]
                            )
                            text = (
                                f'<div style="'
                                f'margin-bottom:8px; '
                                f'padding:8px 10px; '
                                f'background:#FFFFFF; '
                                f'border-radius:8px;'
                                f'">'
                                f'<b style="color:#1E3A5F;">'
                                f'{class_name[class_code]}'
                                f'</b>'
                                f'<span style="color:#475569;">'
                                f'ค่าเฉลี่ยการยกมือตอบคำถาม '
                                f'<b>{avg_value:.2f}</b> ครั้ง '
                                f'</span>'
                                f'</div>'
                            )
                            semester_text.append(text)
                        # ------------------------------------------------
                        # สร้างกรอบของแต่ละภาคเรียน
                        # ------------------------------------------------
                        if semester_text:
                            semester_block = (
                                f'<div style="'
                                f'flex:1; '
                                f'min-width:0; '
                                f'padding:10px 12px; '
                                f'background:#FFFFFF; '
                                f'border:1px solid #E2E8F0; '
                                f'border-radius:10px;'
                                f'">'
                                f'<div style="'
                                f'font-weight:700; '
                                f'color:#0A2540; '
                                f'font-size:15px; '
                                f'margin-bottom:8px;'
                                f'">'
                                f'{semester}'
                                f'</div>'
                                f'{"".join(semester_text)}'
                                f'</div>'
                            )
                            insight_parts.append(
                                semester_block
                            )
                    # ====================================================
                    # 📊 แสดงกล่องคำอธิบายเพียง 1 กรอบ
                    # ====================================================
                    st.markdown(
                        f'''
                        <div style="
                        background-color:#F8FAFC;
                        border:1px solid #CBD5E1;
                        border-left:5px solid #1E3A5F;
                        border-radius:14px;
                        padding:16px 18px;
                        margin-top:14px;
                        margin-bottom:10px;
                        box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                        font-weight:700;
                        color:#0A2540;
                        font-size:16px;
                        margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                        display:flex;
                        gap:16px;
                        align-items:stretch;
                        color:#334155;
                        font-size:14px;
                        line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                        margin-top:10px;
                        padding-top:8px;
                        border-top:1px solid #E2E8F0;
                        color:#64748B;
                        font-size:13px;
                        line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าที่แสดงเป็นค่าเฉลี่ยจำนวนครั้งการยกมือตอบคำถาม
                        ของผู้เรียนในแต่ละระดับผลสัมฤทธิ์ทางการเรียน
                        แยกตามภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            # ============================================================
            # 3️⃣ การเข้าดูสื่อ
            # ============================================================
            with st.container(border=True):
                st.markdown(
                    "### 3️⃣ การเข้าดูสื่อการเรียนที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )
                fig3, df3 = (
                    create_behavior_effect_chart(
                        behavior_df,
                        "VisITedResources",
                        "ค่าเฉลี่ยการเข้าดูสื่อ",
                        "จำนวนครั้ง"
                    )
                )
                st.plotly_chart(
                    fig3,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                    key="chart_resources"
                )
                #========================================================
                # 📊 กล่องสรุปกราฟที่ 3 : การเข้าดูสื่อ
                # ========================================================
                if not df3.empty:
                    insight_parts = []
                    # ----------------------------------------------------
                    # แสดงเฉพาะระดับต่ำและระดับสูง
                    # ----------------------------------------------------
                    class_order = [
                        "ระดับต่ำ",
                        "ระดับสูง"
                    ]
                    class_name = {
                        "ระดับต่ำ": "ระดับต่ำ",
                        "ระดับสูง": "ระดับสูง"
                    }
                    # ----------------------------------------------------
                    # สร้างข้อมูลแยกตามภาคเรียน
                    # ----------------------------------------------------
                    for semester in [
                        "ภาคเรียนที่ 1",
                        "ภาคเรียนที่ 2"
                    ]:
                        semester_df = df3[
                            df3["Semester_Label"] == semester
                        ].copy()
                        if semester_df.empty:
                            continue
                        semester_text = []
                        # ------------------------------------------------
                        # ระดับผลสัมฤทธิ์ L และ H
                        # ------------------------------------------------
                        for class_code in class_order:
                            class_df = semester_df[
                                semester_df["Class_Label"] == class_code
                            ].copy()

                            if class_df.empty:
                                continue
                            row = class_df.iloc[0]

                            avg_value = float(
                                row["VisITedResources"]
                            )
                            text = (
                                f'<div style="'
                                f'margin-bottom:8px; '
                                f'padding:8px 10px; '
                                f'background:#FFFFFF; '
                                f'border-radius:8px;'
                                f'">'
                                f'<b style="color:#1E3A5F;">'
                                f'{class_name[class_code]}'
                                f'</b>'
                                f'<span style="color:#475569;">'
                                f'ค่าเฉลี่ยการเข้าดูสื่อ '
                                f'<b>{avg_value:.2f}</b> ครั้ง '
                                f'</span>'
                                f'</div>'
                            )
                            semester_text.append(text)
                        # ------------------------------------------------
                        # สร้างกรอบของแต่ละภาคเรียน
                        # ------------------------------------------------
                        if semester_text:
                            semester_block = (
                                f'<div style="'
                                f'flex:1; '
                                f'min-width:0; '
                                f'padding:10px 12px; '
                                f'background:#FFFFFF; '
                                f'border:1px solid #E2E8F0; '
                                f'border-radius:10px;'
                                f'">'
                                f'<div style="'
                                f'font-weight:700; '
                                f'color:#0A2540; '
                                f'font-size:15px; '
                                f'margin-bottom:8px;'
                                f'">'
                                f'{semester}'
                                f'</div>'
                                f'{"".join(semester_text)}'
                                f'</div>'
                            )
                            insight_parts.append(
                                semester_block
                            )
                    # ====================================================
                    # 📊 แสดงกล่องคำอธิบายเพียง 1 กรอบ
                    # ====================================================
                    st.markdown(
                        f'''
                        <div style="
                        background-color:#F8FAFC;
                        border:1px solid #CBD5E1;
                        border-left:5px solid #1E3A5F;
                        border-radius:14px;
                        padding:16px 18px;
                        margin-top:14px;
                        margin-bottom:10px;
                        box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                        font-weight:700;
                        color:#0A2540;
                        font-size:16px;
                        margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                        display:flex;
                        gap:16px;
                        align-items:stretch;
                        color:#334155;
                        font-size:14px;
                        line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                        margin-top:10px;
                        padding-top:8px;
                        border-top:1px solid #E2E8F0;
                        color:#64748B;
                        font-size:13px;
                        line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าที่แสดงเป็นค่าเฉลี่ยจำนวนครั้งการเข้าดูสื่อ
                        ของผู้เรียนในแต่ละระดับผลสัมฤทธิ์ทางการเรียน
                        แยกตามภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            # ============================================================
            # 4️⃣ การเข้าดูประกาศ
            # ============================================================
            with st.container(border=True):
                st.markdown(
                    "### 4️⃣ การเข้าดูประกาศที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )
                fig4, df4 = (
                    create_behavior_effect_chart(
                        behavior_df,
                        "AnnouncementsView",
                        "ค่าเฉลี่ยการเข้าดูประกาศ",
                        "จำนวนครั้ง"
                    )
                )
                st.plotly_chart(
                    fig4,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                    key="chart_announcements"
                )
                #========================================================
                # 📊 กล่องสรุปกราฟที่ 4 : การเข้าดูประกาส
                # ========================================================
                if not df4.empty:
                    insight_parts = []
                    # ----------------------------------------------------
                    # แสดงเฉพาะระดับต่ำและระดับสูง
                    # ----------------------------------------------------
                    class_order = [
                        "ระดับต่ำ",
                        "ระดับสูง"
                    ]
                    class_name = {
                        "ระดับต่ำ": "ระดับต่ำ",
                        "ระดับสูง": "ระดับสูง"
                    }
                    # ----------------------------------------------------
                    # สร้างข้อมูลแยกตามภาคเรียน
                    # ----------------------------------------------------
                    for semester in [
                        "ภาคเรียนที่ 1",
                        "ภาคเรียนที่ 2"
                    ]:
                        semester_df = df4[
                            df4["Semester_Label"] == semester
                        ].copy()
                        if semester_df.empty:
                            continue
                        semester_text = []
                        # ------------------------------------------------
                        # ระดับผลสัมฤทธิ์ L และ H
                        # ------------------------------------------------
                        for class_code in class_order:
                            class_df = semester_df[
                                semester_df["Class_Label"] == class_code
                            ].copy()                
                            if class_df.empty:
                                continue
                            row = class_df.iloc[0]            
                            avg_value = float(
                                row["AnnouncementsView"]
                            )
                            text = (
                                f'<div style="'
                                f'margin-bottom:8px; '
                                f'padding:8px 10px; '
                                f'background:#FFFFFF; '
                                f'border-radius:8px;'
                                f'">'
                                f'<b style="color:#1E3A5F;">'
                                f'{class_name[class_code]}'
                                f'</b>'
                                f'<span style="color:#475569;">'
                                f'ค่าเฉลี่ยการเข้าดูประกาศ '
                                f'<b>{avg_value:.2f}</b> ครั้ง '
                                f'</span>'
                                f'</div>'
                            )
                            semester_text.append(text)
                        # ------------------------------------------------
                        # สร้างกรอบของแต่ละภาคเรียน
                        # ------------------------------------------------
                        if semester_text:
                            semester_block = (
                                f'<div style="'
                                f'flex:1; '
                                f'min-width:0; '
                                f'padding:10px 12px; '
                                f'background:#FFFFFF; '
                                f'border:1px solid #E2E8F0; '
                                f'border-radius:10px;'
                                f'">'
                                f'<div style="'
                                f'font-weight:700; '
                                f'color:#0A2540; '
                                f'font-size:15px; '
                                f'margin-bottom:8px;'
                                f'">'
                                f'{semester}'
                                f'</div>'
                                f'{"".join(semester_text)}'
                                f'</div>'
                            )
                            insight_parts.append(
                                semester_block
                            )
                    # ====================================================
                    # 📊 แสดงกล่องคำอธิบายเพียง 1 กรอบ
                    # ====================================================
                    st.markdown(
                        f'''
                        <div style="
                        background-color:#F8FAFC;
                        border:1px solid #CBD5E1;
                        border-left:5px solid #1E3A5F;
                        border-radius:14px;
                        padding:16px 18px;
                        margin-top:14px;
                        margin-bottom:10px;
                        box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                        font-weight:700;
                        color:#0A2540;
                        font-size:16px;
                        margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                        display:flex;
                        gap:16px;
                        align-items:stretch;
                        color:#334155;
                        font-size:14px;
                        line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                        margin-top:10px;
                        padding-top:8px;
                        border-top:1px solid #E2E8F0;
                        color:#64748B;
                        font-size:13px;
                        line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าที่แสดงเป็นค่าเฉลี่ยจำนวนครั้งการเข้าดูประกาศ
                        ของผู้เรียนในแต่ละระดับผลสัมฤทธิ์ทางการเรียน
                        แยกตามภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            # ============================================================
            # 5️⃣ การร่วมอภิปราย
            # ============================================================
            with st.container(border=True):
                st.markdown(
                    "### 5️⃣ การร่วมอภิปรายที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )
                fig5, df5 = (
                    create_behavior_effect_chart(
                        behavior_df,
                        "Discussion",
                        "ค่าเฉลี่ยการร่วมอภิปราย",
                        "จำนวนครั้ง"
                    )
                )
                st.plotly_chart(
                    fig5,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                    key="chart_discussion"
                )
                #========================================================
                # 📊 กล่องสรุปกราฟที่ 5 : การร่วมอภิปราย
                # ========================================================
                if not df5.empty:
                    insight_parts = []
                    # ----------------------------------------------------
                    # แสดงเฉพาะระดับต่ำและระดับสูง
                    # ----------------------------------------------------
                    class_order = [
                        "ระดับต่ำ",
                        "ระดับสูง"
                    ]
                    class_name = {
                        "ระดับต่ำ": "ระดับต่ำ",
                        "ระดับสูง": "ระดับสูง"
                    }
                    # ----------------------------------------------------
                    # สร้างข้อมูลแยกตามภาคเรียน
                    # ----------------------------------------------------
                    for semester in [
                        "ภาคเรียนที่ 1",
                        "ภาคเรียนที่ 2"
                    ]:
                        semester_df = df5[
                            df5["Semester_Label"] == semester
                        ].copy()
                        if semester_df.empty:
                            continue
                        semester_text = []
                        # ------------------------------------------------
                        # ระดับผลสัมฤทธิ์ L และ H
                        # ------------------------------------------------
                        for class_code in class_order:
                            class_df = semester_df[
                                semester_df["Class_Label"] == class_code
                            ].copy()                
                            if class_df.empty:
                                continue
                            row = class_df.iloc[0]            
                            avg_value = float(
                                row["Discussion"]
                            )
                            text = (
                                f'<div style="'
                                f'margin-bottom:8px; '
                                f'padding:8px 10px; '
                                f'background:#FFFFFF; '
                                f'border-radius:8px;'
                                f'">'
                                f'<b style="color:#1E3A5F;">'
                                f'{class_name[class_code]}'
                                f'</b>'
                                f'<span style="color:#475569;">'
                                f'ค่าเฉลี่ยการร่วมอภิปราย '
                                f'<b>{avg_value:.2f}</b> ครั้ง '
                                f'</span>'
                                f'</div>'
                            )
                            semester_text.append(text)
                        # ------------------------------------------------
                        # สร้างกรอบของแต่ละภาคเรียน
                        # ------------------------------------------------
                        if semester_text:
                            semester_block = (
                                f'<div style="'
                                f'flex:1; '
                                f'min-width:0; '
                                f'padding:10px 12px; '
                                f'background:#FFFFFF; '
                                f'border:1px solid #E2E8F0; '
                                f'border-radius:10px;'
                                f'">'
                                f'<div style="'
                                f'font-weight:700; '
                                f'color:#0A2540; '
                                f'font-size:15px; '
                                f'margin-bottom:8px;'
                                f'">'
                                f'{semester}'
                                f'</div>'
                                f'{"".join(semester_text)}'
                                f'</div>'
                            )
                            insight_parts.append(
                                semester_block
                            )
                    # ====================================================
                    # 📊 แสดงกล่องคำอธิบายเพียง 1 กรอบ
                    # ====================================================
                    st.markdown(
                        f'''
                        <div style="
                        background-color:#F8FAFC;
                        border:1px solid #CBD5E1;
                        border-left:5px solid #1E3A5F;
                        border-radius:14px;
                        padding:16px 18px;
                        margin-top:14px;
                        margin-bottom:10px;
                        box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                        font-weight:700;
                        color:#0A2540;
                        font-size:16px;
                        margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                        display:flex;
                        gap:16px;
                        align-items:stretch;
                        color:#334155;
                        font-size:14px;
                        line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                        margin-top:10px;
                        padding-top:8px;
                        border-top:1px solid #E2E8F0;
                        color:#64748B;
                        font-size:13px;
                        line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าที่แสดงเป็นค่าเฉลี่ยจำนวนครั้งการร่วมอภิปราย
                        ของผู้เรียนในแต่ละระดับผลสัมฤทธิ์ทางการเรียน
                        แยกตามภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
            # ============================================================
            # 6️⃣ ภาพรวมการมีส่วนร่วมทั้งหมด
            # ============================================================
            with st.container(border=True):
                st.markdown(
                    "### 6️⃣ ภาพรวมการมีส่วนร่วมทั้งหมดส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน"
                )
                fig6, df6 = (
                    create_behavior_effect_chart(
                        behavior_df,
                        "ParticipationScore",
                        "คะแนนการมีส่วนร่วมรวม",
                        "คะแนน"
                    )
                )
                st.plotly_chart(
                    fig6,
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                    key="chart_participation"
                )
                #========================================================
                # 📊 กล่องสรุปกราฟที่ 6 : ภาพรวมทุกกราฟ
                # ========================================================
                if not df6.empty:
                    insight_parts = []
                    # ----------------------------------------------------
                    # แสดงเฉพาะระดับต่ำและระดับสูง
                    # ----------------------------------------------------
                    class_order = [
                        "ระดับต่ำ",
                        "ระดับสูง"
                    ]
                    class_name = {
                        "ระดับต่ำ": "ระดับต่ำ",
                        "ระดับสูง": "ระดับสูง"
                    }
                    # ----------------------------------------------------
                    # สร้างข้อมูลแยกตามภาคเรียน
                    # ----------------------------------------------------
                    for semester in [
                        "ภาคเรียนที่ 1",
                        "ภาคเรียนที่ 2"
                    ]:
                        semester_df = df6[
                            df6["Semester_Label"] == semester
                        ].copy()
                        if semester_df.empty:
                            continue
                        semester_text = []
                        # ------------------------------------------------
                        # ระดับผลสัมฤทธิ์ L และ H
                        # ------------------------------------------------
                        for class_code in class_order:
                            class_df = semester_df[
                                semester_df["Class_Label"] == class_code
                            ].copy()                
                            if class_df.empty:
                                continue
                            row = class_df.iloc[0]            
                            avg_value = float(
                                row["ParticipationScore"]
                            )
                            text = (
                                f'<div style="'
                                f'margin-bottom:8px; '
                                f'padding:8px 10px; '
                                f'background:#FFFFFF; '
                                f'border-radius:8px;'
                                f'">'
                                f'<b style="color:#1E3A5F;">'
                                f'{class_name[class_code]}'
                                f'</b>'
                                f'<span style="color:#475569;">'
                                f'ค่าเฉลี่ยคะแนนการมีส่วนร่วมรวม '
                                f'<b>{avg_value:.2f}</b> ครั้ง '
                                f'</span>'
                                f'</div>'
                            )
                            semester_text.append(text)
                        # ------------------------------------------------
                        # สร้างกรอบของแต่ละภาคเรียน
                        # ------------------------------------------------
                        if semester_text:
                            semester_block = (
                                f'<div style="'
                                f'flex:1; '
                                f'min-width:0; '
                                f'padding:10px 12px; '
                                f'background:#FFFFFF; '
                                f'border:1px solid #E2E8F0; '
                                f'border-radius:10px;'
                                f'">'
                                f'<div style="'
                                f'font-weight:700; '
                                f'color:#0A2540; '
                                f'font-size:15px; '
                                f'margin-bottom:8px;'
                                f'">'
                                f'{semester}'
                                f'</div>'
                                f'{"".join(semester_text)}'
                                f'</div>'
                            )
                            insight_parts.append(
                                semester_block
                            )
                    # ====================================================
                    # 📊 แสดงกล่องคำอธิบายเพียง 1 กรอบ
                    # ====================================================
                    st.markdown(
                        f'''
                        <div style="
                        background-color:#F8FAFC;
                        border:1px solid #CBD5E1;
                        border-left:5px solid #1E3A5F;
                        border-radius:14px;
                        padding:16px 18px;
                        margin-top:14px;
                        margin-bottom:10px;
                        box-shadow:0px 2px 5px rgba(0,0,0,0.04);
                        ">
                        <div style="
                        font-weight:700;
                        color:#0A2540;
                        font-size:16px;
                        margin-bottom:12px;
                        ">
                        📊 คำอธิบาย
                        </div>
                        <div style="
                        display:flex;
                        gap:16px;
                        align-items:stretch;
                        color:#334155;
                        font-size:14px;
                        line-height:1.8;
                        ">
                        {"".join(insight_parts)}
                        </div>
                        <div style="
                        margin-top:10px;
                        padding-top:8px;
                        border-top:1px solid #E2E8F0;
                        color:#64748B;
                        font-size:13px;
                        line-height:1.7;
                        ">
                        💡 <b>หมายเหตุ:</b>
                        ค่าที่แสดงเป็นค่าเฉลี่ยคะแนนการมีส่วนร่วมรวม
                        ของผู้เรียนในแต่ละระดับผลสัมฤทธิ์ทางการเรียน
                        แยกตามภาคเรียน
                        </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
        else:
            st.info("ไม่พบข้อมูลผลสัมฤทธิ์ทางการเรียนสำหรับแสดงผล")

# ================================================================
# 📚 พฤติกรรมการเรียนรู้
# ================================================================
elif menu == "พฤติกรรมการเรียนรู้":
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h1 style="
                color: #0A2540;
                font-size: 30px;
                font-weight: 700;
                margin-bottom: 6px;
            ">
                📚 พฤติกรรมการเรียนรู้
            </h1>
        <div style="
                color: #64748B;
                font-size: 16px;
        ">
                ภาพรวมและรูปแบบพฤติกรรมการเรียนรู้ของผู้เรียน
                จำแนกตามข้อมูลในชุดข้อมูล
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ================================================================
    # 📦 ตรวจสอบคอลัมน์ที่จำเป็น
    # ================================================================

    behavior_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion"
    ]

    required_cols = [
        "StageID",
        "GradeID",
        "Semester",
        "Topic",
        "Class"
    ] + behavior_cols

    missing_cols = [
        col for col in required_cols
        if col not in filtered_df.columns
    ]

    if missing_cols:

        st.error(
            "ไม่พบคอลัมน์ที่จำเป็นในข้อมูล: "
            + ", ".join(missing_cols)
        )

        st.stop()

    # ================================================================
    # 🔄 เตรียมข้อมูล
    # ================================================================

    behavior_df = filtered_df.copy()

    for col in behavior_cols:
        behavior_df[col] = pd.to_numeric(
            behavior_df[col],
            errors="coerce"
        )

    behavior_df = behavior_df.dropna(
        subset=behavior_cols
    ).copy()

    # ================================================================
    # 🏷️ Mapping รายวิชา
    # ================================================================

    topic_map = {
        "IT": "เทคโนโลยี",
        "Math": "คณิตศาสตร์",
        "Science": "วิทยาศาสตร์",
        "English": "ภาษาอังกฤษ",
        "French": "ภาษาฝรั่งเศส",
        "Spanish": "ภาษาสเปน",
        "Arabic": "ภาษาอาหรับ",
        "Biology": "ชีววิทยา",
        "Chemistry": "เคมี",
        "Geology": "ธรณีวิทยา",
        "History": "ประวัติศาสตร์",
        "Quran": "อัลกุรอาน"
    }

    # ================================================================
    # 🏫 Mapping ช่วงชั้น
    # ================================================================

    stage_map = {
        "lowerlevel": "ประถมศึกษา",
        "MiddleSchool": "มัธยมต้น",
        "HighSchool": "มัธยมปลาย"
    }

    # ================================================================
    # 🎓 Mapping ระดับชั้น
    # ================================================================

    grade_map = {
        "G-01": "ป.1",
        "G-02": "ป.2",
        "G-03": "ป.3",
        "G-04": "ป.4",
        "G-05": "ป.5",
        "G-06": "ป.6",
        "G-07": "ม.1",
        "G-08": "ม.2",
        "G-09": "ม.3",
        "G-10": "ม.4",
        "G-11": "ม.5",
        "G-12": "ม.6"
    }

    # ================================================================
    # 📅 Mapping ภาคเรียน
    # ================================================================

    semester_map = {
        "F": "ภาคเรียนที่ 1",
        "S": "ภาคเรียนที่ 2"
    }

    # ================================================================
    # 🏆 Mapping ระดับผลการเรียนรู้
    # ================================================================

    class_map = {
        "L": "ระดับต่ำ",
        "M": "ระดับปานกลาง",
        "H": "ระดับสูง"
    }

    # ================================================================
    # 🔤 สร้างชื่อภาษาไทย
    # ================================================================

    behavior_df["Topic_TH"] = (
        behavior_df["Topic"]
        .astype(str)
        .map(topic_map)
        .fillna(
            behavior_df["Topic"].astype(str)
        )
    )

    behavior_df["Stage_TH"] = (
        behavior_df["StageID"]
        .astype(str)
        .map(stage_map)
        .fillna(
            behavior_df["StageID"].astype(str)
        )
    )

    behavior_df["Grade_TH"] = (
        behavior_df["GradeID"]
        .astype(str)
        .map(grade_map)
        .fillna(
            behavior_df["GradeID"].astype(str)
        )
    )

    behavior_df["Semester_Label"] = (
        behavior_df["Semester"]
        .astype(str)
        .map(semester_map)
        .fillna(
            behavior_df["Semester"].astype(str)
        )
    )

    behavior_df["Class_TH"] = (
        behavior_df["Class"]
        .astype(str)
        .map(class_map)
        .fillna(
            behavior_df["Class"].astype(str)
        )
    )

    # ================================================================
    # 📊 คะแนนเฉลี่ยการมีส่วนร่วม
    # ================================================================

    behavior_df["ParticipationScore"] = (
        behavior_df[behavior_cols].mean(axis=1)
    )

    # ================================================================
    # 🔢 กำหนดลำดับ
    # ================================================================

    grade_order = [
        "ป.1",
        "ป.2",
        "ป.3",
        "ป.4",
        "ป.5",
        "ป.6",
        "ม.1",
        "ม.2",
        "ม.3",
        "ม.4",
        "ม.5",
        "ม.6"
    ]

    stage_order = [
        "ประถมศึกษา",
        "มัธยมต้น",
        "มัธยมปลาย"
    ]

    semester_order = [
        "ภาคเรียนที่ 1",
        "ภาคเรียนที่ 2"
    ]

    class_order = [
        "ระดับต่ำ",
        "ระดับปานกลาง",
        "ระดับสูง"
    ]
    # ================================================================
    # 📌 1. ภาพรวมพฤติกรรมการเรียนรู้
    # ================================================================

    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 10px;
            margin-bottom: 15px;
        ">
            📌 ภาพรวมพฤติกรรมการเรียนรู้
        </h2>
        """,
        unsafe_allow_html=True
    )

   # ------------------------------------------------
    # KPI
    # ------------------------------------------------
    # ใช้รูปแบบการ์ดเดียวกับ KPI ด้านบน
    kpi_data = [
        (
            "🙋",
            "การยกมือตอบคำถาม",
            "raisedhands"
        ),
        (
            "📖",
            "การเข้าดูสื่อการเรียน",
            "VisITedResources"
        ),
        (
            "📢",
            "การดูประกาศ",
            "AnnouncementsView"
        ),
        (
            "💬",
            "การอภิปราย",
            "Discussion"
        )
    ]

    kpi_cols = st.columns(4)
    for i, (icon, title, col_name) in enumerate(kpi_data):
        value = behavior_df[col_name].mean()
        with kpi_cols[i]:
            st.markdown(
                f"""
                <div style="
                    background-color:#FFFFFF;
                    border:1.5px solid #1E3A5F;
                    border-radius:30px;
                    padding:12px 16px;
                    display:flex;
                    align-items:center;
                    gap:12px;
                    box-shadow:0px 2px 5px rgba(0,0,0,0.05);
                    min-height:70px;
                ">
                    <div style="
                        width:44px;
                        height:44px;
                        border-radius:50%;
                        background-color:#E0F2FE;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:20px;
                        flex-shrink:0;
                    ">
                        {icon}
                    </div>
                    <div>
                        <div style="
                            font-size:0.85rem;
                            color:#1E3A5F;
                            font-weight:600;
                            line-height:1.2;
                        ">
                            {title}
                        </div>
                        <div style="
                            font-size:1.1rem;
                            font-weight:bold;
                            color:#0A2540;
                            margin-top:2px;
                        ">
                            {value:.2f}
                        </div>
                        <div style="
                            font-size:0.75rem;
                            color:#64748B;
                            font-weight:500;
                        ">
                            คะแนนเฉลี่ย (0–100)
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ================================================================
    # 📊 2. เปรียบเทียบพฤติกรรม 4 ด้าน
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            📊 คะแนนเฉลี่ยพฤติกรรมการเรียนรู้ 4 ด้าน
        </h2>

        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            เปรียบเทียบระดับการมีส่วนร่วมของผู้เรียนในแต่ละพฤติกรรม
            เพื่อดูว่าพฤติกรรมด้านใดมีค่าเฉลี่ยสูงหรือต่ำกว่าด้านอื่น
        </div>
        """,
        unsafe_allow_html=True
    )

    overall_df = pd.DataFrame({
        "พฤติกรรม": [
            "การยกมือตอบคำถาม",
            "การเข้าดูสื่อการเรียน",
            "การดูประกาศ",
            "การอภิปราย"
        ],
        "คะแนนเฉลี่ย": [
            behavior_df["raisedhands"].mean(),
            behavior_df["VisITedResources"].mean(),
            behavior_df["AnnouncementsView"].mean(),
            behavior_df["Discussion"].mean()
        ]
    })
    overall_df = overall_df.sort_values(
        "คะแนนเฉลี่ย",
        ascending=True
    ).reset_index(drop=True)
    bar_colors = [
        "#C9A66B",  # คะแนนต่ำสุด
        "#B88A4A",
        "#A06F2A",
        "#7A4E16"   # คะแนนสูงสุด
    ]
    with st.container(border=True):
        fig_overall = px.bar(
            overall_df,
            x="พฤติกรรม",
            y="คะแนนเฉลี่ย",
            text="คะแนนเฉลี่ย",
            labels={
                "พฤติกรรม": "พฤติกรรมการเรียนรู้",
                "คะแนนเฉลี่ย": "คะแนนเฉลี่ย (0–100)"
            }
        )

        fig_overall.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_color=bar_colors
        )

        fig_overall.update_layout(
            height=430,
            yaxis=dict(
                range=[0, 100],
                title="คะแนนเฉลี่ย (0–100)",
                gridcolor="#E2E8F0"
            ),
            xaxis=dict(
                title="พฤติกรรมการเรียนรู้",
                showgrid=False
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False
        )

        st.plotly_chart(
            fig_overall,
            use_container_width=True
        )

        highest_behavior = overall_df.loc[
            overall_df["คะแนนเฉลี่ย"].idxmax()
        ]
        lowest_behavior = overall_df.loc[
            overall_df["คะแนนเฉลี่ย"].idxmin()
        ]
        st.markdown(
            f"""
            <div style="
                border-left: 5px solid #3B82F6;
                background: #EFF6FF;
                padding: 15px 18px;
                border-radius: 8px;
                margin-top: 12px;
                margin-bottom: 25px;
            ">
            <div style="
                font-weight: 700;
                color: #1E3A5F;
                margin-bottom: 6px;
            ">🔍 จากข้อมูล</div>
            <div style="
                color: #334155;
                line-height: 1.8;
            ">
                พฤติกรรมที่มีคะแนนเฉลี่ยสูงที่สุด คือ
                <b>{highest_behavior["พฤติกรรม"]}</b>
                ({highest_behavior["คะแนนเฉลี่ย"]:.2f} คะแนน)
                ขณะที่พฤติกรรมที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                <b>{lowest_behavior["พฤติกรรม"]}</b>
                ({lowest_behavior["คะแนนเฉลี่ย"]:.2f} คะแนน)
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================================================================
    # 🏫 3. จำแนกตามช่วงชั้น
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            🏫 คะแนนเฉลี่ยการมีส่วนร่วมจำแนกตามช่วงชั้น
        </h2>

        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            เปรียบเทียบระดับการมีส่วนร่วมของผู้เรียนในแต่ละช่วงชั้น
            และดูความแตกต่างระหว่างภาคเรียนที่ 1 และภาคเรียนที่ 2
        </div>
        """,
        unsafe_allow_html=True
    )

    stage_df = (
        behavior_df
        .groupby(
            [
                "Stage_TH",
                "Semester_Label"
            ],
            as_index=False
        )["ParticipationScore"]
        .mean()
    )

    stage_df["Stage_TH"] = pd.Categorical(
        stage_df["Stage_TH"],
        categories=stage_order,
        ordered=True
    )

    stage_df["Semester_Label"] = pd.Categorical(
        stage_df["Semester_Label"],
        categories=semester_order,
        ordered=True
    )

    stage_df = stage_df.sort_values(
        [
            "Stage_TH",
            "Semester_Label"
        ]
    )
    if not stage_df.empty:
        with st.container(border=True):

            fig_stage = px.bar(
                stage_df,
                x="Stage_TH",
                y="ParticipationScore",
                color="Semester_Label",
                barmode="group",
                text="ParticipationScore",
                category_orders={
                    "Stage_TH": stage_order,
                    "Semester_Label": semester_order
                },
                color_discrete_map={
                    "ภาคเรียนที่ 1": "#35A6DB",
                    "ภาคเรียนที่ 2": "#383AB5"
                },
                labels={
                    "Stage_TH": "ช่วงชั้น",
                    "ParticipationScore": "คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                    "Semester_Label": ""
                }
            )

            fig_stage.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig_stage.update_layout(
                height=450,

                yaxis=dict(
                    range=[0, 100],
                    title="คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                    gridcolor="#E2E8F0"
                ),

                xaxis=dict(
                    title="ช่วงชั้น",
                    showgrid=False
                ),

                plot_bgcolor="white",
                paper_bgcolor="white",

                legend=dict(
                    title_text="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),

                margin=dict(
                    t=80,
                    b=60,
                    l=60,
                    r=30
                )
            )

            st.plotly_chart(
                fig_stage,
                use_container_width=True
            )

            highest_stage = stage_df.loc[
                stage_df["ParticipationScore"].idxmax()
            ]
            lowest_stage = stage_df.loc[
                stage_df["ParticipationScore"].idxmin()
            ]
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid #35A6DB;
                    background: #F0F9FF;
                    padding: 15px 18px;
                    border-radius: 8px;
                    margin-top: 12px;
                    margin-bottom: 25px;
                ">
                <div style="
                    font-weight: 700;
                    color: #1E3A5F;
                    margin-bottom: 6px;
                ">
                    🔍 จากข้อมูล
                </div>
                <div style="
                    color: #334155;
                    line-height: 1.8;
                ">
                    เมื่อจำแนกตามช่วงชั้น พบว่า
                    <b>{highest_stage["Stage_TH"]}</b>
                    ใน{highest_stage["Semester_Label"]}
                    มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด
                    ({highest_stage["ParticipationScore"]:.2f} คะแนน)
                    ขณะที่ค่าต่ำที่สุดพบใน
                    <b>{lowest_stage["Stage_TH"]}</b>
                    ใน{lowest_stage["Semester_Label"]}
                    ({lowest_stage["ParticipationScore"]:.2f} คะแนน)
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ================================================================
    # 🎓 4. จำแนกตามระดับชั้น ป.1 – ม.6
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            🎓 คะแนนเฉลี่ยการมีส่วนร่วมจำแนกตามระดับชั้น
        </h2>
        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            แสดงระดับการมีส่วนร่วมของผู้เรียนตั้งแต่ระดับ ป.1 ถึง ม.6
            และเปรียบเทียบข้อมูลระหว่างภาคเรียน
        </div>
        """,
        unsafe_allow_html=True
    )
    grade_df = (
        behavior_df
        .groupby(
            [
                "Grade_TH",
                "Semester_Label"
            ],
            as_index=False
        )["ParticipationScore"]
        .mean()
    )
    if not grade_df.empty:
        # ============================================================
        # 🔄 เรียงภาคเรียนภายในแต่ละระดับชั้น
        #    จากคะแนนน้อย → มาก
        # ============================================================
        ordered_grade_df = []
        for grade in grade_order:
            temp = grade_df[
                grade_df["Grade_TH"] == grade
            ].copy()
            if not temp.empty:
                temp = temp.sort_values(
                    "ParticipationScore",
                    ascending=True
                )
                ordered_grade_df.append(temp)
        if ordered_grade_df:
            ordered_grade_df = pd.concat(
                ordered_grade_df,
                ignore_index=True
            )
            # ========================================================
            # 📊 สร้างตำแหน่งแท่งกราฟ
            # ========================================================
            x_positions = []
            x_labels = []
            semester_values = []
            tick_positions = []
            tick_labels = []
            current_x = 0
            for grade in grade_order:
                temp = ordered_grade_df[
                    ordered_grade_df["Grade_TH"] == grade
                ].copy()
                if temp.empty:
                    continue
                positions = []
                for _, row in temp.iterrows():
                    x_positions.append(current_x)
                    x_labels.append(grade)
                    semester_values.append(row["Semester_Label"])
                    positions.append(current_x)
                    current_x += 1
                # ตำแหน่งกึ่งกลางของระดับชั้น
                tick_positions.append(
                    sum(positions) / len(positions)
                )
                tick_labels.append(grade)
                # เว้นระยะก่อนระดับชั้นถัดไป
                current_x += 1
            # ========================================================
            # 🔵 ภาคเรียนที่ 1
            # 🟣 ภาคเรียนที่ 2
            # ========================================================
            semester1_x = []
            semester1_y = []
            semester1_text = []
            semester2_x = []
            semester2_y = []
            semester2_text = []
            for x, semester in zip(
                x_positions,
                semester_values
            ):
                row = ordered_grade_df[
                    (
                        ordered_grade_df["Semester_Label"]
                        == semester
                    )
                    &
                    (
                        ordered_grade_df.index
                        == ordered_grade_df[
                            ordered_grade_df["Semester_Label"] == semester
                        ].index[
                            ordered_grade_df[
                                ordered_grade_df["Semester_Label"] == semester
                            ]["ParticipationScore"]
                            .sub(
                                ordered_grade_df.loc[
                                    ordered_grade_df.index,
                                    "ParticipationScore"
                                ]
                            )
                            .abs()
                            .idxmin()
                        ]
                    )
                ]

            # ========================================================
            # ใช้ข้อมูลตามลำดับจริงของ ordered_grade_df
            # ========================================================
            semester1_x = []
            semester1_y = []
            semester1_text = []
            semester2_x = []
            semester2_y = []
            semester2_text = []

            for x, (_, row) in zip(
                x_positions,
                ordered_grade_df.iterrows()
            ):
                if row["Semester_Label"] == "ภาคเรียนที่ 1":
                    semester1_x.append(x)
                    semester1_y.append(
                        row["ParticipationScore"]
                    )
                    semester1_text.append(
                        f'{row["ParticipationScore"]:.2f}'
                    )
                elif row["Semester_Label"] == "ภาคเรียนที่ 2":
                    semester2_x.append(x)
                    semester2_y.append(
                        row["ParticipationScore"]
                    )
                    semester2_text.append(
                        f'{row["ParticipationScore"]:.2f}'
                    )

            # ========================================================
            # 📈 แสดงกราฟ
            # ========================================================
            with st.container(border=True):
                fig_grade = go.Figure()
                # 🔵 ภาคเรียนที่ 1
                fig_grade.add_trace(
                    go.Bar(
                        x=semester1_x,
                        y=semester1_y,
                        text=semester1_text,
                        name="ภาคเรียนที่ 1",
                        marker_color="#35A6DB",
                        textposition="outside"
                    )
                )
                # 🟣 ภาคเรียนที่ 2
                fig_grade.add_trace(
                    go.Bar(
                        x=semester2_x,
                        y=semester2_y,
                        text=semester2_text,
                        name="ภาคเรียนที่ 2",
                        marker_color="#383AB5",
                        textposition="outside"
                    )
                )
                fig_grade.update_layout(
                    height=500,
                    barmode="group",
                    yaxis=dict(
                        range=[0, 100],
                        title="คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                        gridcolor="#E2E8F0"
                    ),
                    xaxis=dict(
                        title="ระดับชั้น",
                        tickmode="array",
                        tickvals=tick_positions,
                        ticktext=tick_labels,
                        showgrid=False
                    ),

                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    legend=dict(
                        title_text="",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(
                        t=80,
                        b=60,
                        l=60,
                        r=30
                    )
                )
                st.plotly_chart(
                    fig_grade,
                    use_container_width=True
                )

                # ========================================================
                # 🔍 วิเคราะห์ข้อมูลจากกราฟจริง
                #    ใช้ข้อมูลชุดเดียวกับที่สร้างกราฟ
                # ========================================================

                graph_df = ordered_grade_df.copy()

                # ------------------------------------------------
                # แยกข้อมูลตามภาคเรียนจากข้อมูลในกราฟ
                # ------------------------------------------------
                semester1_df = graph_df[
                    graph_df["Semester_Label"] == "ภาคเรียนที่ 1"
                ].copy()

                semester2_df = graph_df[
                    graph_df["Semester_Label"] == "ภาคเรียนที่ 2"
                ].copy()

                # ------------------------------------------------
                # ตรวจสอบว่ามีข้อมูลกี่ภาคเรียน
                # ------------------------------------------------
                available_semesters = (
                    graph_df["Semester_Label"]
                    .dropna()
                    .unique()
                    .tolist()
                )

                # ========================================================
                # 📌 กรณีมีข้อมูลครบ 2 ภาคเรียน
                # ========================================================
                if (
                    not semester1_df.empty
                    and not semester2_df.empty
                ):

                    # ------------------------------------------------
                    # ระดับชั้นที่มีคะแนนสูงสุด / ต่ำสุด
                    # จากข้อมูลที่แสดงในกราฟจริง
                    # ------------------------------------------------
                    highest_semester1 = semester1_df.loc[
                        semester1_df["ParticipationScore"].idxmax()
                    ]

                    lowest_semester1 = semester1_df.loc[
                        semester1_df["ParticipationScore"].idxmin()
                    ]

                    highest_semester2 = semester2_df.loc[
                        semester2_df["ParticipationScore"].idxmax()
                    ]

                    lowest_semester2 = semester2_df.loc[
                        semester2_df["ParticipationScore"].idxmin()
                    ]

                    # ========================================================
                    # 📝 แสดงกล่องคำอธิบาย
                    # ========================================================
                    st.markdown(
                        f"""
                        <div style="
                            border-left: 5px solid #383AB5;
                            background: #F5F5FF;
                            padding: 15px 18px;
                            border-radius: 8px;
                            margin-top: 12px;
                            margin-bottom: 25px;
                        ">
                        <div style="
                            font-weight: 700;
                            color: #1E3A5F;
                            margin-bottom: 6px;
                        ">
                        🔍 จากข้อมูลในกราฟ
                        </div>
                        <div style="
                        color: #334155;
                        line-height: 1.8;
                        ">
                        <b>ภาคเรียนที่ 1</b>
                        ระดับชั้นที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                        <b>{highest_semester1["Grade_TH"]}</b>
                        ({highest_semester1["ParticipationScore"]:.2f} คะแนน)
                        และระดับชั้นที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                        <b>{lowest_semester1["Grade_TH"]}</b>
                        ({lowest_semester1["ParticipationScore"]:.2f} คะแนน)
                        <br>
                        <b>ภาคเรียนที่ 2</b>
                        ระดับชั้นที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                        <b>{highest_semester2["Grade_TH"]}</b>
                        ({highest_semester2["ParticipationScore"]:.2f} คะแนน)
                        และระดับชั้นที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                        <b>{lowest_semester2["Grade_TH"]}</b>
                        ({lowest_semester2["ParticipationScore"]:.2f} คะแนน)
                        <br>
                        <span style="
                            color: #64748B;
                            font-size: 13px;
                        ">
                        หมายเหตุ: ข้อความสรุปนี้ดึงจากข้อมูลชุดเดียวกับ
                        ที่ใช้แสดงในกราฟ และจะปรับเปลี่ยนตามข้อมูลในไฟล์ CSV
                        โดยอัตโนมัติ
                        </span>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ========================================================
                # 📌 กรณีมีข้อมูลเพียง 1 ภาคเรียน
                # ========================================================
                elif len(available_semesters) == 1:

                    available_semester = available_semesters[0]

                    available_df = graph_df[
                        graph_df["Semester_Label"]
                        == available_semester
                    ].copy()

                    highest_grade = available_df.loc[
                        available_df["ParticipationScore"].idxmax()
                    ]

                    lowest_grade = available_df.loc[
                        available_df["ParticipationScore"].idxmin()
                    ]

                    available_avg = (
                        available_df["ParticipationScore"].mean()
                    )

                    st.markdown(
                        f"""
                        <div style="
                            border-left: 5px solid #383AB5;
                            background: #F5F5FF;
                            padding: 15px 18px;
                            border-radius: 8px;
                            margin-top: 12px;
                            margin-bottom: 25px;
                        ">

                            <div style="
                                font-weight: 700;
                                color: #1E3A5F;
                                margin-bottom: 6px;
                            ">
                                🔍 จากข้อมูลในกราฟ
                            </div>

                            <div style="
                                color: #334155;
                                line-height: 1.8;
                            ">

                                ข้อมูลที่แสดงในกราฟเป็น
                                <b>{available_semester}</b>
                                โดยมีคะแนนเฉลี่ยการมีส่วนร่วม
                                <b>{available_avg:.2f} คะแนน</b>

                                <br><br>

                                ระดับชั้นที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                                <b>{highest_grade["Grade_TH"]}</b>
                                ({highest_grade["ParticipationScore"]:.2f} คะแนน)

                                <br>

                                ระดับชั้นที่มีคะแนนเฉลี่ยการมีส่วนร่วมต่ำที่สุด คือ
                                <b>{lowest_grade["Grade_TH"]}</b>
                                ({lowest_grade["ParticipationScore"]:.2f} คะแนน)

                                <br><br>

                                <span style="
                                    color: #64748B;
                                    font-size: 13px;
                                ">
                                    หมายเหตุ: ข้อความสรุปนี้ดึงจากข้อมูลชุดเดียวกับ
                                    ที่ใช้แสดงในกราฟ และจะปรับเปลี่ยนตามข้อมูลในไฟล์ CSV
                                    โดยอัตโนมัติ
                                </span>

                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    # ================================================================
    # 📚 5. จำแนกตามรายวิชา
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            📚 คะแนนเฉลี่ยการมีส่วนร่วมจำแนกตามรายวิชา
        </h2>
        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            เปรียบเทียบระดับการมีส่วนร่วมของผู้เรียนในแต่ละรายวิชา
            และแสดงความแตกต่างระหว่างภาคเรียน
        </div>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------
    # คำนวณคะแนนเฉลี่ยรายวิชา × ภาคเรียน
    # ------------------------------------------------
    topic_df = (
        behavior_df
        .groupby(
            [
                "Topic_TH",
                "Semester_Label"
            ],
            as_index=False
        )["ParticipationScore"]
        .mean()
    )

    # ------------------------------------------------
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    # ------------------------------------------------
    if not topic_df.empty:
        # ------------------------------------------------
        # หาลำดับรายวิชา
        # เรียงจากคะแนนเฉลี่ยรวม "น้อย → มาก"
        # ------------------------------------------------
        topic_order = (
            topic_df
            .groupby("Topic_TH")["ParticipationScore"]
            .mean()
            .sort_values(ascending=True)
            .index
            .tolist()
        )
        # ------------------------------------------------
        # สร้างข้อมูลสำหรับจัดลำดับแท่ง
        #
        # ภายในแต่ละรายวิชา:
        # คะแนนต่ำ → คะแนนสูง
        # ------------------------------------------------
        ordered_topic_data = []
        for topic in topic_order:
            temp = topic_df[
                topic_df["Topic_TH"] == topic
            ].copy()
            temp = temp.sort_values(
                "ParticipationScore",
                ascending=True
            )
            for _, row in temp.iterrows():
                ordered_topic_data.append(
                    {
                        "Topic_TH": topic,
                        "Semester_Label": row["Semester_Label"],
                        "ParticipationScore": row["ParticipationScore"]
                    }
                )
        ordered_topic_df = pd.DataFrame(
            ordered_topic_data
        )

        # ------------------------------------------------
        # กำหนดตำแหน่ง X ของแท่งแต่ละแท่ง
        # เพื่อให้แต่ละรายวิชาเรียงตามคะแนนจริง
        # ------------------------------------------------

        x_positions = []
        x_labels = []
        x_values = []
        bar_colors = []
        legend_status = []
        position = 0

        for topic in topic_order:

            temp = ordered_topic_df[
                ordered_topic_df["Topic_TH"] == topic
            ].copy()
            temp = temp.sort_values(
                "ParticipationScore",
                ascending=True
            )
            for _, row in temp.iterrows():
                x_positions.append(position)
                x_labels.append(topic)
                x_values.append(
                    row["ParticipationScore"]
                )
                if row["Semester_Label"] == "ภาคเรียนที่ 1":
                    bar_colors.append("#35A6DB")
                    legend_status.append(
                        "ภาคเรียนที่ 1"
                    )

                else:
                    bar_colors.append("#383AB5")
                    legend_status.append(
                        "ภาคเรียนที่ 2"
                    )
                position += 1
            # เว้นระยะระหว่างรายวิชา
            position += 0.8

        # ------------------------------------------------
        # สร้างกราฟ
        # ------------------------------------------------
        with st.container(border=True):
            fig_topic = go.Figure()
            # ------------------------------------------------
            # เพิ่มแท่งภาคเรียนที่ 1
            # ------------------------------------------------
            semester1_x = []
            semester1_y = []
            semester1_text = []
            # ------------------------------------------------
            # เพิ่มแท่งภาคเรียนที่ 2
            # ------------------------------------------------
            semester2_x = []
            semester2_y = []
            semester2_text = []
            current_position = 0
            for topic in topic_order:
                temp = ordered_topic_df[
                    ordered_topic_df["Topic_TH"] == topic
                ].copy()

                temp = temp.sort_values(
                    "ParticipationScore",
                    ascending=True
                )
                for _, row in temp.iterrows():
                    if row["Semester_Label"] == "ภาคเรียนที่ 1":
                        semester1_x.append(
                            current_position
                        )
                        semester1_y.append(
                            row["ParticipationScore"]
                        )
                        semester1_text.append(
                            f'{row["ParticipationScore"]:.2f}'
                        )
                    else:
                        semester2_x.append(
                            current_position
                        )
                        semester2_y.append(
                            row["ParticipationScore"]
                        )
                        semester2_text.append(
                            f'{row["ParticipationScore"]:.2f}'
                        )
                    current_position += 1
                current_position += 0.8

            # ------------------------------------------------
            # แท่งภาคเรียนที่ 1
            # ------------------------------------------------

            fig_topic.add_trace(
                go.Bar(
                    x=semester1_x,
                    y=semester1_y,
                    name="ภาคเรียนที่ 1",
                    marker_color="#35A6DB",
                    text=semester1_text,
                    textposition="outside",
                    width=0.75
                )
            )
            # ------------------------------------------------
            # แท่งภาคเรียนที่ 2
            # ------------------------------------------------
            fig_topic.add_trace(
                go.Bar(
                    x=semester2_x,
                    y=semester2_y,
                    name="ภาคเรียนที่ 2",
                    marker_color="#383AB5",
                    text=semester2_text,
                    textposition="outside",
                    width=0.75
                )
            )

            # ------------------------------------------------
            # ตำแหน่งกึ่งกลางของแต่ละรายวิชา
            # ------------------------------------------------
            tick_positions = []
            tick_labels = []
            current_position = 0
            for topic in topic_order:
                count = len(
                    ordered_topic_df[
                        ordered_topic_df["Topic_TH"] == topic
                    ]
                )
                if count > 0:
                    start_position = current_position
                    end_position = (
                        current_position
                        + count
                        - 1
                    )
                    tick_positions.append(
                        (start_position + end_position) / 2
                    )
                    tick_labels.append(topic)
                    current_position += count + 0.8

            # ------------------------------------------------
            # ตั้งค่ากราฟ
            # ------------------------------------------------
            fig_topic.update_layout(
                height=550,
                barmode="group",
                yaxis=dict(
                    range=[0, 100],
                    title="คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                    gridcolor="#E2E8F0"
                ),
                xaxis=dict(
                    title="รายวิชา",
                    tickmode="array",
                    tickvals=tick_positions,
                    ticktext=tick_labels,
                    showgrid=False,
                    tickangle=-35
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(
                    title_text="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(
                    t=80,
                    b=100,
                    l=60,
                    r=30
                )
            )
            st.plotly_chart(
                fig_topic,
                use_container_width=True
            )

            # ============================================================
            # 🔍 คำอธิบายจากข้อมูลในกราฟ
            # ============================================================

            # ใช้ข้อมูลชุดเดียวกับที่นำไปสร้างกราฟ
            graph_df = ordered_topic_df.copy()

            # ------------------------------------------------
            # ตรวจสอบว่ามีข้อมูลทั้ง 2 ภาคเรียนหรือไม่
            # ------------------------------------------------
            semester1_df = graph_df[
                graph_df["Semester_Label"] == "ภาคเรียนที่ 1"
            ].copy()

            semester2_df = graph_df[
                graph_df["Semester_Label"] == "ภาคเรียนที่ 2"
            ].copy()

            if not semester1_df.empty and not semester2_df.empty:

                # ------------------------------------------------
                # หารายวิชาที่มีคะแนนสูงสุดและต่ำสุด
                # จากข้อมูลที่แสดงในกราฟจริง
                # ------------------------------------------------
                highest_semester1 = semester1_df.loc[
                    semester1_df["ParticipationScore"].idxmax()
                ]

                lowest_semester1 = semester1_df.loc[
                    semester1_df["ParticipationScore"].idxmin()
                ]

                highest_semester2 = semester2_df.loc[
                    semester2_df["ParticipationScore"].idxmax()
                ]

                lowest_semester2 = semester2_df.loc[
                    semester2_df["ParticipationScore"].idxmin()
                ]

                # ------------------------------------------------
                # เปรียบเทียบคะแนนสูงสุดระหว่างภาคเรียน
                # ------------------------------------------------
                if (
                    highest_semester2["ParticipationScore"]
                    > highest_semester1["ParticipationScore"]
                ):
                    highest_compare_text = (
                        f"เมื่อพิจารณาคะแนนสูงสุดจากกราฟ "
                        f"พบว่า <b>ภาคเรียนที่ 2</b> มีคะแนนสูงสุดมากกว่า "
                        f"ภาคเรียนที่ 1 "
                        f"โดยมีค่า <b>"
                        f"{highest_semester2['ParticipationScore']:.2f}"
                        f"</b> คะแนน"
                    )

                elif (
                    highest_semester1["ParticipationScore"]
                    > highest_semester2["ParticipationScore"]
                ):
                    highest_compare_text = (
                        f"เมื่อพิจารณาคะแนนสูงสุดจากกราฟ "
                        f"พบว่า <b>ภาคเรียนที่ 1</b> มีคะแนนสูงสุดมากกว่า "
                        f"ภาคเรียนที่ 2 "
                        f"โดยมีค่า <b>"
                        f"{highest_semester1['ParticipationScore']:.2f}"
                        f"</b> คะแนน"
                    )

                else:
                    highest_compare_text = (
                        f"เมื่อพิจารณาคะแนนสูงสุดจากกราฟ "
                        f"พบว่า <b>ทั้งสองภาคเรียนมีคะแนนสูงสุดเท่ากัน</b> "
                        f"คือ <b>"
                        f"{highest_semester1['ParticipationScore']:.2f}"
                        f"</b> คะแนน"
                    )

                # ------------------------------------------------
                # แสดงคำอธิบาย
                # ------------------------------------------------
                st.markdown(
                    f"""
                    <div style="
                        border-left: 5px solid #35A6DB;
                        background: #F0F9FF;
                        padding: 15px 18px;
                        border-radius: 8px;
                        margin-top: 12px;
                        margin-bottom: 25px;
                    ">
                    <div style="
                            font-weight: 700;
                            color: #1E3A5F;
                            margin-bottom: 6px;
                        ">
                            🔍 จากข้อมูลในกราฟ
                    </div>
                    <div style="
                            color: #334155;
                            line-height: 1.8;
                    ">
                    {highest_compare_text}
                    <br>
                    <b>ภาคเรียนที่ 1</b>
                    รายวิชาที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                    <b>{highest_semester1["Topic_TH"]}</b>
                    ({highest_semester1["ParticipationScore"]:.2f} คะแนน)
                    และรายวิชาที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                    <b>{lowest_semester1["Topic_TH"]}</b>
                    ({lowest_semester1["ParticipationScore"]:.2f} คะแนน)
                    <br>
                    <b>ภาคเรียนที่ 2</b>
                    รายวิชาที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                    <b>{highest_semester2["Topic_TH"]}</b>
                    ({highest_semester2["ParticipationScore"]:.2f} คะแนน)
                    และรายวิชาที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                    <b>{lowest_semester2["Topic_TH"]}</b>
                    ({lowest_semester2["ParticipationScore"]:.2f} คะแนน)
                    <br>
                    <span style="
                        color: #64748B;
                        font-size: 13px;
                    ">
                    หมายเหตุ: ข้อความสรุปนี้ดึงจากข้อมูลชุดเดียวกับ
                    ที่ใช้แสดงในกราฟ จึงปรับเปลี่ยนตามข้อมูลในไฟล์ CSV
                    โดยอัตโนมัติ
                    </span>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # ------------------------------------------------
                # กรณีมีข้อมูลเพียงภาคเรียนเดียว
                # ------------------------------------------------
                available_semester = graph_df["Semester_Label"].iloc[0]

                highest_topic = graph_df.loc[
                    graph_df["ParticipationScore"].idxmax()
                ]

                lowest_topic = graph_df.loc[
                    graph_df["ParticipationScore"].idxmin()
                ]

                st.markdown(
                    f"""
                    <div style="
                        border-left: 5px solid #35A6DB;
                        background: #F0F9FF;
                        padding: 15px 18px;
                        border-radius: 8px;
                        margin-top: 12px;
                        margin-bottom: 25px;
                    ">
                    <div style="
                        font-weight: 700;
                        color: #1E3A5F;
                        margin-bottom: 6px;
                    ">
                    🔍 จากข้อมูลในกราฟ
                    </div>
                    <div style="
                        color: #334155;
                        line-height: 1.8;
                    ">
                    ข้อมูลที่แสดงในกราฟเป็น
                    <b>{available_semester}</b>
                    <br><br>
                    รายวิชาที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                    <b>{highest_topic["Topic_TH"]}</b>
                    ({highest_topic["ParticipationScore"]:.2f} คะแนน)
                    <br>
                    รายวิชาที่มีคะแนนเฉลี่ยการมีส่วนร่วมต่ำที่สุด คือ
                    <b>{lowest_topic["Topic_TH"]}</b>
                    ({lowest_topic["ParticipationScore"]:.2f} คะแนน)
                    <br>
                    <span style="
                    color: #64748B;
                    font-size: 13px;
                    ">
                    หมายเหตุ: ข้อความสรุปนี้ดึงจากข้อมูลชุดเดียวกับ
                    ที่ใช้แสดงในกราฟ และจะเปลี่ยนแปลงตามข้อมูลในไฟล์ CSV
                    </span>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    # ================================================================
    # 📅 6. เปรียบเทียบระหว่างภาคเรียน
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            📅 คะแนนเฉลี่ยการมีส่วนร่วมระหว่างภาคเรียน
        </h2>
        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            เปรียบเทียบคะแนนเฉลี่ยการมีส่วนร่วมของผู้เรียน
            ระหว่างภาคเรียนที่ 1 และภาคเรียนที่ 2
        </div>
        """,
        unsafe_allow_html=True
    )

    semester_df = (
        behavior_df
        .groupby(
            "Semester_Label",
            as_index=False
        )["ParticipationScore"]
        .mean()
    )

    semester_df["Semester_Label"] = pd.Categorical(
        semester_df["Semester_Label"],
        categories=semester_order,
        ordered=True
    )

    semester_df = semester_df.sort_values(
        "Semester_Label"
    )

    if not semester_df.empty:
        with st.container(border=True):

            fig_semester = px.bar(
                semester_df,
                x="Semester_Label",
                y="ParticipationScore",
                text="ParticipationScore",
                color="Semester_Label",
                color_discrete_map={
                    "ภาคเรียนที่ 1": "#35A6DB",
                    "ภาคเรียนที่ 2": "#383AB5"
                },
                labels={
                    "Semester_Label": "ภาคเรียน",
                    "ParticipationScore": "คะแนนเฉลี่ยการมีส่วนร่วม (0–100)"
                }
            )

            fig_semester.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                showlegend=False
            )

            fig_semester.update_layout(
                height=420,
                yaxis=dict(
                    range=[0, 100],
                    title="คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                    gridcolor="#E2E8F0"
                ),
                xaxis=dict(
                    title="ภาคเรียน",
                    showgrid=False
                ),
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white"
            )

            st.plotly_chart(
                fig_semester,
                use_container_width=True
            )

            if len(semester_df) >= 2:
                highest_semester = semester_df.loc[
                    semester_df["ParticipationScore"].idxmax()
                ]
                lowest_semester = semester_df.loc[
                    semester_df["ParticipationScore"].idxmin()
                ]
                difference = (
                    highest_semester["ParticipationScore"]
                    - lowest_semester["ParticipationScore"]
                )
                st.markdown(
                    f"""
                    <div style="
                        border-left: 5px solid #383AB5;
                        background: #F5F5FF;
                        padding: 15px 18px;
                        border-radius: 8px;
                        margin-top: 12px;
                        margin-bottom: 25px;
                    ">
                    <div style="
                        font-weight: 700;
                        color: #1E3A5F;
                        margin-bottom: 6px;
                    ">🔍 จากข้อมูล</div>
                    <div style="
                        color: #334155;
                        line-height: 1.8;
                    ">
                        <b>{highest_semester["Semester_Label"]}</b>
                        มีคะแนนเฉลี่ยการมีส่วนร่วมสูงกว่า
                        <b>{lowest_semester["Semester_Label"]}</b>
                        อยู่ประมาณ
                        <b>{difference:.2f}</b> คะแนน
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ================================================================
    # 🏆 7. จำแนกตามระดับผลการเรียนรู้
    # ================================================================
    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            🏆 คะแนนเฉลี่ยการมีส่วนร่วมจำแนกตามระดับผลการเรียนรู้
        </h2>
        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            เปรียบเทียบลักษณะพฤติกรรมการเรียนรู้ของผู้เรียน
            ในกลุ่มระดับต่ำ ระดับปานกลาง และระดับสูง
        </div>
        """,
        unsafe_allow_html=True
    )

    class_df = (
        behavior_df
        .groupby(
            "Class_TH",
            as_index=False
        )
        .agg(
            ParticipationScore=("ParticipationScore", "mean"),
            Count=("Class_TH", "size")
        )
    )

    # เรียงตามคะแนนเฉลี่ยจริง
    class_df = class_df.sort_values(
        by="ParticipationScore",
        ascending=True
    ).reset_index(drop=True)

    if not class_df.empty:
        with st.container(border=True):
            # กำหนดสีตามระดับผลการเรียนรู้
            color_map = {
                "ระดับต่ำ": "#b0120a",
                "ระดับปานกลาง": "#f57f17",
                "ระดับสูง": "#33691a"
            }

            fig_class = px.bar(
                class_df,
                x="Class_TH",
                y="ParticipationScore",
                text="ParticipationScore",
                labels={
                    "Class_TH": "ระดับผลการเรียนรู้",
                    "ParticipationScore": "คะแนนเฉลี่ยการมีส่วนร่วม (0–100)"
                },
                color="Class_TH",
                color_discrete_map=color_map
            )
            fig_class.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )
            fig_class.update_layout(
                height=430,
                yaxis=dict(
                    range=[0, 100],
                    title="คะแนนเฉลี่ยการมีส่วนร่วม (0–100)",
                    gridcolor="#E2E8F0"
                ),
                xaxis=dict(
                    title="ระดับผลการเรียนรู้",
                    showgrid=False
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False
            )
            st.plotly_chart(
                fig_class,
                use_container_width=True
            )

            highest_class = class_df.loc[
                class_df["ParticipationScore"].idxmax()
            ]
            lowest_class = class_df.loc[
                class_df["ParticipationScore"].idxmin()
            ]
            st.markdown(
                f"""
                <div style="
                    border-left: 5px solid #60A5FA;
                    background: #EFF6FF;
                    padding: 15px 18px;
                    border-radius: 8px;
                    margin-top: 12px;
                    margin-bottom: 25px;
                ">
                <div style="
                    font-weight: 700;
                        color: #1E3A5F;
                    margin-bottom: 6px;
                ">🔍 จากข้อมูล</div>
                <div style="
                    color: #334155;
                    line-height: 1.8;
                ">
                    กลุ่มที่มีคะแนนเฉลี่ยการมีส่วนร่วมสูงที่สุด คือ
                    <b>{highest_class["Class_TH"]}</b>
                    ({highest_class["ParticipationScore"]:.2f} คะแนน)
                    ขณะที่กลุ่มที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                    <b>{lowest_class["Class_TH"]}</b>
                    ({lowest_class["ParticipationScore"]:.2f} คะแนน)
                    <br>
                    <span style="
                        color: #64748B;
                        font-size: 13px;
                    ">
                        หมายเหตุ: กราฟนี้ใช้สำหรับเปรียบเทียบ
                        ลักษณะพฤติกรรมการเรียนรู้ของแต่ละกลุ่ม
                        ไม่ได้สรุปความสัมพันธ์เชิงเหตุและผล
                    </span>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ================================================================
    # 📋 8. สรุปภาพรวมพฤติกรรมการเรียนรู้
    # ================================================================

    st.markdown(
        """
        <h2 style="
            color: #1E3A5F;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 6px;
        ">
            📋 สรุปภาพรวมพฤติกรรมการเรียนรู้
        </h2>
        <div style="
            color: #64748B;
            font-size: 14px;
            margin-bottom: 15px;
        ">
            สรุปข้อมูลสำคัญจากพฤติกรรมการเรียนรู้ของผู้เรียน
            ตามข้อมูลที่ผู้ใช้เลือก
        </div>
        """,
        unsafe_allow_html=True
    )

    total_students = len(behavior_df)

    if total_students > 0:

        overall_score = behavior_df[
            "ParticipationScore"
        ].mean()

        highest_behavior_name = overall_df.loc[
            overall_df["คะแนนเฉลี่ย"].idxmax(),
            "พฤติกรรม"
        ]

        lowest_behavior_name = overall_df.loc[
            overall_df["คะแนนเฉลี่ย"].idxmin(),
            "พฤติกรรม"
        ]

        st.markdown(
            f"""
            <div style="
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 20px;
                background: white;
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            ">
            <div style="
                color: #334155;
                font-size: 16px;
                line-height: 2;
            ">
                📌 จากข้อมูลที่เลือกจำนวน
                <b>{total_students:,}</b> คน
                พบว่าคะแนนเฉลี่ยการมีส่วนร่วม
                จากพฤติกรรมการเรียนรู้ทั้ง 4 ด้าน
                เท่ากับ <b>{overall_score:.2f}</b> คะแนน
                <br>
                📈 พฤติกรรมที่มีคะแนนเฉลี่ยสูงที่สุด คือ
                <b>{highest_behavior_name}</b>
                <br>
                📉 พฤติกรรมที่มีคะแนนเฉลี่ยต่ำที่สุด คือ
                <b>{lowest_behavior_name}</b>
                <br><br>
                <span style="
                    color: #64748B;
                    font-size: 13px;
                ">
                    หมายเหตุ:
                    การนำเสนอข้อมูลนี้เป็นการวิเคราะห์เชิงพรรณนา
                    เพื่อแสดงลักษณะและรูปแบบของพฤติกรรมการเรียนรู้
                    จากข้อมูลที่มีอยู่
                    โดยไม่ได้สรุปความสัมพันธ์เชิงเหตุและผล
                </span>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
# =====================================================
# การมีส่วนร่วมของผู้ปกครอง 
# =====================================================
elif menu == "การมีส่วนร่วมของผู้ปกครอง":
    st.title("👨‍👩‍👧 การมีส่วนร่วมของผู้ปกครอง")
    st.caption("วิเคราะห์การมีส่วนร่วมของผู้ปกครอง สัดส่วนผู้ดูแลหลัก และความสัมพันธ์ต่อผลสัมฤทธิ์ทางการเรียนของผู้เรียน")
    st.markdown("---")
    plot_df = filtered_df.copy()
    total_parents = len(plot_df)

    # 📌 คำนวณข้อมูลผู้ดูแลหลัก
    relation_counts = (
        plot_df["Relation"]
        .astype(str)
        .replace({"Father": "บิดา", "Mum": "มารดา"})
        .value_counts()
        .reset_index()
    )
    relation_counts.columns = ["ผู้ดูแลหลัก", "จำนวน"]
    relation_counts["ร้อยละ"] = (relation_counts["จำนวน"] / total_parents) * 100

    father_cnt = relation_counts[relation_counts["ผู้ดูแลหลัก"] == "บิดา"]["จำนวน"].values[0] if not relation_counts[relation_counts["ผู้ดูแลหลัก"] == "บิดา"].empty else 0
    father_pct = relation_counts[relation_counts["ผู้ดูแลหลัก"] == "บิดา"]["ร้อยละ"].values[0] if not relation_counts[relation_counts["ผู้ดูแลหลัก"] == "บิดา"].empty else 0
    
    mother_cnt = relation_counts[relation_counts["ผู้ดูแลหลัก"] == "มารดา"]["จำนวน"].values[0] if not relation_counts[relation_counts["ผู้ดูแลหลัก"] == "มารดา"].empty else 0
    mother_pct = relation_counts[relation_counts["ผู้ดูแลหลัก"] == "มารดา"]["ร้อยละ"].values[0] if not relation_counts[relation_counts["ผู้ดูแลหลัก"] == "มารดา"].empty else 0
    
    # =====================================================
    # 📌 แถวที่ 1: กราฟโดนัท + การ์ดสรุปข้อมูลแบบในรูปตัวอย่าง
    # =====================================================
    with st.container(border=True):
        st.markdown("<h3 style='margin-bottom:0px;'>👨‍👩‍👧 จำนวนนักเรียนตามผู้ดูแลหลัก</h3>", unsafe_allow_html=True)
        st.caption("การกระจายตัวของผู้เรียนจำแนกตามผู้ดูแลหลัก (บิดา / มารดา)")
        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_summary = st.columns([1.5, 1])

        # --- ฝั่งซ้าย: กราฟโดนัท ---
        with col_chart:
            fig_donut = px.pie(
                relation_counts,
                values="จำนวน",
                names="ผู้ดูแลหลัก",
                hole=0.62,
                color="ผู้ดูแลหลัก",
                color_discrete_map={"บิดา": "#2285B0", "มารดา": "#36BCAA"} # สีน้ำเงินฟ้า และ สีม่วงอ่อน ตามรูปตัวอย่าง
            )
            
            fig_donut.update_traces(
                rotation=180,
                direction="clockwise",
                textposition="inside",
                textinfo="percent+value",
                texttemplate="<b>%{percent:.0%}</b><br>%{value} คน",
                textfont=dict(family="Sarabun, sans-serif", size=14, color="#FFFFFF"),
                hovertemplate="ผู้ดูแลหลัก: <b>%{label}</b><br>จำนวน: <b>%{value:,} คน</b> (%{percent})<extra></extra>"
            )
            
            # ใส่ข้อความรวมไว้ตรงกลางวงโดนัท ( Center Annotation )
            # ปรับแต่ง Annotation ตรงกลางวงโดนัท (ใช้แท็กมาตรฐานของ Plotly เพื่อป้องกันแท็กโชว์ค้าง)
            fig_donut.update_layout(
                annotations=[{
                    "text": f"<span style='font-size: 13px; color: #64748B;'>นักเรียนทั้งหมด</span><br><br><b style='font-size: 24px; color: #1E293B;'>{total_parents:,} คน</b>",
                    "x": 0.5, 
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"family": "Sarabun, sans-serif"},
                    "align": "center"
                }],
                height=380,
                margin=dict(t=10, b=10, l=10, r=10),
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Sarabun, sans-serif", size=14)
                )
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        # --- ฝั่งขวา: การ์ดสรุปข้อมูล (สไตล์เดียวกับรูปตัวอย่าง) ---
        with col_summary:
            with st.container(border=True):
                # หัวข้อสรุปข้อมูล
                st.markdown(
                    """
                    <div style='background-color: #DBEAFE; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px;'>
                        <b style='color: #1E40AF; font-size: 16px;'>📊 สรุปข้อมูล</b>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # บล็อกที่ 1: บิดา
                st.markdown(
                    f"""
                    <div style='background-color: #EFF6FF; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid #38BDF8;'>
                        <span style='color: #1E3A8A; font-weight: bold; font-size: 14px;'>🟦 บิดา (ผู้ดูแลหลัก)</span><br>
                        <b style='color: #2285B0; font-size: 22px;'>{father_cnt:,} คน</b><br>
                        <span style='color: #64748B; font-size: 12px;'>คิดเป็น {father_pct:.1f}%</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # บล็อกที่ 2: มารดา
                st.markdown(
                    f"""
                    <div style='background-color: #F5F3FF; padding: 12px 16px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #2DD4BF;'>
                        <span style='color: #4C1D95; font-weight: bold; font-size: 14px;'>🟪 มารดา (ผู้ดูแลหลัก)</span><br>
                        <b style='color: #36BCAA; font-size: 22px;'>{mother_cnt:,} คน</b><br>
                        <span style='color: #64748B; font-size: 12px;'>คิดเป็น {mother_pct:.1f}%</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # บล็อกข้อสังเกตเชิงสถิติ ด้านล่างสุด
                st.markdown(
                    f"""
                    <div style='background-color: #F6F2FF; padding: 12px 14px; border-radius: 8px; 
                    border: 1px solid #FCD34D; margin-top: 0px; margin-bottom: 10px;'>
                        <span style='color: #92400E; font-size: 13px;'>
                            💡 <b>ข้อสังเกตเชิงสถิติ:</b> ผู้เรียนส่วนใหญ่มีผู้ดูแลหลักเป็น <b>บิดา</b> จำนวน {father_cnt:,} คน ({father_pct:.1f}%) จากผู้เรียนทั้งหมด {total_parents:,} คน
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # 📌 แถวที่ 2: กราฟแท่งเปรียบเทียบตามผู้ดูแลหลัก (บิดา vs มารดา)
    # =====================================================
    # คำนวณจำนวนนับจำแนกตาม Relation (บิดา / มารดา)
    survey_rel = pd.crosstab(
        plot_df["Relation"],
        plot_df["ParentAnsweringSurvey"]
    )
    sat_rel = pd.crosstab(
        plot_df["Relation"],
        plot_df["ParentschoolSatisfaction"]
    )   
    # ดึงค่าจำนวนคน
    f_surv_yes = survey_rel.loc["Father", "Yes"] if "Father" in survey_rel.index and "Yes" in survey_rel.columns else 0
    f_surv_no = survey_rel.loc["Father", "No"] if "Father" in survey_rel.index and "No" in survey_rel.columns else 0
    m_surv_yes = survey_rel.loc["Mum", "Yes"] if "Mum" in survey_rel.index and "Yes" in survey_rel.columns else 0
    m_surv_no = survey_rel.loc["Mum", "No"] if "Mum" in survey_rel.index and "No" in survey_rel.columns else 0

    f_sat_good = sat_rel.loc["Father", "Good"] if "Father" in sat_rel.index and "Good" in sat_rel.columns else 0
    f_sat_bad = sat_rel.loc["Father", "Bad"] if "Father" in sat_rel.index and "Bad" in sat_rel.columns else 0
    m_sat_good = sat_rel.loc["Mum", "Good"] if "Mum" in sat_rel.index and "Good" in sat_rel.columns else 0
    m_sat_bad = sat_rel.loc["Mum", "Bad"] if "Mum" in sat_rel.index and "Bad" in sat_rel.columns else 0

    # =====================================================
    # 📊 จัดโครงสร้างข้อมูลสำหรับกราฟ
    # =====================================================
    bar_data_rel = pd.DataFrame([
        # การตอบแบบสำรวจ
        {"หัวข้อ": "ตอบ", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_surv_yes},
        {"หัวข้อ": "ตอบ", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_surv_yes},
        {"หัวข้อ": "ไม่ตอบ", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_surv_no},
        {"หัวข้อ": "ไม่ตอบ", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_surv_no},

        {"หัวข้อ": "พอใจ", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_sat_good},
        {"หัวข้อ": "พอใจ", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_sat_good},
        {"หัวข้อ": "ไม่พอใจ", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_sat_bad},
        {"หัวข้อ": "ไม่พอใจ", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_sat_bad},
    ])
    with st.container(border=True):
            st.markdown("<h3 style='margin-bottom:0px;'>📊 เปรียบเทียบการมีส่วนร่วมจำแนกตามผู้ดูแลหลัก (บิดา / มารดา)</h3>", unsafe_allow_html=True)
            st.caption("เปรียบเทียบการตอบแบบสำรวจและความพึงพอใจแยกระหว่างบิดาและมารดา")
            st.markdown("<br>", unsafe_allow_html=True)
    
            # =====================================================
            # 🔽 เรียงแท่งกราฟทุกแท่งตามจำนวนจริง จากน้อย → มาก
            # =====================================================
            bar_data_rel = bar_data_rel.sort_values(
                by="จำนวน",
                ascending=True
            ).reset_index(drop=True)
            # ตำแหน่งแท่งกราฟ
            bar_data_rel["ตำแหน่ง"] = range(len(bar_data_rel))

            row2_left, row2_right = st.columns([1.5, 1])    
            with row2_left:
            # =====================================================
            # 📊 กราฟ
            # =====================================================
                fig_bar = px.bar(
                    bar_data_rel,
                    x="ตำแหน่ง",
                    y="จำนวน",
                    color="ผู้ดูแลหลัก",
                    text="จำนวน",
                    color_discrete_map={
                        "บิดา": "#2285B0",
                        "มารดา": "#36BCAA"
                    },
                    labels={
                        "ตำแหน่ง": "",
                        "จำนวน": "จำนวน (คน)",
                        "ผู้ดูแลหลัก": "ผู้ดูแลหลัก"
                    }
                )
                # =====================================================
                # 🏷️ แสดงชื่อใต้แท่ง
                # =====================================================
                fig_bar.update_xaxes(
                    tickmode="array",
                    tickvals=bar_data_rel["ตำแหน่ง"].tolist(),
                    ticktext=bar_data_rel["หัวข้อ"].tolist(),
                    showgrid=False
                )
                fig_bar.update_traces(
                    texttemplate="<b>%{y:,} คน</b>",
                    textposition="outside",
                    textfont=dict(
                        family="Sarabun, sans-serif",
                        size=12
                    ),
                    hovertemplate=(
                        "หัวข้อ: <b>%{customdata[0]}</b><br>"
                        "ผู้ดูแลหลัก: <b>%{customdata[1]}</b><br>"
                        "จำนวน: <b>%{y:,} คน</b>"
                        "<extra></extra>"
                    ),
                    customdata=bar_data_rel[
                        ["หัวข้อ", "ผู้ดูแลหลัก"]
                    ].values
                    )
                max_y = bar_data_rel["จำนวน"].max()
                fig_bar.update_layout(
                    height=390,
                    # ⭐ ไม่จัดกลุ่มบิดา-มารดา
                    # เพราะต้องเรียงแท่งจริงจากน้อย → มาก
                    bargap=0.25,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.3,
                        xanchor="center",
                        x=0.5,
                        font=dict(
                            family="Sarabun, sans-serif",
                            size=13
                        )
                    ),
                    margin=dict(
                        t=30,
                        b=55,
                        l=10,
                        r=10
                    ),
                    font=dict(
                        family="Sarabun, sans-serif",
                        size=12
                    ),
                    template="plotly_white",
                    yaxis=dict(
                        range=[0, max_y * 1.22],
                        tickformat=",d"
                    )
                )
                st.plotly_chart(
                    fig_bar,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )   
            # กรอบสรุปข้อมูลแถวที่ 2
            with row2_right:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div style='background-color: #DCFCE7; padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 15px;'>
                        <b style='color: #166534; font-size: 16px;'>📝 สรุปเปรียบเทียบตามผู้ดูแล</b>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
                    * **การตอบแบบสำรวจ:**
                        * **บิดา:** ตอบ {f_surv_yes:,} คน / ไม่ตอบ {f_surv_no:,} คน
                        * **มารดา:** ตอบ {m_surv_yes:,} คน / ไม่ตอบ {m_surv_no:,} คน
                    * **ความพึงพอใจต่อโรงเรียน:**
                        * **บิดา:** พึงพอใจ {f_sat_good:,} คน / ไม่พึงพอใจ {f_sat_bad:,} คน
                        * **มารดา:** พึงพอใจ {m_sat_good:,} คน / ไม่พึงพอใจ {m_sat_bad:,} คน
                        """)
                    st.markdown(
                        f"""
                        <div style='background-color: #FEF3FF; padding: 10px 12px; border-radius: 8px; 
                        border: 1px solid #FCD34D; margin-top: 10px; margin-top: 0px; margin-bottom: 10px;'>
                                <span style='color: #92400E; font-size: 12px;'>
                            💡 <b>ข้อสังเกต:</b> มารดามีสัดส่วนการตอบแบบสำรวจ ({ (m_surv_yes/(m_surv_yes+m_surv_no))*100:.1f}% ) 
                                และความพึงพอใจ ({ (m_sat_good/(m_sat_good+m_sat_bad))*100:.1f}% ) สูงกว่าบิดา
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )   
    # =====================================================
    # 📌 ส่วนที่ 3 (ด้านล่างสุด): Descriptive Analysis (กราฟคู่ ซ้าย-ขวา)
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3>📌 เปรียบเทียบการมีส่วนร่วมของผู้ดูแลหลัก</h3>", unsafe_allow_html=True)
    st.caption("เปรียบเทียบระหว่างการมีส่วนร่วมของผู้ปกครองกับพฤติกรรมและผลสัมฤทธิ์ทางการเรียนของผู้เรียน")
    st.markdown("<br>", unsafe_allow_html=True)

    # แบ่งเลย์เอาต์เป็น 2 คอลัมน์เท่าๆ กัน (ซ้าย-ขวา)
    col_left, col_right = st.columns([1, 1])

    # -----------------------------------------------------
    # 👈 กราฟฝั่งซ้าย: แก้ไขอักษรซ้อนทับ
    # -----------------------------------------------------
    with col_left:
        with st.container(border=True):
            avg_act = (
                plot_df.groupby("Relation")[["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]]
                .mean()
                .reset_index()
            )
            melted_act = avg_act.melt(id_vars=["Relation"], var_name="กิจกรรม", value_name="ค่าเฉลี่ย")
            melted_act["Relation"] = melted_act["Relation"].replace({"Father": "บิดา", "Mum": "มารดา"})
            activity_map = {
                "raisedhands": "ยกมือตอบคำถาม",
                "VisITedResources": "เข้าดูสื่อการเรียน",
                "AnnouncementsView": "การดูประกาศ",
                "Discussion": "การร่วมอภิปราย"
            }
            melted_act["กิจกรรม"] = melted_act["กิจกรรม"].map(activity_map)

            fig_act = px.bar(
                melted_act,
                x="กิจกรรม",
                y="ค่าเฉลี่ย",
                color="Relation",
                barmode="group",
                text="ค่าเฉลี่ย",
                title="<b>🏃‍♂️ พฤติกรรมเรียนจำแนกตามผู้ดูแลหลัก</b>",
                color_discrete_map={"บิดา": "#2285B0", "มารดา": "#36BCAA"}
            )
            fig_act.update_traces(
                texttemplate='<b>%{y:.1f}</b>',
                textposition='outside',
                textfont=dict(family="Sarabun, sans-serif", size=11),
                hovertemplate="กิจกรรม: <b>%{x}</b><br>%{fullData.name}: <b>%{y:.1f} ครั้ง</b><extra></extra>"
            )
            max_act_y = melted_act["ค่าเฉลี่ย"].max()
            fig_act.update_layout(
                height=360,
                bargap=0.18,
                bargroupgap=0.06,
                margin=dict(t=40, b=60, l=10, r=10), # เพิ่ม margin b (ด้านล่าง) เป็น 60
                font=dict(family="Sarabun, sans-serif", size=11),
                template="plotly_white",
                xaxis=dict(title=""), # ลบชื่อแกน X ออกเพื่อไม่ให้ลอยมาซ้อนทับ Legend
                yaxis=dict(title="ค่าเฉลี่ย (ครั้ง)", range=[0, max_act_y * 1.25]),
                legend=dict(
                    title="", # ลบหัวข้อ Legend
                    orientation="h",
                    yanchor="top",
                    y=-0.18, # ดัน Legend ลงมาข้างล่าง
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_act, use_container_width=True, config={"displayModeBar": False})
            mother_resource_avg = plot_df.loc[
                plot_df["Relation"] == "Mum",
                "VisITedResources"
            ].mean()

            father_resource_avg = plot_df.loc[
                plot_df["Relation"] == "Father",
                "VisITedResources"
            ].mean()

            mother_hand_avg = plot_df.loc[
                plot_df["Relation"] == "Mum",
                "raisedhands"
            ].mean()

            father_hand_avg = plot_df.loc[
                plot_df["Relation"] == "Father",
                "raisedhands"
            ].mean()
            # สรุป Insight ฝั่งซ้าย
            st.markdown(
                f"""
                <div style='background-color: #F7F8FF; padding: 10px 12px; border-radius: 8px;
                    border: 1px solid #FCD34D; margin-top: 0px; margin-bottom: 10px'>
                    <span style='color: #92400E; font-size: 12px; line-height: 1.5; display: block;'>
                    💡 <b>ข้อสังเกต:</b>
                        นักเรียนที่มี <b>มารดา</b> เป็นผู้ดูแลหลัก
                    มีค่าเฉลี่ยการเข้าดูสื่อการเรียน
                    <b>{mother_resource_avg:.1f}</b> ครั้ง และการยกมือตอบคำถาม
                    <b>{mother_hand_avg:.1f}</b> ครั้ง เทียบกับบิดาที่มีค่าเฉลี่ย
                    <b>{father_resource_avg:.1f}</b> และ <b>{father_hand_avg:.1f}</b> ครั้ง ตามลำดับ
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # 👉 กราฟฝั่งขวา ความสัมพันธ์กับผลสัมฤทธิ์
    # -----------------------------------------------------
    with col_right:
        with st.container(border=True):
            # =====================================================
            # 📌 ความสัมพันธ์กับผลสัมฤทธิ์ทางการเรียน
            # =====================================================
            ct_survey_class = pd.crosstab(
                plot_df["Class"],
                plot_df["ParentAnsweringSurvey"],
                normalize="index"
            ) * 100
            ct_survey_class = ct_survey_class.reset_index()
            # แปลงชื่อระดับผลการเรียน
            ct_survey_class["Class"] = ct_survey_class["Class"].replace({
                "L": "ระดับต่ำ",
                "M": "ระดับปานกลาง",
                "H": "ระดับสูง"
            })
            # กำหนดลำดับของ Class
            ct_survey_class["Class"] = pd.Categorical(
                ct_survey_class["Class"],
                categories=[
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ],
                ordered=True
            )
            ct_survey_class = ct_survey_class.sort_values("Class")
            # =====================================================
            # 📊 แปลงข้อมูลสำหรับกราฟ
            # =====================================================
            melted_survey = ct_survey_class.melt(
                id_vars=["Class"],
                var_name="ParentAnsweringSurvey",
                value_name="สัดส่วน (%)"
            )
            melted_survey["ParentAnsweringSurvey"] = (
                melted_survey["ParentAnsweringSurvey"]
                .replace({
                    "Yes": "ตอบแบบสำรวจ",
                    "No": "ไม่ตอบแบบสำรวจ"
                })
            )
            # =====================================================
            # ⭐ เรียงแท่งตามค่าจริงจากน้อย → มาก
            # =====================================================
            melted_survey = melted_survey.sort_values(
                by="สัดส่วน (%)",
                ascending=True
            ).reset_index(drop=True)
            # สร้างตำแหน่งแท่ง
            melted_survey["ตำแหน่ง"] = range(len(melted_survey))
            # =====================================================
            # 📊 สร้างกราฟ
            # =====================================================
            fig_class = px.bar(
                melted_survey,
                x="ตำแหน่ง",
                y="สัดส่วน (%)",
                color="ParentAnsweringSurvey",
                text="สัดส่วน (%)",
                title="<b>🎓 ความสัมพันธ์กับผลสัมฤทธิ์ทางการเรียน</b>",
                color_discrete_map={
                    "ตอบแบบสำรวจ": "#689f38",
                    "ไม่ตอบแบบสำรวจ": "#afb42b"
                },
                labels={
                    "ตำแหน่ง": "",
                    "สัดส่วน (%)": "สัดส่วน (%)",
                    "ParentAnsweringSurvey": ""
                }
            )
            # =====================================================
            # 🏷️ ชื่อใต้แท่ง
            # =====================================================
            fig_class.update_xaxes(
                tickmode="array",
                tickvals=melted_survey["ตำแหน่ง"].tolist(),
                ticktext=(
                    melted_survey["Class"]
                    .astype(str)
                    .replace({
                        "ระดับต่ำ": "L",
                        "ระดับปานกลาง": "M",
                        "ระดับสูง": "H"
                    })
                    .tolist()
                ),
                showgrid=False
            )
            # =====================================================
            # 🖱️ Hover + ตัวเลขบนแท่ง
            # =====================================================
            fig_class.update_traces(
                texttemplate="<b>%{y:.1f}%</b>",
                textposition="outside",
                textfont=dict(
                    family="Sarabun, sans-serif",
                    size=11
                ),
                hovertemplate=(
                    "กลุ่มผลการเรียน: <b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}: <b>%{y:.1f}%</b>"
                    "<extra></extra>"
                ),
                customdata=melted_survey[
                    ["Class", "ParentAnsweringSurvey"]
                ].values
            )
            # =====================================================
            # ⚙️ ตั้งค่ากราฟ
            # =====================================================
            max_y = melted_survey["สัดส่วน (%)"].max()
            fig_class.update_layout(
                height=360,
                bargap=0.25,
                margin=dict(
                    t=40,
                    b=60,
                    l=10,
                    r=10
                ),
                font=dict(
                    family="Sarabun, sans-serif",
                    size=11
                ),
                template="plotly_white",
                xaxis=dict(
                    title=""
                ),
                yaxis=dict(
                    title="สัดส่วน (%)",
                    range=[0, max_y * 1.18]
                ),
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="top",
                    y=-0.18,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(
                fig_class,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            father_survey_total = f_surv_yes + f_surv_no
            mother_survey_total = m_surv_yes + m_surv_no

            father_sat_total = f_sat_good + f_sat_bad
            mother_sat_total = m_sat_good + m_sat_bad

            father_survey_pct = (
                f_surv_yes / father_survey_total * 100
                if father_survey_total > 0 else 0
            )

            mother_survey_pct = (
                m_surv_yes / mother_survey_total * 100
                if mother_survey_total > 0 else 0
            )

            father_sat_pct = (
                f_sat_good / father_sat_total * 100
                if father_sat_total > 0 else 0
            )

            mother_sat_pct = (
                m_sat_good / mother_sat_total * 100
                if mother_sat_total > 0 else 0
            )
            # สรุป Insight ฝั่งขวา
            st.markdown(
                f"""
                <div style='background-color: #FEF3FF; padding: 10px 12px; border-radius: 8px;
                    border: 1px solid #FCD34D; margin-top: 0px; margin-bottom: 10px;'>
                    <span style='color: #92400E; font-size: 12px; line-height: 1.5;'>
                    💡 <b>ข้อสังเกต:</b>
                        มารดามีสัดส่วนการตอบแบบสำรวจ
                    <b>{mother_survey_pct:.1f}%</b>
                    และความพึงพอใจต่อโรงเรียน
                    <b>{mother_sat_pct:.1f}%</b>
                    เทียบกับบิดาที่มีสัดส่วน
                    <b>{father_survey_pct:.1f}%</b>
                    และ <b>{father_sat_pct:.1f}%</b> ตามลำดับ
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )


# ================================================================
# 🔗 การวิเคราะห์ความสัมพันธ์ของตัวแปร
# ================================================================
elif menu == "การวิเคราะห์ความสัมพันธ์ของตัวแปร":
    # ============================================================
    # 🎨 CSS ตกแต่งหน้า
    # ============================================================
    st.markdown(
        """
        <style>
        /* กรอบของแต่ละส่วน */
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border: 1.5px solid #1E3A5F !important;
            border-radius: 18px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.04) !important;
        }
        /* หัวข้อหลัก */
        .relationship-title {
            color: #0A2540;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 5px;
        }
        /* คำอธิบายใต้หัวข้อ */
        .relationship-subtitle {
            color: #64748B;
            font-size: 15px;
            margin-bottom: 20px;
        }
        /* หัวข้อกราฟ */
        .chart-title {
            color: #1E3A5F;
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        /* คำอธิบายกราฟ */
        .chart-description {
            color: #64748B;
            font-size: 13px;
            margin-bottom: 10px;
        }
        /* กล่องคำอธิบาย */
        .analysis-note {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #3B82F6;
            border-radius: 10px;
            padding: 14px 16px;
            color: #334155;
            font-size: 14px;
            line-height: 1.8;
            margin-top: 10px;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # 🏷️ หัวข้อหน้า
    # ============================================================
    st.markdown(
        """
        <div class="relationship-title">
            🔗 การวิเคราะห์ความสัมพันธ์ของตัวแปร
        </div>
        <div class="relationship-subtitle">
            วิเคราะห์ความสัมพันธ์ระหว่างพฤติกรรมการเรียนรู้
            และระดับผลการเรียนของผู้เรียน
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # 📌 กำหนดตัวแปร
    # ============================================================
    # ตัวแปรอิสระ
    independent_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion"
    ]
    # ชื่อตัวแปรภาษาไทย
    variable_names = {
        "raisedhands": "การยกมือ",
        "VisITedResources": "การเข้าดูสื่อ",
        "AnnouncementsView": "การดูประกาศ",
        "Discussion": "การอภิปราย"
    }
    # ตัวแปรตาม
    dependent_col = "Class"
    # ชื่อระดับผลการเรียน
    class_names = {
        "L": "ระดับต่ำ",
        "M": "ระดับปานกลาง",
        "H": "ระดับสูง"
    }
    # ลำดับระดับผลการเรียน
    class_order = [
        "L",
        "M",
        "H"
    ]
    # ============================================================
    # 🎨 กำหนดสีของระดับผลการเรียน
    # ============================================================

    class_colors = {
        "ระดับสูง": "#33691e",       # 🟦 น้ำเงิน
        "ระดับปานกลาง": "#f57f17",   # 🟨 เหลือง/ส้ม
        "ระดับต่ำ": "#b0120a"        # 🟥 แดง
    }
    # ============================================================
    # 📊 ส่วนที่ 1 : Correlation Heatmap
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                📊 เมทริกซ์ความสัมพันธ์ของพฤติกรรมการเรียน
            </div>
            <div class="chart-description">
                แสดงระดับความสัมพันธ์ระหว่างตัวแปรอิสระแต่ละคู่
            </div>
            """,
            unsafe_allow_html=True
        )
        # --------------------------------------------------------
        # คำนวณค่าสหสัมพันธ์
        # --------------------------------------------------------
        corr_matrix = filtered_df[
            independent_cols
        ].corr()
        # --------------------------------------------------------
        # เปลี่ยนชื่อแถวและคอลัมน์เป็นภาษาไทย
        # --------------------------------------------------------
        corr_matrix.index = [
            variable_names[col]
            for col in corr_matrix.index
        ]
        corr_matrix.columns = [
            variable_names[col]
            for col in corr_matrix.columns
        ]

        # --------------------------------------------------------
        # สร้าง Heatmap
        # --------------------------------------------------------
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="equal",
            color_continuous_scale=[
                "#EFF6FF",
                "#BFDBFE",
                "#93C5FD",
                "#60A5FA",
                "#3B82F6",
                "#1E3A8A"
            ],
            labels={
                "x": "",
                "y": "",
                "color": "ค่าความสัมพันธ์"
            }
        )

        # --------------------------------------------------------
        # ปรับตัวเลขใน Heatmap
        # --------------------------------------------------------
        fig_corr.update_traces(
            textfont=dict(
                size=14
            )
        )

        # --------------------------------------------------------
        # ปรับรูปแบบ Heatmap
        # --------------------------------------------------------
        fig_corr.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            coloraxis_colorbar=dict(
                title=dict(
                    text="ค่าความสัมพันธ์",
                    font=dict(
                        color="#1E3A5F",
                        size=13
                    )
                ),
                tickfont=dict(
                    color="#64748B",
                    size=11
                )
            ),
            xaxis=dict(
                tickfont=dict(
                    color="#64748B",
                    size=12
                )
            ),
            yaxis=dict(
                tickfont=dict(
                    color="#64748B",
                    size=12
                )
            )
        )
        st.plotly_chart(
            fig_corr,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
        # ========================================================
        # 🔍 คำอธิบายจากข้อมูลใน Heatmap
        # ========================================================
        # คำนวณจากข้อมูลเดียวกับที่ใช้สร้าง Heatmap
        corr_data = filtered_df[independent_cols].corr()

        # หาคู่ตัวแปรที่มีความสัมพันธ์สูงสุดและต่ำสุด
        corr_pairs = []

        for i in range(len(independent_cols)):
            for j in range(i + 1, len(independent_cols)):

                var1 = independent_cols[i]
                var2 = independent_cols[j]
                corr_value = corr_data.loc[var1, var2]

                if pd.notna(corr_value):
                    corr_pairs.append(
                        {
                            "var1": var1,
                            "var2": var2,
                            "corr": corr_value
                        }
                    )
        if corr_pairs:

            # ความสัมพันธ์สูงสุด
            max_pair_data = max(
                corr_pairs,
                key=lambda x: abs(x["corr"])
            )

            # ความสัมพันธ์ต่ำสุด
            min_pair_data = min(
                corr_pairs,
                key=lambda x: abs(x["corr"])
            )

            max_var1 = variable_names[max_pair_data["var1"]]
            max_var2 = variable_names[max_pair_data["var2"]]
            max_corr_value = max_pair_data["corr"]

            min_var1 = variable_names[min_pair_data["var1"]]
            min_var2 = variable_names[min_pair_data["var2"]]
            min_corr_value = min_pair_data["corr"]

            # ----------------------------------------------------
            # แปลระดับความสัมพันธ์
            # ----------------------------------------------------
            if abs(max_corr_value) >= 0.70:
                max_level = "ค่อนข้างสูง"
            elif abs(max_corr_value) >= 0.40:
                max_level = "ปานกลาง"
            elif abs(max_corr_value) >= 0.20:
                max_level = "ค่อนข้างต่ำ"
            else:
                max_level = "ต่ำ"

            if abs(min_corr_value) >= 0.70:
                min_level = "ค่อนข้างสูง"
            elif abs(min_corr_value) >= 0.40:
                min_level = "ปานกลาง"
            elif abs(min_corr_value) >= 0.20:
                min_level = "ค่อนข้างต่ำ"
            else:
                min_level = "ต่ำ"

            # ----------------------------------------------------
            # ทิศทางความสัมพันธ์
            # ----------------------------------------------------
            max_direction = (
                "ทิศทางเดียวกัน"
                if max_corr_value > 0
                else "ทิศทางตรงข้าม"
                if max_corr_value < 0
                else "ไม่มีทิศทาง"
            )

            min_direction = (
                "ทิศทางเดียวกัน"
                if min_corr_value > 0
                else "ทิศทางตรงข้าม"
                if min_corr_value < 0
                else "ไม่มีทิศทาง"
            )

            # ----------------------------------------------------
            # แสดงคำอธิบายใต้กราฟ
            # ----------------------------------------------------
            st.markdown(
                f"""
                <div class="analysis-note">
                <b>🔍 คำอธิบายจากกราฟ</b>
                <br>
                จากเมทริกซ์ความสัมพันธ์ พบว่า
                <b>{max_var1}</b> กับ <b>{max_var2}</b>
                มีความสัมพันธ์มากที่สุด
                โดยมีค่าสหสัมพันธ์ <b>{max_corr_value:.2f}</b>
                ซึ่งเป็นความสัมพันธ์ใน<b>{max_direction}</b>
                และอยู่ในระดับ<b>{max_level}</b>
                <br>
                ขณะที่ <b>{min_var1}</b> กับ <b>{min_var2}</b>
                มีความสัมพันธ์น้อยที่สุดในชุดข้อมูล
                โดยมีค่าสหสัมพันธ์ <b>{min_corr_value:.2f}</b>
                ซึ่งเป็นความสัมพันธ์ใน<b>{min_direction}</b>
                และอยู่ในระดับ<b>{min_level}</b>
                <br>
                <span style="color:#64748B; font-size:13px;">
                💡 ค่าสหสัมพันธ์มีค่าอยู่ระหว่าง -1 ถึง 1
                โดยค่าที่เข้าใกล้ 1 หรือ -1
                แสดงถึงความสัมพันธ์ที่มากขึ้น
                ส่วนค่าที่เข้าใกล้ 0
                แสดงถึงความสัมพันธ์ที่น้อยลง
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )        
    # ============================================================
    # 📊 คำนวณค่าเฉลี่ยของพฤติกรรมการเรียน
    # ============================================================
    class_mean = (
        filtered_df
        .groupby(dependent_col)[independent_cols]
        .mean()
        .reindex(class_order)
    )

    # ============================================================
    # 📊 กราฟที่ 1 : การยกมือ
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                📊 การยกมือกับระดับผลการเรียน
            </div>
            <div class="chart-description">
                เปรียบเทียบค่าเฉลี่ยการยกมือของผู้เรียนแต่ละระดับผลการเรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        chart_df = class_mean[
            ["raisedhands"]
        ].reset_index()
        chart_df["Class_TH"] = (
            chart_df["Class"]
            .map(class_names)
        )
        # --------------------------------------------------------
        # สร้างกราฟ
        # --------------------------------------------------------
        fig_raisedhands = px.bar(
            chart_df,
            x="Class_TH",
            y="raisedhands",
            color="Class_TH",
            # 🎨 กำหนดสีแต่ละระดับ
            color_discrete_map=class_colors,
            text="raisedhands",
            labels={
                "Class_TH": "ระดับผลการเรียน",
                "raisedhands": "ค่าเฉลี่ยการยกมือ"
            },
            category_orders={
                "Class_TH": [
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ]
            }
        )
        fig_raisedhands.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0
        )
        fig_raisedhands.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="ระดับผลการเรียน",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                showgrid=False
            ),
            yaxis=dict(
                title="ค่าเฉลี่ยการยกมือ",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                gridcolor="#E2E8F0",
                zeroline=False
            ),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            )
        )
        st.plotly_chart(
            fig_raisedhands,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ============================================================
    # 📊 กราฟที่ 2 : การเข้าดูสื่อ
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                📊 การเข้าดูสื่อกับระดับผลการเรียน
            </div>
            <div class="chart-description">
                เปรียบเทียบค่าเฉลี่ยการเข้าดูสื่อของผู้เรียนแต่ละระดับผลการเรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        chart_df = class_mean[
            ["VisITedResources"]
        ].reset_index()
        chart_df["Class_TH"] = (
            chart_df["Class"]
            .map(class_names)
        )
        fig_resources = px.bar(
            chart_df,
            x="Class_TH",
            y="VisITedResources",
            color="Class_TH",
            # 🎨 กำหนดสีแต่ละระดับ
            color_discrete_map=class_colors,
            text="VisITedResources",
            labels={
                "Class_TH": "ระดับผลการเรียน",
                "VisITedResources": "ค่าเฉลี่ยการเข้าดูสื่อ"
            },
            category_orders={
                "Class_TH": [
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ]
            }
        )
        fig_resources.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0
        )
        fig_resources.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="ระดับผลการเรียน",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                showgrid=False
            ),
            yaxis=dict(
                title="ค่าเฉลี่ยการเข้าดูสื่อ",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                gridcolor="#E2E8F0",
                zeroline=False
            ),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            )
        )
        st.plotly_chart(
            fig_resources,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ============================================================
    # 📊 กราฟที่ 3 : การดูประกาศ
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                📊 การดูประกาศกับระดับผลการเรียน
            </div>

            <div class="chart-description">
                เปรียบเทียบค่าเฉลี่ยการดูประกาศของผู้เรียนแต่ละระดับผลการเรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        chart_df = class_mean[
            ["AnnouncementsView"]
        ].reset_index()
        chart_df["Class_TH"] = (
            chart_df["Class"]
            .map(class_names)
        )
        fig_announcements = px.bar(
            chart_df,
            x="Class_TH",
            y="AnnouncementsView",
            color="Class_TH",
            # 🎨 กำหนดสีแต่ละระดับ
            color_discrete_map=class_colors,
            text="AnnouncementsView",
            labels={
                "Class_TH": "ระดับผลการเรียน",
                "AnnouncementsView": "ค่าเฉลี่ยการดูประกาศ"
            },
            category_orders={
                "Class_TH": [
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ]
            }
        )
        fig_announcements.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0
        )

        fig_announcements.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="ระดับผลการเรียน",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                showgrid=False
            ),
            yaxis=dict(
                title="ค่าเฉลี่ยการดูประกาศ",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                gridcolor="#E2E8F0",
                zeroline=False
            ),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            )
        )
        st.plotly_chart(
            fig_announcements,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ============================================================
    # 📊 กราฟที่ 4 : การอภิปราย
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                📊 การอภิปรายกับระดับผลการเรียน
            </div>
            <div class="chart-description">
                เปรียบเทียบค่าเฉลี่ยการอภิปรายของผู้เรียนแต่ละระดับผลการเรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        chart_df = class_mean[
            ["Discussion"]
        ].reset_index()
        chart_df["Class_TH"] = (
            chart_df["Class"]
            .map(class_names)
        )
        fig_discussion = px.bar(
            chart_df,
            x="Class_TH",
            y="Discussion",
            color="Class_TH",
            # 🎨 กำหนดสีแต่ละระดับ
            color_discrete_map=class_colors,
            text="Discussion",
            labels={
                "Class_TH": "ระดับผลการเรียน",
                "Discussion": "ค่าเฉลี่ยการอภิปราย"
            },
            category_orders={
                "Class_TH": [
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ]
            }
        )
        fig_discussion.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            marker_line_width=0
        )
        fig_discussion.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="ระดับผลการเรียน",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                showgrid=False
            ),
            yaxis=dict(
                title="ค่าเฉลี่ยการอภิปราย",
                tickfont=dict(
                    color="#64748B",
                    size=12
                ),
                gridcolor="#E2E8F0",
                zeroline=False
            ),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            )
        )
        st.plotly_chart(
            fig_discussion,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    # ============================================================
    # 🔍 ส่วนที่ 3 : คำอธิบายผลการวิเคราะห์
    # ============================================================
    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-title">
                🔍 คำอธิบายผลการวิเคราะห์
            </div>
            """,
            unsafe_allow_html=True
        )

        # ========================================================
        # 📌 หาค่าเฉลี่ยของแต่ละพฤติกรรมตามระดับผลการเรียน
        # ========================================================
        analysis_mean = (
            filtered_df
            .groupby("Class")[independent_cols]
            .mean()
            .reindex(["L", "M", "H"])
        )
        # ========================================================
        # 📌 คำนวณค่าความสัมพันธ์
        # ========================================================
        analysis_corr = filtered_df[
            independent_cols
        ].corr()

        # ========================================================
        # 📌 หาคู่ตัวแปรที่มีความสัมพันธ์สูงสุด
        # ========================================================
        max_corr = -1
        max_pair = None

        for i in range(len(independent_cols)):
            for j in range(i + 1, len(independent_cols)):
                corr_value = analysis_corr.iloc[i, j]
                if corr_value > max_corr:
                    max_corr = corr_value
                    max_pair = (
                        independent_cols[i],
                        independent_cols[j]
                    )
        max_var1 = variable_names[max_pair[0]]
        max_var2 = variable_names[max_pair[1]]

        # ========================================================
        # 📌 หาพฤติกรรมที่มีค่าเฉลี่ยสูงสุดในแต่ละระดับ
        # ========================================================
        highest_behavior = {}
        for class_code in class_order:
            row = analysis_mean.loc[class_code]
            highest_col = row.idxmax()
            highest_behavior[class_code] = (
                variable_names[highest_col],
                row[highest_col]
            )

        # ========================================================
        # 📌 ดึงค่าเฉลี่ยแต่ละพฤติกรรม
        # ========================================================
        raised_L = analysis_mean.loc[
            "L",
            "raisedhands"
        ]
        raised_M = analysis_mean.loc[
            "M",
            "raisedhands"
        ]
        raised_H = analysis_mean.loc[
            "H",
            "raisedhands"
        ]
        resources_L = analysis_mean.loc[
            "L",
            "VisITedResources"
        ]
        resources_M = analysis_mean.loc[
            "M",
            "VisITedResources"
        ]
        resources_H = analysis_mean.loc[
            "H",
            "VisITedResources"
        ]
        announcement_L = analysis_mean.loc[
            "L",
            "AnnouncementsView"
        ]
        announcement_M = analysis_mean.loc[
            "M",
            "AnnouncementsView"
        ]
        announcement_H = analysis_mean.loc[
            "H",
            "AnnouncementsView"
        ]
        discussion_L = analysis_mean.loc[
            "L",
            "Discussion"
        ]
        discussion_M = analysis_mean.loc[
            "M",
            "Discussion"
        ]
        discussion_H = analysis_mean.loc[
            "H",
            "Discussion"
        ]

        # ========================================================
        # 📝 แสดงผลการวิเคราะห์
        # ========================================================
        st.markdown(
            f"""
            <div class="analysis-note">
            <b>📌 ผลการวิเคราะห์จากข้อมูล</b>
            <br>
            จากข้อมูลพบว่า
            <b>พฤติกรรมการเรียนรู้มีแนวโน้มเพิ่มขึ้นตามระดับผลการเรียน</b>
            โดยเมื่อเปรียบเทียบค่าเฉลี่ยของผู้เรียนระดับต่ำ
            ระดับปานกลาง และระดับสูง
            พบว่าค่าของพฤติกรรมทั้ง 4 ด้าน
            มีแนวโน้มสูงขึ้นตามลำดับ
            <br>
            <b>1. การยกมือ</b><br>
            ระดับต่ำมีค่าเฉลี่ย<b> {raised_L:.2f}</b> ครั้ง
            ระดับปานกลางมีค่าเฉลี่ย <b>{raised_M:.2f}</b> ครั้ง
            และระดับสูงมีค่าเฉลี่ย <b>{raised_H:.2f}</b> ครั้ง
            <br>
            <b>2. การเข้าดูสื่อ</b><br>
            ระดับต่ำมีค่าเฉลี่ย <b>{resources_L:.2f}</b> ครั้ง
            ระดับปานกลางมีค่าเฉลี่ย <b>{resources_M:.2f}</b> ครั้ง
            และระดับสูงมีค่าเฉลี่ย <b>{resources_H:.2f}</b> ครั้ง
            <br>
            <b>3. การดูประกาศ</b><br>
            ระดับต่ำมีค่าเฉลี่ย <b>{announcement_L:.2f}</b> ครั้ง
            ระดับปานกลางมีค่าเฉลี่ย <b>{announcement_M:.2f}</b> ครั้ง
            และระดับสูงมีค่าเฉลี่ย <b>{announcement_H:.2f}</b> ครั้ง
            <br>
            <b>4. การอภิปราย</b><br>
            ระดับต่ำมีค่าเฉลี่ย <b>{discussion_L:.2f}</b> ครั้ง
            ระดับปานกลางมีค่าเฉลี่ย <b>{discussion_M:.2f}</b> ครั้ง
            และระดับสูงมีค่าเฉลี่ย <b>{discussion_H:.2f}</b> ครั้ง
            <br>
            <b>🔗 ความสัมพันธ์ระหว่างตัวแปรอิสระ</b><br>
            ตัวแปรที่มีความสัมพันธ์กันสูงที่สุดคือ <b>{max_var1}</b> กับ <b>{max_var2}</b>
            โดยมีค่าสหสัมพันธ์เท่ากับ <b>{max_corr:.2f}</b>
            <br>
            <b>💡 ข้อสังเกต</b><br>
            จากข้อมูลพบว่า ผู้เรียนที่อยู่ในระดับผลการเรียนสูง
            มีค่าเฉลี่ยพฤติกรรมการเรียนรู้ทั้ง 4 ด้าน สูงกว่าผู้เรียนระดับปานกลางและระดับต่ำอย่างชัดเจน
            โดยพฤติกรรมที่มีค่าเฉลี่ยสูงที่สุดของผู้เรียนเป็นดังนี้
            <b>ระดับสูง</b> คือ <b>{highest_behavior["H"][0]}</b>
            มีค่าเฉลี่ย <b>{highest_behavior["H"][1]:.2f}</b>
            <br>
            <b>หมายเหตุ:</b>
            ผลการวิเคราะห์นี้เป็นการแสดงความสัมพันธ์ และการเปรียบเทียบค่าเฉลี่ยของข้อมูล
            ไม่สามารถสรุปได้ว่าพฤติกรรมการเรียนรู้เป็นสาเหตุโดยตรงที่ทำให้ระดับผลการเรียนสูงขึ้น
            </div>
            """,
            unsafe_allow_html=True
        )
# ================================================================
# 🤖 การทำนายผลการเรียนของนักเรียน
# ================================================================
elif menu == "การทำนายผลการเรียนของนักเรียน":
    st.title("🤖 โมเดลทำนายผลการเรียนรู้")
    st.caption(
        "ใช้พฤติกรรมการเรียนรู้ของผู้เรียนเพื่อทำนายระดับผลการเรียนด้วย Random Forest"
    )
    st.markdown("---")
    # ============================================================
    # 1. กำหนดตัวแปรที่ใช้ในการทำนาย
    # ============================================================
    feature_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion",
        "StudentAbsenceDays"
    ]
    target_col = "Class"
    feature_labels = {
        "raisedhands": "การยกมือ",
        "VisITedResources": "การเข้าดูแหล่งเรียนรู้",
        "AnnouncementsView": "การดูประกาศ",
        "Discussion": "การอภิปราย",
        "StudentAbsenceDays": "จำนวนวันที่ขาดเรียน"
    }
    # สีของระดับผลการเรียน
    class_labels = {
        "L": "ระดับต่ำ",
        "M": "ระดับปานกลาง",
        "H": "ระดับสูง"
    }
    class_colors = {
        "L": "#FCA5A5",
        "M": "#FDE68A",
        "H": "#7BF3C3"
    }
    # ============================================================
    # 2. เตรียมข้อมูลสำหรับสร้างโมเดล
    # ============================================================
    model_df = df[
        feature_cols + [target_col]
    ].copy()
    # ============================================================
    # แปลงตัวแปรพฤติกรรมเป็นตัวเลข
    # ============================================================
    numeric_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion"
    ]
    for col in numeric_cols:
        model_df[col] = pd.to_numeric(
            model_df[col],
            errors="coerce"
        )
    # ============================================================
    # แปลงข้อมูลการขาดเรียน
    # Under-7  = ขาดเรียนน้อยกว่า 7 วัน
    # Above-7  = ขาดเรียนมากกว่า 7 วัน
    # ============================================================
    absence_mapping = {
        "Under-7": 0,
        "Above-7": 1
    }
    model_df["StudentAbsenceDays"] = (
        model_df["StudentAbsenceDays"]
        .map(absence_mapping)
    )
    # ============================================================
    # ลบข้อมูลที่ไม่สมบูรณ์
    # ============================================================
    model_df = model_df.dropna(
        subset=feature_cols + [target_col]
    )
    X = model_df[feature_cols]
    y = model_df[target_col].astype(str)

    # ============================================================
    # 3. แบ่งข้อมูล Train / Test
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ============================================================
    # 4. สร้างโมเดล Random Forest ตรวจสเปก model ในการสร้าง random forest 100 ต้น
    # ============================================================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    ## ให้ model เรียนรู้ข้อมูลในรูปแบบ จากข้อมูลชุดฝึก
    model.fit(   
        X_train,
        y_train
    )

    # ============================================================
    # 5. ทำนายข้อมูลชุดทดสอบ (model เรียนรู้) 
    # ============================================================
    pred = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        pred
    )

    # ============================================================
    # 6. KPI
    # ============================================================
    total_data = len(model_df)
    train_data = len(X_train)
    accuracy_percent = accuracy * 100
    # ============================================================
    # 🎨 CSS
    # ============================================================
    st.markdown(
        """
        <style>
        /* =========================================
           KPI Container
           ========================================= */
        .kpi-container {
            display: flex;
            flex-direction: row;
            gap: 15px;
            width: 100%;
            margin-bottom: 20px;
            flex-wrap: nowrap;
        }


        /* =========================================
           KPI Card
           ========================================= */
        .kpi-card {
            flex: 1 1 0;
            width: 33.33%;
            min-width: 0;
            background-color: #FFFFFF;
            border: 2px solid #03254C;
            border-radius: 50px;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            box-sizing: border-box;
        }
        /* =========================================
           วงกลมไอคอน
           ========================================= */
        .kpi-icon-circle {
            width: 55px;
            height: 55px;
            min-width: 55px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
        }
        /* สี KPI */
        .kpi-blue {
            background-color: #DBEAFE;
        }
       .kpi-green {
            background-color: #D1FAE5;
        }
        .kpi-orange {
            background-color: #FEF3C7;
        }
        /* =========================================
           ข้อมูล KPI
           ========================================= */
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
            font-size: 24px;
            font-weight: 700;
            margin-top: 3px;
            line-height: 1.2;
        }
        .kpi-unit {
            font-size: 16px;
            font-weight: 600;
            margin-left: 2px;
        }
        .kpi-description {
            color: #718096;
            font-size: 11px;
            margin-top: 2px;
            line-height: 1.2;            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # ============================================================
    # 📊 แสดง KPI
    # ============================================================
    st.markdown(
        f"""
        <div class="kpi-container">
            <!-- KPI 1 -->
            <div class="kpi-card">
            <div class="kpi-icon-circle kpi-blue">👥</div>
                <div class="kpi-info">
                    <div class="kpi-title">จำนวนข้อมูลทั้งหมด</div>
                    <div class="kpi-value">
                        {total_data:,}
                        <span class="kpi-unit">คน</span>
                    </div>
                    <div class="kpi-description">
                        100% ของข้อมูลทั้งหมด
                    </div>
                </div>
            </div>
            <!-- KPI 2 -->
            <div class="kpi-card">
            <div class="kpi-icon-circle kpi-green">🎓</div>
                <div class="kpi-info">
                    <div class="kpi-title">ข้อมูลสำหรับฝึกโมเดล</div>
                    <div class="kpi-value">
                        {train_data:,}
                        <span class="kpi-unit">คน</span>
                    </div>
                </div>
            </div>
            <!-- KPI 3 -->
            <div class="kpi-card">
            <div class="kpi-icon-circle kpi-orange">🎯</div>
                <div class="kpi-info">
                    <div class="kpi-title">ความแม่นยำของโมเดล</div>
                    <div class="kpi-value">
                        {accuracy_percent:.2f}
                        <span class="kpi-unit">%</span>
                    </div>
                    <div class="kpi-description">
                        ผลการทำนายถูกต้อง
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # 📊 Confusion Matrix
    # ============================================================
    with st.container(border=True):
        st.markdown(
            "### 📊 Confusion Matrix"
        )
        cm = confusion_matrix(
            y_test,
            pred,
            labels=["L", "M", "H"]
        )
        cm_df = pd.DataFrame(
            cm,
            index=[
                "ระดับต่ำ",
                "ระดับปานกลาง",
                "ระดับสูง"
            ],
            columns=[
                "ระดับต่ำ",
                "ระดับปานกลาง",
                "ระดับสูง"
            ]
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            labels={
                "x": "ผลการทำนาย",
                "y": "ผลจริง",
                "color": "จำนวนผู้เรียน"
            },
            color_continuous_scale=[
                "#D9F2E6",
                "#A7DDBF",
                "#6BC18E",
                "#3A9D63",
                "#1F6B43"
            ],
            zmin=0,
            zmax=cm.max()
        )
        fig_cm.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            )
        )
        fig_cm.update_traces(
            textfont=dict(
            size=16,
            color="#1E293B"
            )
        )
        st.plotly_chart(
            fig_cm,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
        # ============================================================
        # 🔍 คำอธิบาย Confusion Matrix จากข้อมูลในกราฟ
        # ============================================================
        # ชื่อระดับผลการเรียน
        cm_labels = {
            "L": "ระดับต่ำ",
            "M": "ระดับปานกลาง",
            "H": "ระดับสูง"
        }
        # ------------------------------------------------------------
        # จำนวนการทำนายถูกต้องของแต่ละระดับ
        # ------------------------------------------------------------
        correct_low = int(cm[0, 0])
        correct_mid = int(cm[1, 1])
        correct_high = int(cm[2, 2])
        # ------------------------------------------------------------
        # จำนวนทั้งหมดของแต่ละระดับจริง
        # ------------------------------------------------------------
        actual_low = int(cm[0, :].sum())
        actual_mid = int(cm[1, :].sum())
        actual_high = int(cm[2, :].sum())
        # ------------------------------------------------------------
        # หาเปอร์เซ็นต์การทำนายถูกต้องในแต่ละระดับ
        # ------------------------------------------------------------
        low_percent = (
            correct_low / actual_low * 100
            if actual_low > 0 else 0
        )
        mid_percent = (
            correct_mid / actual_mid * 100
            if actual_mid > 0 else 0
        )
        high_percent = (
            correct_high / actual_high * 100
            if actual_high > 0 else 0
        )       
        # ------------------------------------------------------------
        # หา cell ที่มีจำนวนมากที่สุดใน Confusion Matrix
        # ------------------------------------------------------------
        max_row, max_col = divmod(cm.argmax(), cm.shape[1])
        actual_class = cm_labels[
            ["L", "M", "H"][max_row]
        ]
        predicted_class = cm_labels[
            ["L", "M", "H"][max_col]
        ]
        max_value = int(cm[max_row, max_col])
        # ------------------------------------------------------------
        # แสดงคำอธิบาย
        # ------------------------------------------------------------
        st.markdown(
            f"""
            <div style="
                background-color:#F8FAFC;
                border:1px solid #E2E8F0;
                border-left:4px solid #3B82F6;
                border-radius:8px;
                padding:12px 15px;
                margin-top:10px;
                margin-bottom:20px;
                color:#334155;
                line-height:1.8;
                font-size:14px;
            ">
            💡 <strong>คำอธิบายจาก Confusion Matrix:</strong><br>
            โมเดลทำนายผู้เรียนที่มีผลการเรียน
            <b>ระดับต่ำ</b> ถูกต้อง
            <b>{correct_low:,} คน</b>
            จากทั้งหมด {actual_low:,} คน
            คิดเป็น <b>{low_percent:.1f}%</b><br>
            ระดับปานกลางทำนายถูกต้อง
            <b>{correct_mid:,} คน</b>
            จากทั้งหมด {actual_mid:,} คน
            คิดเป็น <b>{mid_percent:.1f}%</b><br>
            ระดับสูงทำนายถูกต้อง
            <b>{correct_high:,} คน</b>
            จากทั้งหมด {actual_high:,} คน
            คิดเป็น <b>{high_percent:.1f}%</b><br>
            โดยช่องที่มีจำนวนข้อมูลมากที่สุดในตารางคือ
            <b>{actual_class} → {predicted_class}</b>
            จำนวน <b>{max_value:,} คน</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    # ============================================================
    # 🌟 Feature Importance
    # ============================================================
    with st.container(border=True):
        st.markdown(
            "### 🌟 ความสำคัญของตัวแปรที่ใช้ในการทำนาย"
        )
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        })
        importance_df["ตัวแปร"] = (
            importance_df["Feature"]
            .map(feature_labels)
        )
        importance_df = importance_df.sort_values(
            "Importance",
            ascending=True
        )
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="ตัวแปร",
            orientation="h",
            text="Importance"
        )
        fig_importance.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker=dict(
                color="#F97AB6",
                cornerradius=6
            )
        )
        fig_importance.update_layout(
            height=400,
            template="plotly_white",
            showlegend=False,
            xaxis=dict(
                title="ค่าความสำคัญของตัวแปร",
                range=[
                    0,
                    max(
                        importance_df["Importance"].max() * 1.25,
                        0.1
                    )
                ]
            ),
            yaxis=dict(
                title=""
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=60,
                t=20,
                b=20
            )
        )
        st.plotly_chart(
            fig_importance,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )
        # ============================================================
        # 💡 รายละเอียดตัวแปร
        # ============================================================
        st.markdown(
            """
            <div style="
                background-color:#F8FAFC;
                border:1px solid #E2E8F0;
                border-left:4px solid #3B82F6;
                border-radius:8px;
                padding:12px 15px;
                color:#334155;
                line-height:1.7;
                font-size:14px;
            ">
            💡 <strong>คำอธิบาย:</strong><br>
                โมเดล Random Forest ใช้ข้อมูลพฤติกรรมการเรียนรู้
                5 ตัวแปร ได้แก่ การยกมือ การเข้าดูแหล่งเรียนรู้
                การดูประกาศ การอภิปราย และการขาดเรียน
                เพื่อทำนายระดับผลการเรียนของผู้เรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
    # ============================================================
    # 🔮 ส่วนทำนายผลผู้เรียนใหม่
    # ============================================================
    with st.container(border=True):
        st.markdown(
            "### 🔮 ทำนายระดับผลการเรียนของผู้เรียน"
        )
        st.markdown(
            """
            <div style="
            background-color:#EFF6FF;
            border:1px solid #BFDBFE;
            border-left:4px solid #3B82F6;
            border-radius:8px;
            padding:12px 15px;
            margin-bottom:18px;
            color:#334155;
            line-height:1.6;
            font-size:14px;
            ">
            กรอกข้อมูลพฤติกรรมการเรียนรู้ของผู้เรียน
            แล้วกดปุ่ม <strong>ทำนายผลการเรียน</strong>
            เพื่อให้โมเดลประเมินระดับผลการเรียน
            </div>
            """,
            unsafe_allow_html=True
        )
        # ============================================================
        # ค่าเริ่มต้นของช่องกรอก
        # ============================================================
        if "raisedhands_input" not in st.session_state:
            st.session_state["raisedhands_input"] = 0

        if "visited_input" not in st.session_state:
            st.session_state["visited_input"] = 0

        if "announcements_input" not in st.session_state:
            st.session_state["announcements_input"] = 0

        if "discussion_input" not in st.session_state:
            st.session_state["discussion_input"] = 0

        if "absence_input" not in st.session_state:
            st.session_state["absence_input"] = 0
        # ========================================================
        # ช่องกรอกข้อมูล
        # ========================================================
        col1, col2 = st.columns(2)
        with col1:
            raisedhands_input = st.number_input(
                "🙋 การยกมือ",
                min_value=0,
                max_value=100,
                step=1,
                key="raisedhands_input",
                help="จำนวนครั้งที่ผู้เรียนยกมือในชั้นเรียน"
            )
            visited_input = st.number_input(
                "📚 การเข้าดูแหล่งเรียนรู้",
                min_value=0,
                max_value=100,
                step=1,
                key="visited_input",
                help="จำนวนครั้งที่ผู้เรียนเข้าดูแหล่งเรียนรู้"
            )
            absence_input = st.number_input(
                "📅 การขาดเรียน",
                min_value=0,
                max_value=30,
                step=1,
                key="absence_input",
                help="จำนวนวันที่ผู้เรียนขาดเรียน"
            )
        with col2:
            announcements_input = st.number_input(
                "📢 การดูประกาศ",
                min_value=0,
                max_value=100,
                step=1,
                key="announcements_input",
                help="จำนวนครั้งที่ผู้เรียนดูประกาศ"
            )
            discussion_input = st.number_input(
                "💬 การร่วมอภิปราย",
                min_value=0,
                max_value=100,
                step=1,
                key="discussion_input",
                help="จำนวนครั้งที่ผู้เรียนมีส่วนร่วมในการอภิปราย"
            )
        st.markdown("<br>", unsafe_allow_html=True)
        # ========================================================
        # ปุ่มทำนาย
        # ========================================================
        def predict_student():
            # ====================================================
            # เตรียมข้อมูลผู้เรียน
            # ====================================================
            # แปลงจำนวนวันขาดเรียนให้ตรงกับข้อมูลที่ใช้ฝึกโมเดล
            if st.session_state["absence_input"] <= 7:
                absence_model_value = 0
            else:
                absence_model_value = 1
            # ====================================================
            # สร้างข้อมูลสำหรับทำนาย
            # ====================================================
            input_data = pd.DataFrame([
                {
                    "raisedhands": st.session_state["raisedhands_input"],
                    "VisITedResources": st.session_state["visited_input"],
                    "AnnouncementsView": st.session_state["announcements_input"],
                    "Discussion": st.session_state["discussion_input"],
                    "StudentAbsenceDays": absence_model_value
                }
            ])
            # ====================================================
            # ทำนาย
            # ====================================================
            predicted_class = model.predict(
                input_data
            )[0]
            probabilities = model.predict_proba(
                input_data
            )[0]
            # ====================================================
            # เก็บผลการทำนาย
            # ====================================================
            st.session_state["prediction_result"] = predicted_class
            st.session_state["prediction_probabilities"] = probabilities
            # ====================================================
            # รีเซ็ตช่องกรอกกลับเป็น 0
            # ====================================================
            st.session_state["raisedhands_input"] = 0
            st.session_state["visited_input"] = 0
            st.session_state["announcements_input"] = 0
            st.session_state["discussion_input"] = 0
            st.session_state["absence_input"] = 0
        st.markdown(
            """
            <style>
                div.stButton > button {
                    background-color: #33691e !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: 700 !important;
                }
                div.stButton > button:hover {
                    background-color: #2a5718 !important;
                    color: white !important;
                }   
            </style>
            """,
            unsafe_allow_html=True
        )
        predict_button = st.button(
            "🔮 ทำนายผลการเรียน",
            use_container_width=True,
            type="primary",
            on_click=predict_student
        )
        # ========================================================
        # แสดงผลการทำนาย
        # ========================================================
        if "prediction_result" in st.session_state:
            predicted_class = st.session_state["prediction_result"]
            probabilities = st.session_state["prediction_probabilities"]
            # ====================================================
            # แปลงชื่อระดับผลการเรียน
            # ====================================================
            predicted_label = class_labels.get(
                predicted_class,
                predicted_class
            )
            result_color = class_colors.get(
                predicted_class,
                "#7BF3C3"
            )
            # ====================================================
            # แสดงผลการทำนาย
            # ====================================================
            st.markdown(
                "<br>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    background-color:#FFFFFF;
                    border:2px solid {result_color};
                    border-radius:16px;
                    padding:20px;
                    text-align:center;
                    margin-bottom:20px;
                ">
                    <div style="
                        color:#64748B;
                        font-size:14px;
                        font-weight:600;
                    ">
                        ผลการทำนาย
                    </div>
                    <div style="
                        color:#334155;
                        font-size:14px;
                        margin-top:8px;
                    ">
                        ระดับผลการเรียนที่คาดการณ์
                    </div>
                    <div style="
                        color:#1E3A5F;
                        font-size:32px;
                        font-weight:800;
                        margin-top:5px;
                    ">
                        {predicted_label}
                    </div>
                    <div style="
                        color:#64748B;
                        font-size:13px;
                        margin-top:5px;
                    ">
                        รหัสระดับผลการเรียน: {predicted_class}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # ====================================================
            # 📊 ความน่าจะเป็นของแต่ละระดับ
            # ====================================================
            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })
            probability_df["ระดับผลการเรียน"] = (
                probability_df["Class"]
                .map(class_labels)
            )
            probability_df["ร้อยละ"] = (
                probability_df["Probability"] * 100
            )
            probability_df["ระดับผลการเรียน"] = pd.Categorical(
                probability_df["ระดับผลการเรียน"],
                categories=[
                    "ระดับต่ำ",
                    "ระดับปานกลาง",
                    "ระดับสูง"
                ],
                ordered=True
            )
            # ====================================================
            # 🔽 เรียงค่าความน่าจะเป็นจากน้อย → มาก
            # ====================================================
            probability_df = (
                probability_df
                .sort_values("ร้อยละ", ascending=True)
                .reset_index(drop=True)
            )
            # ====================================================
            # กราฟความน่าจะเป็น
            # ====================================================
            st.markdown(
                "### 📊 ความน่าจะเป็นของแต่ละระดับผลการเรียน"
            )
            class_colors = {
                "ระดับต่ำ": "#b0120a",
                "ระดับปานกลาง": "#f57f17",
                "ระดับสูง": "#33691e"
            }
            fig_probability = px.bar(
                probability_df,
                x="ระดับผลการเรียน",
                y="ร้อยละ",
                text="ร้อยละ",
                color="ระดับผลการเรียน",
                color_discrete_map=class_colors,
                category_orders={
                    "ระดับผลการเรียน": probability_df["ระดับผลการเรียน"].tolist()
                }
            )
            fig_probability.update_traces(
                texttemplate="<b>%{y:.1f}%</b>",
                textposition="outside"
            )
            fig_probability.update_layout(
                height=400,
                template="plotly_white",
                showlegend=False,
                yaxis=dict(
                    title="ความน่าจะเป็น (%)",
                    range=[
                        0,
                        110
                    ]
                ),
                xaxis=dict(
                    title=""
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=30
                )
            )
            st.plotly_chart(
                fig_probability,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
            # ====================================================
            # 💡 คำอธิบายกราฟ
            # ====================================================
            st.markdown(
                """
                <div style="
                    background-color:#F8FAFC;
                    border:1px solid #E2E8F0;
                    border-left:4px solid #3B82F6;
                    border-radius:8px;
                    padding:12px 15px;
                    margin-top:10px;
                    margin-bottom:20px;
                    color:#334155;
                    line-height:1.7;
                    font-size:14px;
                ">
                    💡 <strong>คำอธิบาย:</strong><br>
                    กราฟแสดงความน่าจะเป็นที่โมเดล Random Forest
                    คาดการณ์ว่าผู้เรียนจะอยู่ในแต่ละระดับผลการเรียน
                    โดยค่าร้อยละที่สูงกว่าแสดงถึงระดับที่โมเดลมีความมั่นใจมากกว่า
                    และระดับที่มีค่าความน่าจะเป็นสูงสุดจะเป็นผลการทำนายของโมเดล
                </div>
                """,
                unsafe_allow_html=True
            )