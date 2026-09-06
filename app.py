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
        "MiddleSchool": "มัธยมตอนต้น", 
        "HighSchool": "มัธยมตอนปลาย", 
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
                    [0.00, "#F8F7FF"],  # อ่อนสุด (ม่วงสว่างสบายตา)
                    [0.25, "#D0D7DE"],  # อ่อน (ม่วงเทานุ่มนวล)
                    [0.50, "#8C92AC"],  # กลาง (ม่วงหม่นเย็น/Cool Slate Purple)
                    [0.75, "#5A639C"],  # เข้ม (ม่วงครามเย็นตา)
                    [1.00, "#2B3160"]   # เข้มสุด (ม่วงน้ำเงินเข้มลึก)
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
            # 🔢 เรียงช่วงชั้นตามลำดับการศึกษา
            # ================================================================
            stage_order = [
                "ประถม",
                "มัธยมต้น",
                "มัธยมปลาย"
            ]
            stage_count["Stage"] = pd.Categorical(
                stage_count["Stage"],
                categories=stage_order,
                ordered=True
            )
            stage_count = stage_count.sort_values(
                "Stage"
                ).reset_index(drop=True)
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
            # 🔽 เรียงระดับชั้น
            # ================================================================
            grade_order = [
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
            grade_count["Grade"] = pd.Categorical(
                grade_count["Grade"],
                categories=grade_order,
                ordered=True
            )
            grade_count = (
                grade_count
                .sort_values("Grade")
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
                        "GradeLabel": [
                        "ชั้นประถมศึกษาปีที่ 1",
                        "ชั้นประถมศึกษาปีที่ 2",
                        "ชั้นประถมศึกษาปีที่ 3",
                        "ชั้นประถมศึกษาปีที่ 4",
                        "ชั้นประถมศึกษาปีที่ 5",
                        "ชั้นประถมศึกษาปีที่ 6",
                        "ชั้นมัธยมศึกษาปีที่ 1",
                        "ชั้นมัธยมศึกษาปีที่ 2",
                        "ชั้นมัธยมศึกษาปีที่ 3",
                        "ชั้นมัธยมศึกษาปีที่ 4",
                        "ชั้นมัธยมศึกษาปีที่ 5",
                        "ชั้นมัธยมศึกษาปีที่ 6"
                    ]
                },
                color_discrete_sequence=[
                    "#E6F4F1",  # เขียวพาสเทลอ่อนมาก
                    "#CCECE6",  # เขียวมิ้นต์อ่อน
                    "#99D8C9",  # เขียวมิ้นต์
                    "#66C2A4",  # เขียวหยกอ่อน
                    "#41AE76",  # เขียวมรกตสด
                    "#238B45",  # เขียวมรกตกลาง
                    "#006D2C",  # เขียวมรกตเข้ม
                    "#00441B",  # เขียวไพน์เข้ม
                    "#003615",  # เขียวอมฟ้าเข้ม
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
    # 2. โซนกลาง ( Grouped Bar Chart ตามช่วงชั้น )
    # =====================================================
    with st.container(border=True):
        st.markdown(
            """
            <h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>การกระจายผลสัมฤทธิ์ทางการเรียนตามช่วงชั้น</h3>
            <p style='color: #64748B; font-size: 0.8rem; margin-top: 2px; margin-bottom: 10px;'>
                สัดส่วนจำแนกตามกลุ่มผลสัมฤทธิ์ (High / Medium / Low) แบ่งตามช่วงชั้น • <i>เกณฑ์: H (70-100%), M (40-69%), L (&lt;40%)</i>
            </p>
        """,
            unsafe_allow_html=True,
        )

        if "Class" in filtered_df.columns and not filtered_df.empty:
            # Map ชื่อช่วงชั้นภาษาไทย (ถ้ายังไม่มี)
            stage_map = {
                "lowerlevel": "ประถม",
                "MiddleSchool": "มัธยมต้น",
                "HighSchool": "มัธยมปลาย",
            }
            if "StageID" in filtered_df.columns:
                filtered_df["Stage_TH"] = (
                    filtered_df["StageID"]
                    .map(stage_map)
                    .fillna(filtered_df["StageID"])
                )

            # Map ชื่อระดับผลสัมฤทธิ์ภาษาไทย
            class_label_map = {
                "H": "ระดับสูง (H)",
                "M": "ระดับปานกลาง (M)",
                "L": "ระดับต่ำ (L)",
            }
            filtered_df["Class_TH"] = filtered_df["Class"].map(class_label_map)

            # Groupby คำนวณจำนวนและสัดส่วน %
            stage_class_df = (
                filtered_df.groupby(
                    ["Stage_TH", "Class", "Class_TH"], as_index=False
                )
                .size()
                .rename(columns={"size": "Count"})
            )

            # คำนวณ % เทียบในแต่ละช่วงชั้น
            total_per_stage = stage_class_df.groupby("Stage_TH")[
                "Count"
            ].transform("sum")
            stage_class_df["Percent"] = (
                stage_class_df["Count"] / total_per_stage * 100
            )

            # ลำดับช่วงชั้น และ ระดับผลสัมฤทธิ์
            stage_order = ["ประถม", "มัธยมต้น", "มัธยมปลาย"]
            class_order = [
                "ระดับสูง (H)",
                "ระดับปานกลาง (M)",
                "ระดับต่ำ (L)",
            ]

            # สร้าง Grouped Bar Chart
            fig_bar = px.bar(
                stage_class_df,
                x="Stage_TH",
                y="Count",
                color="Class_TH",
                barmode="group",
                text=stage_class_df["Percent"].round(1).astype(str) + "%",
                category_orders={
                    "Stage_TH": stage_order,
                    "Class_TH": class_order,
                },
                color_discrete_map={
                    "ระดับสูง (H)": "#7BF3C3",    # เขียวมิ้นต์เย็นสบายตา (Cool Mint)
                    "ระดับปานกลาง (M)": "#FDE68A",  # เหลืองพาสเทลนุ่มนวล (Soft Warm Yellow)
                    "ระดับต่ำ (L)": "#FCA5A5"     # แดงพาสเทลละมุน (Soft Pastel Red)
                }
            )

            fig_bar.update_traces(
                textposition="outside",
                textfont=dict(size=10),
                cliponaxis=False,
                hovertemplate="<b>%{x} - %{fullData.name}</b><br>จำนวน: <b>%{y:,} คน</b> (%{text})<extra></extra>",
            )

            fig_bar.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title_text="",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            fig_bar.update_xaxes(
                title_text="ช่วงชั้น", tickfont=dict(size=11), showgrid=False
            )
            fig_bar.update_yaxes(
                title_text="จำนวนนักเรียน (คน)",
                tickfont=dict(size=10),
                gridcolor="#E2E8F0",
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                config={"displayModeBar": False},
                key="chart_bar_stage_academic",
            )

            # คำนวณสรุปข้อมูลแบบ Dynamic สำหรับ Insights
            total_students = len(filtered_df)
            counts = filtered_df["Class"].value_counts()
            count_m = counts.get("M", 0)
            count_l = counts.get("L", 0)
            pct_m = (
                (count_m / total_students * 100) if total_students > 0 else 0
            )
            pct_l = (
                (count_l / total_students * 100) if total_students > 0 else 0
            )

            # กล่องสรุป Insights
            st.markdown(
                f"""
                <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; border-radius: 12px; padding: 12px 16px; margin-top: 5px; margin-bottom: 10px; box-shadow: 0px 1px 3px rgba(0,0,0,0.03);'>
                    <p style='color: #1E293B; font-size: 0.83rem; margin: 0; line-height: 1.5;'>
                        💡 <strong>ข้อสังเกตและการนำไปใช้:</strong><br>
                        • นักเรียนส่วนใหญ่อยู่ใน <strong>ระดับปานกลาง (M) {pct_m:.1f}% ({count_m:,} คน)</strong> ซึ่งเป็นกลุ่มเป้าหมายสำคัญที่มีศักยภาพในการยกระดับผลสัมฤทธิ์ขึ้นสู่ระดับสูง<br>
                        • กลุ่มที่ต้องได้รับการดูแลเร่งด่วนคือ <strong>ระดับต่ำ (L) {pct_l:.1f}% ({count_l:,} คน)</strong> เพื่อวางมาตรการช่วยเหลือและลดอัตราการเรียนตกค้าง
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.info("ไม่พบข้อมูลผลสัมฤทธิ์ทางการเรียนสำหรับแสดงผล")

    # =================================================
    # โซน วิชา
    # =================================================
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
        "Quran": "อัลกุรอาน",
    }

    stage_map = {
        "LowerLevel": "ประถม",
        "MiddleSchool": "มัธยมต้น",
        "HighSchool": "มัธยมปลาย",
    }

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
        "G-12": "ม.6",
    }

    semester_map = {"F": "ภาคการศึกษาที่ 1", "S": "ภาคการศึกษาที่ 2"}

    # =================================================
    # 2. กำหนดลำดับช่วงชั้น ระดับชั้น และวิชาแบบกำหนดเอง
    # =================================================
    topic_order = [
        "เทคโนโลยี",
        "คณิตศาสตร์",
        "วิทยาศาสตร์",
        "ภาษาอังกฤษ",
        "ภาษาฝรั่งเศส",
        "ภาษาสเปน",
        "ภาษาอาหรับ",
        "ชีววิทยา",
        "เคมี",
        "ธรณีวิทยา",
        "ประวัติศาสตร์",
        "อัลกุรอาน",
    ]

    stage_order = ["ประถม", "มัธยมต้น", "มัธยมปลาย"]
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
        "ม.6",
    ]

    # =================================================
    # 3. จัดเตรียมข้อมูล (สร้าง stage_topic_df และ grade_topic_df)
    # =================================================
    if "df" in locals() and isinstance(df, pd.DataFrame) and not df.empty:
        if "Engagement" not in df.columns and all(
            c in df.columns
            for c in [
                "raisedhands",
                "VisITedResources",
                "AnnouncementsView",
                "Discussion",
            ]
        ):
            df["Engagement"] = (
                df["raisedhands"]
                + df["VisITedResources"]
                + df["AnnouncementsView"]
                + df["Discussion"]
            ) / 4

        if "Topic" in df.columns:
            df["Topic_TH"] = df["Topic"].map(topic_map).fillna(df["Topic"])
        if "StageID" in df.columns:
            df["Stage_TH"] = df["StageID"].map(stage_map).fillna(df["StageID"])
        if "GradeID" in df.columns:
            df["Grade_TH"] = df["GradeID"].map(grade_map).fillna(df["GradeID"])
        if "Semester" in df.columns:
            df["Semester_TH"] = (
                df["Semester"].map(semester_map).fillna(df["Semester"])
            )

        if all(c in df.columns for c in ["Stage_TH", "Topic_TH", "Semester_TH"]):
            stage_topic_df = (
                df.groupby(["Stage_TH", "Topic_TH", "Semester_TH"], as_index=False)
                .agg(Avg_Engagement=("Engagement", "mean"))
            )
        else:
            stage_topic_df = pd.DataFrame()

        if all(c in df.columns for c in ["Grade_TH", "Topic_TH", "Semester_TH"]):
            grade_topic_df = (
                df.groupby(["Grade_TH", "Topic_TH", "Semester_TH"], as_index=False)
                .agg(Avg_Engagement=("Engagement", "mean"))
            )
        else:
            grade_topic_df = pd.DataFrame()
    else:
        stage_topic_df = pd.DataFrame()
        grade_topic_df = pd.DataFrame()

    # =================================================
    # 4. แสดงผลกราฟช่วงชั้น (STAGE)
    # =================================================
    with st.container(border=True):
        st.markdown(
            """
            <div style='margin-bottom:10px;'>
            <h3 style='font-size:1.15rem; font-weight:700; color:#1E293B; margin:0;'>
            การมีส่วนร่วมตามช่วงชั้น
            </h3>
            <p style='color:#64748B; font-size:0.82rem; margin-top:3px; margin-bottom:10px;'>
            เปรียบเทียบการมีส่วนร่วมในแต่ละรายวิชา ระหว่างภาคการศึกษาที่ 1 และภาคการศึกษาที่ 2
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            isinstance(stage_topic_df, pd.DataFrame)
            and not stage_topic_df.empty
            and "Stage_TH" in stage_topic_df.columns
        ):
            available_stages = [
                s for s in stage_order if s in stage_topic_df["Stage_TH"].unique()
            ]
        else:
            available_stages = []

        for stage in available_stages:
            stage_data = stage_topic_df[
                stage_topic_df["Stage_TH"] == stage
            ].copy()

            st.markdown(
                f"""
                <div style='font-size:1rem; font-weight:700; color:#1E293B; margin-top:15px; margin-bottom:5px;'>
                {stage}
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_stage = px.bar(
                stage_data,
                x="Topic_TH",
                y="Avg_Engagement",
                color="Semester_TH",
                barmode="group",
                text=stage_data["Avg_Engagement"].round(0).astype(int).astype(str)
                + "%",
                category_orders={
                    "Topic_TH": topic_order,
                    "Semester_TH": ["ภาคการศึกษาที่ 1", "ภาคการศึกษาที่ 2"],
                },
                color_discrete_map={
                    "ภาคการศึกษาที่ 1": "#35A6DB",
                    "ภาคการศึกษาที่ 2": "#383AB5",
                },
            )

            fig_stage.update_traces(
                textposition="outside",
                textfont=dict(size=8),
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>การมีส่วนร่วมเฉลี่ย: <b>%{y:.1f}</b> คะแนน<extra></extra>",
            )

            fig_stage.update_layout(
                height=350,
                showlegend=False,
                legend_title="",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=35, b=70),
            )

            fig_stage.update_xaxes(
                title_text="", tickangle=-45, tickfont=dict(size=8), showgrid=False
            )

            fig_stage.update_yaxes(
                title_text="",
                range=[0, 100],
                tickfont=dict(size=8),
                gridcolor="#E2E8F0",
            )

            st.plotly_chart(
                fig_stage,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"information_stage_topic_{stage}",
            )

        # คำอธิบายสี
        st.markdown(
            """
            <div style="display:flex; justify-content:center; align-items:center; gap:25px; margin-top:-5px; margin-bottom:20px; font-size:13px; color:#475569;">
            <div style="display:flex; align-items:center;">
            <span style="width:12px; height:12px; background:#35A6DB; border-radius:2px; display:inline-block; margin-right:7px;"></span>
            ภาคการศึกษาที่ 1
            </div>
            
            <div style="display:flex; align-items:center;">
            <span style="width:12px; height:12px; background:#383AB5; border-radius:2px; display:inline-block; margin-right:7px;"></span>
            ภาคการศึกษาที่ 2
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )   

        # Dynamic Data Insights
        if isinstance(stage_topic_df, pd.DataFrame) and not stage_topic_df.empty:
            max_row = stage_topic_df.loc[stage_topic_df["Avg_Engagement"].idxmax()]
            min_row = stage_topic_df.loc[stage_topic_df["Avg_Engagement"].idxmin()]

            top_stage_info = (
                stage_topic_df.groupby("Stage_TH")["Avg_Engagement"]
                .mean()
                .reset_index()
                .sort_values("Avg_Engagement", ascending=False)
                .iloc[0]
            )

            sem_stage_avg = stage_topic_df.groupby("Semester_TH")[
                "Avg_Engagement"
            ].mean()
            s1_avg = sem_stage_avg.get("ภาคการศึกษาที่ 1", 0)
            s2_avg = sem_stage_avg.get("ภาคการศึกษาที่ 2", 0)

            if s2_avg > s1_avg:
                sem_trend = f"การมีส่วนร่วมเฉลี่ยในภาคการศึกษาที่ 2 สูงกว่าภาคการศึกษาที่ 1 ({s2_avg:.1f}% vs {s1_avg:.1f}%)"
            elif s1_avg > s2_avg:
                sem_trend = f"การมีส่วนร่วมเฉลี่ยในภาคการศึกษาที่ 1 สูงกว่าภาคการศึกษาที่ 2 ({s1_avg:.1f}% vs {s2_avg:.1f}%)"
            else:
                sem_trend = (
                    f"อัตราการมีส่วนร่วมเฉลี่ยทั้งสองภาคการศึกษาเท่ากันที่ {s1_avg:.1f}%"
                )

            st.markdown(
                f"""
                <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:15px; margin-top:10px;">
                    <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                        📌 สรุปตามช่วงชั้น
                    </div>
                    <ul style="margin:0; padding-left:20px; font-size:0.85rem; color:#334155; line-height:1.6;">
                        <li><b>วิชาที่มีการมีส่วนร่วมสูงสุด:</b> วิชา {max_row['Topic_TH']} ช่วงชั้น {max_row['Stage_TH']} ({max_row['Semester_TH']}) เฉลี่ยสูงถึง <b>{max_row['Avg_Engagement']:.1f}%</b></li>
                        <li><b>วิชาที่มีการมีส่วนร่วมต่ำสุด:</b> วิชา {min_row['Topic_TH']} ช่วงชั้น {min_row['Stage_TH']} ({min_row['Semester_TH']}) เฉลี่ยอยู่ที่ <b>{min_row['Avg_Engagement']:.1f}%</b></li>
                        <li><b>ช่วงชั้นที่มีปฏิสัมพันธ์รวมสูงสุด:</b> ช่วงชั้น <b>{top_stage_info['Stage_TH']}</b> (คะแนนเฉลี่ยรวม {top_stage_info['Avg_Engagement']:.1f}%)</li>
                        <li><b>แนวโน้มรายภาคเรียน:</b> {sem_trend}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # -------------------------------------------------
        # เว้นระยะระหว่างกราฟ
        # -------------------------------------------------
        st.markdown(
            "<div style='height:20px;'></div>",
            unsafe_allow_html=True
        )
    # =================================================
    # กราฟระดับชั้น (2 คอลัมน์ต่อแถว + คำอธิบายอัตโนมัติด้านล่าง)
    # =================================================
    with st.container(border=True):
        st.markdown("""
        <div style='margin-bottom:10px;'>
            <h3 style='font-size:1.15rem; font-weight:700; color:#1E293B; margin:0;'>
            การมีส่วนร่วมตามระดับชั้น
            </h3>
            <p style='color:#64748B; font-size:0.82rem; margin-top:3px; margin-bottom:10px;'>
            เปรียบเทียบการมีส่วนร่วมในแต่ละรายวิชา ระหว่างภาคการศึกษาที่ 1 และภาคการศึกษาที่ 2
            </p>
        </div>
        """, unsafe_allow_html=True)

        available_grades = [
            g for g in grade_order 
            if g in grade_topic_df["Grade_TH"].unique()
        ]

        # -------------------------------------------------
        # วน Loop แสดงกราฟแท่งแบบ 2 คอลัมน์ต่อแถว
        # -------------------------------------------------
        for i in range(0, len(available_grades), 2):
            batch = available_grades[i:i+2]
            cols = st.columns(2)
            
            for j, grade in enumerate(batch):
                with cols[j]:
                    with st.container(border=True):
                        grade_data = grade_topic_df[
                            grade_topic_df["Grade_TH"] == grade
                        ].copy()

                        st.markdown(f"""
                        <div style='font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:5px;'>
                        ระดับชั้น {grade}
                        </div>
                        """, unsafe_allow_html=True)

                        fig_grade_single = px.bar(
                            grade_data,
                            x="Topic_TH",
                            y="Avg_Engagement",
                            color="Semester_TH",
                            barmode="group",
                            text=grade_data["Avg_Engagement"].round(0).astype(int).astype(str) + "%",
                            category_orders={
                                "Topic_TH": topic_order,
                                "Semester_TH": ["ภาคการศึกษาที่ 1", "ภาคการศึกษาที่ 2"]
                            },
                            color_discrete_map={
                                "ภาคการศึกษาที่ 1": "#35A6DB",
                                "ภาคการศึกษาที่ 2": "#383AB5"
                            }
                        )

                        fig_grade_single.update_traces(
                            textposition="outside",
                            textfont=dict(size=8),
                            cliponaxis=False,
                            hovertemplate="<b>%{x}</b><br>เฉลี่ย: <b>%{y:.1f}</b> คะแนน<extra></extra>"
                        )

                        fig_grade_single.update_layout(
                            height=260,
                            showlegend=False,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=5, r=5, t=25, b=35)
                        )

                        fig_grade_single.update_xaxes(
                            title_text="",
                            tickangle=-30,
                            tickfont=dict(size=8),
                            showgrid=False
                        )
                        fig_grade_single.update_yaxes(
                            title_text="",
                            range=[0, 100],
                            tickfont=dict(size=8),
                            gridcolor="#E2E8F0"
                        )

                        st.plotly_chart(
                            fig_grade_single,
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"information_grade_card_{grade}"
                        )

        # -------------------------------------------------
        # สัญลักษณ์อธิบายสี (Legend)
        # -------------------------------------------------
        st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; gap:25px; margin-top:15px; margin-bottom:15px; font-size:13px; color:#475569;">
            <div style="display:flex; align-items:center;">
                <span style="width:12px; height:12px; background:#35A6DB; border-radius:2px; display:inline-block; margin-right:7px;"></span>
                ภาคการศึกษาที่ 1
            </div>
            <div style="display:flex; align-items:center;">
                <span style="width:12px; height:12px; background:#383AB5; border-radius:2px; display:inline-block; margin-right:7px;"></span>
                ภาคการศึกษาที่ 2
            </div>
        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # คำอธิบายผลวิเคราะห์แบบ Dynamic (เปลี่ยนตามข้อมูลใน CSV)
        # =================================================
        if not grade_topic_df.empty:
            # คำนวณค่าทางสถิติ
            max_row = grade_topic_df.loc[grade_topic_df["Avg_Engagement"].idxmax()]
            min_row = grade_topic_df.loc[grade_topic_df["Avg_Engagement"].idxmin()]
            
            top_grade_info = (
                grade_topic_df.groupby("Grade_TH")["Avg_Engagement"]
                .mean()
                .reset_index()
                .sort_values("Avg_Engagement", ascending=False)
                .iloc[0]
            )

            sem_avg = grade_topic_df.groupby("Semester_TH")["Avg_Engagement"].mean()
            sem1 = sem_avg.get("ภาคการศึกษาที่ 1", 0)
            sem2 = sem_avg.get("ภาคการศึกษาที่ 2", 0)
            
            if sem2 > sem1:
                sem_compare_text = f"ภาพรวมภาคการศึกษาที่ 2 มีระดับการมีส่วนร่วมเฉลี่ยสูงกว่าภาคการศึกษาที่ 1 ({sem2:.1f}% vs {sem1:.1f}%)"
            elif sem1 > sem2:
                sem_compare_text = f"ภาพรวมภาคการศึกษาที่ 1 มีระดับการมีส่วนร่วมเฉลี่ยสูงกว่าภาคการศึกษาที่ 2 ({sem1:.1f}% vs {sem2:.1f}%)"
            else:
                sem_compare_text = f"ทั้งสองภาคการศึกษามีอัตราการมีส่วนร่วมเฉลี่ยเท่ากันที่ {sem1:.1f}%"

            # แสดงผลกล่องคำอธิบาย
            st.markdown(f"""
            <div style="background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:15px; margin-top: 10px; margin-bottom: 15px;">
                <div style="font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
                    📌 สรุปจากข้อมูล
                </div>
                <ul style="margin:0; padding-left:20px; font-size:0.85rem; color:#334155; line-height:1.6;">
                    <li><b>วิชาที่มีการมีส่วนร่วมสูงสุด:</b> วิชา {max_row['Topic_TH']} ชั้น {max_row['Grade_TH']} ({max_row['Semester_TH']}) เฉลี่ยสูงถึง <b>{max_row['Avg_Engagement']:.1f}%</b></li>
                    <li><b>วิชาที่มีการมีส่วนร่วมต่ำสุด:</b> วิชา {min_row['Topic_TH']} ชั้น {min_row['Grade_TH']} ({min_row['Semester_TH']}) เฉลี่ยอยู่ที่ <b>{min_row['Avg_Engagement']:.1f}%</b></li>
                    <li><b>ระดับชั้นที่มีปฏิสัมพันธ์รวมสูงสุด:</b> ชั้น <b>{top_grade_info['Grade_TH']}</b> (คะแนนเฉลี่ยรวม {top_grade_info['Avg_Engagement']:.1f}%)</li>
                    <li><b>แนวโน้มรายภาคเรียน:</b> {sem_compare_text}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# พฤติกรรมการเรียนรู้
# =====================================================
elif menu == "พฤติกรรมการเรียนรู้":

    st.title("Learning Behavior")
    st.caption("พฤติกรรมการเรียนรู้")
    st.markdown("---")

    # =================================================
    # ตรวจสอบข้อมูล
    # =================================================
    if filtered_df.empty:
        st.warning("ไม่พบข้อมูลสำหรับการแสดงผล")
        st.stop()

    behavior_df = filtered_df.copy()

    # =================================================
    # ตัวแปรพฤติกรรมการเรียนรู้
    # =================================================
    behavior_cols = [
        "raisedhands",
        "VisITedResources",
        "AnnouncementsView",
        "Discussion"
    ]

    # =================================================
    # CSS
    # =================================================
    st.markdown("""
    <style>
        /* =========================================
           KPI CARD
           ========================================= */
        .behavior-card {
            background-color: #FFFFFF;
            border: 2px solid #03254C;
            border-radius: 50px;
            padding: 12px 15px;
            min-height: 90px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            box-sizing: border-box;
        }
        /* =========================================
           วงกลมไอคอน
           ========================================= */
        .behavior-icon {
            width: 50px;
            height: 50px;
            min-width: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            flex-shrink: 0;
        }
        /* =========================================
           สีไอคอน
           ========================================= */
        .behavior-blue {
            background-color: #DBEAFE;
        }
        .behavior-green {
            background-color: #D1FAE5;
        }
        .behavior-orange {
            background-color: #FEF3C7;
        }
        .behavior-purple {
            background-color: #EDE9FE;
        }
        /* =========================================
           ข้อมูล KPI
           ========================================= */
        .behavior-info {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        /* =========================================
           ชื่อ KPI
           ========================================= */
        .behavior-title {
            font-size: 13px;
            color: #2D3748;
            font-weight: 600;
            line-height: 1.3;
            margin: 0;
            white-space: nowrap;
        }
        /* =========================================
           ค่า KPI
           ========================================= */
        .behavior-value {
            font-size: 20px;
            font-weight: 700;
            color: #2B6CB0;
            line-height: 1.2;
            margin-top: 3px;
            white-space: nowrap;
        }
    </style>
    """, unsafe_allow_html=True)

    # =================================================
    # คำนวณค่าเฉลี่ยพฤติกรรม
    # =================================================
    avg_raised = behavior_df["raisedhands"].mean()
    avg_resource = behavior_df["VisITedResources"].mean()
    avg_announcement = behavior_df["AnnouncementsView"].mean()
    avg_discussion = behavior_df["Discussion"].mean()

    # คะแนนการมีส่วนร่วมรวม
    behavior_df["ParticipationScore"] = (
        behavior_df["raisedhands"] +
        behavior_df["VisITedResources"] +
        behavior_df["AnnouncementsView"] +
        behavior_df["Discussion"]
    )

    # =================================================
    # KPI CARDS
    # =================================================
    card_cols = st.columns(4)
    cards = [
        ("✋",
            "behavior-blue",
            "จำนวนการยกมือ",
            f"{avg_raised:.1f} ครั้ง/คน"
        ),
        ("👁️",
            "behavior-green",
            "จำนวนการเข้าดูสื่อ",
            f"{avg_resource:.1f} ครั้ง/คน"
        ),
        ("📣",
            "behavior-orange",
            "จำนวนการดูประกาศ",
            f"{avg_announcement:.1f} ครั้ง/คน"
        ),
        ("💬",
            "behavior-purple",
            "จำนวนการมีส่วนร่วมอภิปราย",
            f"{avg_discussion:.1f} ครั้ง/คน"
        )
    ]
    for col, (icon, icon_color, title, value) in zip(card_cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="behavior-card">
                    <div class="behavior-icon {icon_color}">{icon}</div>
                    <div class="behavior-info">
                        <div class="behavior-title">{title}</div>
                        <div class="behavior-value">{value}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown("<br>", unsafe_allow_html=True)
    # =====================================================
    # เตรียมข้อมูลสำหรับกราฟ (Mapping & Ordering)
    # =====================================================
    behavior_df["Semester_Label"] = behavior_df["Semester"].map({
        "F": "ภาคการศึกษาที่ 1", 
        "S": "ภาคการศึกษาที่ 2"
    }).fillna(behavior_df["Semester"])

    stage_map = {
        "lowerlevel": "ประถม",
        "MiddleSchool": "มัธยมต้น",
        "HighSchool": "มัธยมปลาย"
    }
    behavior_df["Stage_Name"] = behavior_df["StageID"].map(stage_map).fillna(behavior_df["StageID"])
    stage_order = ["ประถม", "มัธยมต้น", "มัธยมปลาย"]

    class_map = {
        "L": "ระดับต่ำ (Low)",
        "M": "ระดับปานกลาง (Medium)",
        "H": "ระดับสูง (High)"
    }
    behavior_df["Class_Label"] = behavior_df["Class"].map(class_map).fillna(behavior_df["Class"])
    class_order = ["ระดับต่ำ (Low)", "ระดับปานกลาง (Medium)", "ระดับสูง (High)"]

    color_class_map = {
        "ระดับต่ำ (Low)": "#FCA5A5",
        "ระดับปานกลาง (Medium)": "#FDE68A",
        "ระดับสูง (High)": "#7BF3C3"
    }

    # ฟังก์ชั่นช่วยสร้างกราฟพฤติกรรม
    def create_behavior_effect_chart(df, y_col, title_text, y_label):
        avg_df = df.groupby(["Stage_Name", "Class_Label", "Semester_Label"])[y_col].mean().reset_index()
        fig = px.bar(
            avg_df,
            x="Stage_Name",
            y=y_col,
            color="Class_Label",
            barmode="group",
            facet_col="Semester_Label",
            facet_col_spacing=0.08,
            text_auto=".1f",
            title=title_text,
            labels={
                "Stage_Name": "ช่วงชั้น",
                y_col: y_label,
                "Class_Label": "ผลสัมฤทธิ์ทางการเรียน",
                "Semester_Label": ""
            },
            category_orders={
                "Stage_Name": stage_order,
                "Class_Label": class_order,
                "Semester_Label": ["ภาคการศึกษาที่ 1", "ภาคการศึกษาที่ 2"]
            },
            color_discrete_map=color_class_map,
            template="plotly_white"
        )
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.update_yaxes(range=[0, avg_df[y_col].max() * 1.25 if not avg_df.empty else 100])
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=390,
            margin=dict(l=30, r=20, t=60, b=70),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center", title_text="")
        )
        return fig, avg_df

    # ฟังก์ชั่นช่วยสร้างคำอธิบาย Insight
    def generate_insight_html(avg_df, col_name, behavior_label, unit_label="ครั้ง"):
        if avg_df.empty:
            return ""
        
        class_avg = avg_df.groupby("Class_Label")[col_name].mean()
        high_val = class_avg.get("ระดับสูง (High)", 0)
        low_val = class_avg.get("ระดับต่ำ (Low)", 0)
        diff_val = high_val - low_val
        
        sem_avg = avg_df.groupby("Semester_Label")[col_name].mean()
        sem1_val = sem_avg.get("ภาคการศึกษาที่ 1", 0)
        sem2_val = sem_avg.get("ภาคการศึกษาที่ 2", 0)
        
        sem_diff_text = ""
        if sem2_val > sem1_val:
            sem_diff_text = f"เพิ่มขึ้นในภาคการศึกษาที่ 2 ({sem2_val:.1f} {unit_label}) เมื่อเทียบกับภาคการศึกษาที่ 1 ({sem1_val:.1f} {unit_label})"
        elif sem1_val > sem2_val:
            sem_diff_text = f"สูงกว่าในภาคการศึกษาที่ 1 ({sem1_val:.1f} {unit_label}) เมื่อเทียบกับภาคการศึกษาที่ 2 ({sem2_val:.1f} {unit_label})"
        else:
            sem_diff_text = f"มีค่าใกล้เคียงกันทั้งสองภาคการศึกษา ({sem1_val:.1f} {unit_label})"

        stage_avg = avg_df.groupby("Stage_Name")[col_name].mean()
        max_stage = stage_avg.idxmax() if not stage_avg.empty else "-"
        max_stage_val = stage_avg.max() if not stage_avg.empty else 0

        html = f"""
        <div class="insight-box">
            <div class="insight-title">📌 สรุปข้อวิเคราะห์เชิงข้อมูล ({behavior_label})</div>
            • <b>ความสัมพันธ์กับผลสัมฤทธิ์:</b> ผู้เรียนกลุ่มที่มีผลสัมฤทธิ์ทางการเรียนระดับสูง (High) มีอัตรา{behavior_label}เฉลี่ยอยู่ที่ <b>{high_val:.1f} {unit_label}</b> ซึ่งสูงกว่ากลุ่มระดับต่ำ (Low) ที่มีเฉลี่ย <b>{low_val:.1f} {unit_label}</b> (ต่างกันประมาณ <b>{diff_val:.1f} {unit_label}</b>)<br>
            • <b>เปรียบเทียบภาคเรียน:</b> แนวโน้ม{behavior_label}{sem_diff_text}<br>
            • <b>ช่วงชั้นที่มีส่วนร่วมสูงสุด:</b> นักเรียนระดับชั้น <b>{max_stage}</b> มีสถิติ{behavior_label}สูงสุด โดยมีค่าเฉลี่ยอยู่ที่ <b>{max_stage_val:.1f} {unit_label}</b>
        </div>
        """
        return html

    # =========================================================================
    # 1. อัตราการขาดเรียน/การเข้าเรียน (StudentAbsenceDays)
    # =========================================================================
    with st.container(border=True):
        st.markdown("### อัตราการขาดเรียนที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        absence_map = {
            "Under-7": "ขาดน้อยกว่า 7 วัน ",
            "Above-7": "ขาดมากกว่า 7 วัน "
        }
        behavior_df["Absence_Label"] = behavior_df["StudentAbsenceDays"].map(absence_map).fillna(behavior_df["StudentAbsenceDays"])
        absence_df = behavior_df.groupby(["Absence_Label", "Class_Label", "Semester_Label"]).size().reset_index(name="StudentCount")

        fig1 = px.bar(
            absence_df,
            x="Absence_Label",
            y="StudentCount",
            color="Class_Label",
            barmode="group",
            facet_col="Semester_Label",
            facet_col_spacing=0.08,
            text="StudentCount",
            title="จำนวนนักเรียนจำแนกตามอัตราการขาดเรียน ผลสัมฤทธิ์ทางการเรียน และภาคการศึกษา",
            labels={
                "Absence_Label": "",  # ซ่อนชื่อแกน X จากตัว px.bar เพื่อไม่ให้เกิดคำซ้ำใต้แต่ละ Facet
                "StudentCount": "จำนวนนักเรียน (คน)",
                "Class_Label": "ผลสัมฤทธิ์ทางการเรียน",
                "Semester_Label": ""
            },
            category_orders={
                "Absence_Label": ["ขาดน้อยกว่า 7 วัน", "ขาดมากกว่า 7 วัน "],
                "Class_Label": class_order,
                "Semester_Label": ["ภาคการศึกษาที่ 1", "ภาคการศึกษาที่ 2"]
            },
            color_discrete_map=color_class_map,
            template="plotly_white"
        )
        fig1.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
        fig1.update_traces(textposition="outside")
        fig1.update_yaxes(range=[0, absence_df["StudentCount"].max() * 1.25 if not absence_df.empty else 10])
        # เพิ่มคำว่า "อัตราการขาดเรียน" แบบ centered ตรงกลางกราฟเพียงจุดเดียว
        fig1.update_layout(
            height=400,
            margin=dict(l=30, r=20, t=60, b=80),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center", title_text=""),
            annotations=list(fig1.layout.annotations) + [
                dict(
                    text="อัตราการขาดเรียน",
                    x=0.5,
                    y=-0.18,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#4A5568")
                )
            ]
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
        if not absence_df.empty:
            under7_high = absence_df[(absence_df["Absence_Label"].str.contains("น้อยกว่า")) & (absence_df["Class_Label"].str.contains("สูง"))]["StudentCount"].sum()
            above7_low = absence_df[(absence_df["Absence_Label"].str.contains("มากกว่า")) & (absence_df["Class_Label"].str.contains("ต่ำ"))]["StudentCount"].sum()
    
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 สรุปข้อวิเคราะห์อัตราการขาดเรียน (StudentAbsenceDays)</div>
                • <b>กลุ่มขาดเรียนน้อยกว่า 7 วัน:</b> มีสัดส่วนผู้เรียนระดับผลสัมฤทธิ์สูง (High) และปานกลาง (Medium) ครองสัดส่วนส่วนใหญ่ (พบกลุ่ม High รวม <b>{under7_high} คน</b>)<br>
                • <b>กลุ่มขาดเรียนมากกว่า 7 วัน:</b> มีความสัมพันธ์อย่างมีนัยสำคัญกับกลุ่มผลสัมฤทธิ์ระดับต่ำ (Low) โดยตรง (พบกลุ่ม Low รวม <b>{above7_low} คน</b>)<br>
                • <b>การนำไปใช้:</b> จำนวนวันขาดเรียนเกิน 7 วัน ถือเป็นดัชนีชี้วัดความเสี่ยงที่ครูผู้สอนควรติดตามและช่วยเหลือเป็นพิเศษ
            </div>
            """, 
            unsafe_allow_html=True
            )
    # =========================================================================
    # 2. การยกมือตอบคำถาม
    # =========================================================================
    with st.container(border=True):
        st.markdown("### การยกมือตอบคำถามที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        fig2, df2 = create_behavior_effect_chart(
            behavior_df, "raisedhands", 
            "ค่าเฉลี่ยการยกมือตอบคำถาม จำแนกตามช่วงชั้น ผลการเรียน และภาคการศึกษา", 
            "ค่าเฉลี่ยการยกมือ (ครั้ง)"
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(generate_insight_html(df2, "raisedhands", "การยกมือตอบคำถาม"), unsafe_allow_html=True)

    # =========================================================================
    # 3. การเข้าดูสื่อ
    # =========================================================================
    with st.container(border=True):
        st.markdown("### การเข้าดูสื่อการเรียนที่ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        fig3, df3 = create_behavior_effect_chart(
            behavior_df, "VisITedResources", 
            "ค่าเฉลี่ยการเข้าดูสื่อการเรียน จำแนกตามช่วงชั้น ผลการเรียน และภาคการศึกษา", 
            "ค่าเฉลี่ยการเข้าดูสื่อ (ครั้ง)"
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown(generate_insight_html(df3, "VisITedResources", "การเข้าดูสื่อการเรียน"), unsafe_allow_html=True)

    # =========================================================================
    # 4. การเข้าดูประกาศ
    # =========================================================================
    with st.container(border=True):
        st.markdown("### การเข้าดูประกาศ ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        fig4, df4 = create_behavior_effect_chart(
            behavior_df, "AnnouncementsView", 
            "ค่าเฉลี่ยการเข้าดูประกาศ จำแนกตามช่วงชั้น ผลการเรียน และภาคการศึกษา", 
            "ค่าเฉลี่ยการดูประกาศ (ครั้ง)"
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
        st.markdown(generate_insight_html(df4, "AnnouncementsView", "การเข้าดูประกาศข่าวสาร"), unsafe_allow_html=True)

    # =========================================================================
    # 5. การร่วมอภิปราย
    # =========================================================================
    with st.container(border=True):
        st.markdown("### การร่วมอภิปราย ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        fig5, df5 = create_behavior_effect_chart(
            behavior_df, "Discussion", 
            "ค่าเฉลี่ยการร่วมอภิปราย จำแนกตามช่วงชั้น ผลการเรียน และภาคการศึกษา", 
            "ค่าเฉลี่ยการร่วมอภิปราย (ครั้ง)"
        )
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
        st.markdown(generate_insight_html(df5, "Discussion", "การเข้าร่วมอภิปราย"), unsafe_allow_html=True)

    # =========================================================================
    # 6. ภาพรวม (คะแนนรวม 4 ด้าน)
    # =========================================================================
    with st.container(border=True):
        st.markdown("### ภาพรวมการมีส่วนร่วมทั้งหมด ส่งผลต่อระดับผลสัมฤทธิ์ทางการเรียน")
        fig6, df6 = create_behavior_effect_chart(
            behavior_df, "ParticipationScore", 
            "คะแนนการมีส่วนร่วมรวม (4 ด้าน) จำแนกตามช่วงชั้น ผลการเรียน และภาคการศึกษา", 
            "คะแนนการมีส่วนร่วมรวม (คะแนนเฉลี่ย)"
        )
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})
        st.markdown(generate_insight_html(df6, "ParticipationScore", "คะแนนการมีส่วนร่วมรวม", unit_label="คะแนน"), unsafe_allow_html=True)

    # =========================================================================
    # 7. เปรียบเทียบรายชั้นปี (ป.1 - ม.6)
    # =========================================================================
    with st.container(border=True):
        st.markdown("### การเปรียบเทียบการมีส่วนร่วมรายชั้นปี จำแนกตามภาคเรียน")
        
        grade_mapping = {
            "G-01": {"name": "ป.1", "order": 1},
            "G-02": {"name": "ป.2", "order": 2},
            "G-03": {"name": "ป.3", "order": 3},
            "G-04": {"name": "ป.4", "order": 4},
            "G-05": {"name": "ป.5", "order": 5},
            "G-06": {"name": "ป.6", "order": 6},
            "G-07": {"name": "ม.1", "order": 7},
            "G-08": {"name": "ม.2", "order": 8},
            "G-09": {"name": "ม.3", "order": 9},
            "G-10": {"name": "ม.4", "order": 10},
            "G-11": {"name": "ม.5", "order": 11},
            "G-12": {"name": "ม.6", "order": 12},
        }

        grade_df = behavior_df.groupby(["GradeID", "Semester_Label"])["ParticipationScore"].mean().reset_index()
        grade_df["Grade_Name"] = grade_df["GradeID"].map(lambda x: grade_mapping.get(x, {}).get("name", x))
        grade_df["Order"] = grade_df["GradeID"].map(lambda x: grade_mapping.get(x, {}).get("order", 99))
        grade_df = grade_df.sort_values(by="Order")

        fig7 = px.bar(
            grade_df,
            x="Grade_Name",
            y="ParticipationScore",
            color="Semester_Label",
            barmode="group",
            text_auto=".1f",
            title="คะแนนเฉลี่ยการมีส่วนร่วมรายระดับชั้น (ป.1 - ม.6) จำแนกตามภาคการศึกษา",
            labels={
                "Grade_Name": "ระดับชั้น",
                "ParticipationScore": "คะแนนการมีส่วนร่วมเฉลี่ย",
                "Semester_Label": "ภาคการศึกษา"
            },
            color_discrete_map={
                "ภาคการศึกษาที่ 1": "#35A6DB",
                "ภาคการศึกษาที่ 2": "#383AB5"
            },
            template="plotly_white"
        )

        fig7.update_traces(textposition="outside")
        fig7.update_yaxes(range=[0, grade_df["ParticipationScore"].max() * 1.18 if not grade_df.empty else 100])
        fig7.update_layout(
            height=400,
            margin=dict(l=30, r=20, t=60, b=70),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", title_text="")
        )

        st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

        if not grade_df.empty and grade_df["ParticipationScore"].notna().any():
            top_grade_row = grade_df.loc[grade_df["ParticipationScore"].idxmax()]
            low_grade_row = grade_df.loc[grade_df["ParticipationScore"].idxmin()]
            
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📌 สรุปข้อวิเคราะห์รายชั้นปี</div>
                • <b>ระดับชั้นที่มีส่วนร่วมสูงสุด:</b> ชั้น <b>{top_grade_row['Grade_Name']}</b> ใน{top_grade_row['Semester_Label']} มีคะแนนการมีส่วนร่วมเฉลี่ยสูงที่สุดอยู่ที่ <b>{top_grade_row['ParticipationScore']:.1f} คะแนน</b><br>
                • <b>ระดับชั้นที่มีส่วนร่วมน้อยที่สุด:</b> ชั้น <b>{low_grade_row['Grade_Name']}</b> ใน{low_grade_row['Semester_Label']} มีคะแนนการมีส่วนร่วมเฉลี่ยอยู่ที่ <b>{low_grade_row['ParticipationScore']:.1f} คะแนน</b><br>
                • <b>ข้อสังเกตเพิ่มเติม:</b> ช่วยให้ผู้สอนสามารถระบุระดับชั้นที่ต้องได้รับการกระตุ้นการมีส่วนร่วมหรือปรับเปลี่ยนรูปแบบกิจกรรมในแต่ละภาคเรียนได้อย่างแม่นยำ
            </div>
            """, unsafe_allow_html=True
            )
# =====================================================
# การมีส่วนร่วมของผู้ปกครอง (ดีไซน์ตามรูปตัวอย่าง)
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
    survey_rel = pd.crosstab(plot_df["Relation"], plot_df["ParentAnsweringSurvey"])
    sat_rel = pd.crosstab(plot_df["Relation"], plot_df["ParentschoolSatisfaction"])

    # ดึงค่าจำนวนคน
    f_surv_yes = survey_rel.loc["Father", "Yes"] if "Father" in survey_rel.index and "Yes" in survey_rel.columns else 0
    f_surv_no = survey_rel.loc["Father", "No"] if "Father" in survey_rel.index and "No" in survey_rel.columns else 0
    m_surv_yes = survey_rel.loc["Mum", "Yes"] if "Mum" in survey_rel.index and "Yes" in survey_rel.columns else 0
    m_surv_no = survey_rel.loc["Mum", "No"] if "Mum" in survey_rel.index and "No" in survey_rel.columns else 0

    f_sat_good = sat_rel.loc["Father", "Good"] if "Father" in sat_rel.index and "Good" in sat_rel.columns else 0
    f_sat_bad = sat_rel.loc["Father", "Bad"] if "Father" in sat_rel.index and "Bad" in sat_rel.columns else 0
    m_sat_good = sat_rel.loc["Mum", "Good"] if "Mum" in sat_rel.index and "Good" in sat_rel.columns else 0
    m_sat_bad = sat_rel.loc["Mum", "Bad"] if "Mum" in sat_rel.index and "Bad" in sat_rel.columns else 0

    # จัดโครงสร้างข้อมูลใหม่สำหรับ Grouped Bar Chart
    bar_data_rel = pd.DataFrame([
        # การตอบแบบสำรวจ
        {"หัวข้อ": "ตอบแบบสำรวจ (ตอบ)", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_surv_yes},
        {"หัวข้อ": "ตอบแบบสำรวจ (ตอบ)", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_surv_yes},
        {"หัวข้อ": "ตอบแบบสำรวจ (ไม่ตอบ)", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_surv_no},
        {"หัวข้อ": "ตอบแบบสำรวจ (ไม่ตอบ)", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_surv_no},
        # ความพึงพอใจ
        {"หัวข้อ": "ความพึงพอใจ (พึงพอใจ)", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_sat_good},
        {"หัวข้อ": "ความพึงพอใจ (พึงพอใจ)", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_sat_good},
        {"หัวข้อ": "ความพึงพอใจ (ไม่พึงพอใจ)", "ผู้ดูแลหลัก": "บิดา", "จำนวน": f_sat_bad},
        {"หัวข้อ": "ความพึงพอใจ (ไม่พึงพอใจ)", "ผู้ดูแลหลัก": "มารดา", "จำนวน": m_sat_bad},
    ])

    with st.container(border=True):
        st.markdown("<h3 style='margin-bottom:0px;'>📊 เปรียบเทียบการมีส่วนร่วมจำแนกตามผู้ดูแลหลัก (บิดา / มารดา)</h3>", unsafe_allow_html=True)
        st.caption("เปรียบเทียบการตอบแบบสำรวจและความพึงพอใจแยกระหว่างบิดาและมารดา")
        st.markdown("<br>", unsafe_allow_html=True)

        row2_left, row2_right = st.columns([1.5, 1])

        with row2_left:
            fig_bar = px.bar(
                bar_data_rel,
                x="หัวข้อ",
                y="จำนวน",
                color="ผู้ดูแลหลัก",
                barmode="group",
                text="จำนวน",
                color_discrete_map={"บิดา": "#2285B0", "มารดา": "#36BCAA"}, # คุมโทนสีตามแถวแรก
                labels={"หัวข้อ": "", "จำนวน": "จำนวน (คน)", "ผู้ดูแลหลัก": "ผู้ดูแลหลัก"}
            )
        
            fig_bar.update_traces(
                texttemplate='<b>%{y:,} คน</b>',
                textposition='outside',
                textfont=dict(family="Sarabun, sans-serif", size=12),
                hovertemplate="หัวข้อ: <b>%{x}</b><br>%{fullData.name}: <b>%{y:,} คน</b><extra></extra>"
            )
        
            max_y = bar_data_rel["จำนวน"].max()
            fig_bar.update_layout(
                height=390,
                bargap=0.15,         # ลดระยะห่างระหว่างกลุ่ม ปรับให้แท่งกราฟใหญ่/หนาขึ้น
                bargroupgap=0.05,    # ลดระยะห่างระหว่างแท่งในกลุ่มเดียวกัน
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Sarabun, sans-serif", size=13)
                ),
                margin=dict(t=30, b=40, l=10, r=10),
                font=dict(family="Sarabun, sans-serif", size=12),
                template="plotly_white",
                yaxis=dict(range=[0, max_y * 1.22], tickformat=",d")
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

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
                            💡 <b>ข้อสังเกต:</b> มารดามีสัดส่วนในการตอบแบบสำรวจ ({ (m_surv_yes/(m_surv_yes+m_surv_no))*100:.1f}% ) และความพึงพอใจ ({ (m_sat_good/(m_sat_good+m_sat_bad))*100:.1f}% ) สูงกว่าบิดาอย่างมีนัยสำคัญ
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

            # สรุป Insight ฝั่งซ้าย
            st.markdown(
                """
                <div style='background-color: #F7F8FF; padding: 10px 12px; border-radius: 8px; 
                border: 1px solid #FCD34D; margin-top: 0px; margin-bottom: 10px'>
                    <span style='color: #92400E; font-size: 12px; line-height: 1.4; display: block;'>
                        💡 <b>ข้อสังเกต:</b> นักเรียนที่มี <b>มารดา</b> ดูแลหลัก มีอัตราการเข้าดูสื่อการเรียน (69.1 ครั้ง) และยกมือตอบคำถามสูงกว่ามีบิดาดูแลหลักอย่างมีนัยสำคัญ
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    # -----------------------------------------------------
    # 👉 กราฟฝั่งขวา
    # -----------------------------------------------------
    with col_right:
        with st.container(border=True):
            ct_survey_class = pd.crosstab(
                plot_df["Class"],
                plot_df["ParentAnsweringSurvey"],
                normalize="index"
            ) * 100
            
            ct_survey_class = ct_survey_class.reset_index()
            ct_survey_class["Class"] = ct_survey_class["Class"].replace({
                "L": "ระดับต่ำ (L)",
                "M": "ระดับปานกลาง (M)",
                "H": "ระดับสูง (H)"
            })
            ct_survey_class["Class"] = pd.Categorical(ct_survey_class["Class"], categories=["ระดับต่ำ (L)", "ระดับปานกลาง (M)", "ระดับสูง (H)"], ordered=True)
            ct_survey_class = ct_survey_class.sort_values("Class")

            melted_survey = ct_survey_class.melt(id_vars=["Class"], var_name="ParentAnsweringSurvey", value_name="สัดส่วน (%)")
            melted_survey["ParentAnsweringSurvey"] = melted_survey["ParentAnsweringSurvey"].replace({"Yes": "ตอบแบบสำรวจ", "No": "ไม่ตอบแบบสำรวจ"})

            fig_class = px.bar(
                melted_survey,
                x="Class",
                y="สัดส่วน (%)",
                color="ParentAnsweringSurvey",
                barmode="group",
                text="สัดส่วน (%)",
                title="<b>🎓 ความสัมพันธ์กับผลสัมฤทธิ์ทางการเรียน</b>",
                color_discrete_map={"ตอบแบบสำรวจ": "#D3B820", "ไม่ตอบแบบสำรวจ": "#E6970F"}
            )
            fig_class.update_traces(
                texttemplate='<b>%{y:.1f}%</b>',
                textposition='outside',
                textfont=dict(family="Sarabun, sans-serif", size=11),
                hovertemplate="กลุ่มผลการเรียน: <b>%{x}</b><br>%{fullData.name}: <b>%{y:.1f}%</b><extra></extra>"
            )
            fig_class.update_layout(
                height=360,
                bargap=0.18,
                bargroupgap=0.06,
                margin=dict(t=40, b=60, l=10, r=10), # เพิ่ม margin b (ด้านล่าง) เป็น 60
                font=dict(family="Sarabun, sans-serif", size=11),
                template="plotly_white",
                xaxis=dict(title=""), # ลบชื่อแกน X ออกเพื่อไม่ให้ลอยมาซ้อนทับ Legend
                yaxis=dict(title="สัดส่วน (%)", range=[0, 118]),
                legend=dict(
                    title="", # ลบหัวข้อ Legend
                    orientation="h",
                    yanchor="top",
                    y=-0.18, # ดัน Legend ลงมาข้างล่าง
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_class, use_container_width=True, config={"displayModeBar": False})

            # สรุป Insight ฝั่งขวา
            st.markdown(
                """
                <div style='background-color: #F6F4FF; padding: 10px 12px; border-radius: 8px; 
                border: 1px solid #93C5FD; margin-top: 0px; margin-bottom: 10px;'>
                    <span style='color: #1E3A8A; font-size: 12px; line-height: 1.4; display: block;'>
                        🎯 <b>ข้อสังเกต:</b> กลุ่ม <b>ระดับสูง (High)</b> ผู้ปกครองตอบแบบสำรวจสูงถึง 80.3% ในขณะที่กลุ่ม <b>ระดับต่ำ (Low)</b> ผู้ปกครองไม่ตอบแบบสำรวจสูงถึง 78.0%
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
        "ระดับสูง": "#7BF3C3",       # 🟦 น้ำเงิน
        "ระดับปานกลาง": "#FDE68A",   # 🟨 เหลือง/ส้ม
        "ระดับต่ำ": "#FCA5A5"        # 🟥 แดง
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
    # 4. สร้างโมเดล Random Forest
    # ============================================================
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )

    # ============================================================
    # 5. ทำนายข้อมูลชุดทดสอบ
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
            "#EFF6FF",
            "#BFDBFE",
            "#60A5FA",
            "#2563EB",
            "#1E3A8A"
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
                color="#7BF3C3",
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
                "💬 การอภิปราย",
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
            probability_df = (
                probability_df
                .sort_values("ระดับผลการเรียน")
                .reset_index(drop=True)
            )
            # ====================================================
            # กราฟความน่าจะเป็น
            # ====================================================
            st.markdown(
                "### 📊 ความน่าจะเป็นของแต่ละระดับผลการเรียน"
            )
            fig_probability = px.bar(
                probability_df,
                x="ระดับผลการเรียน",
                y="ร้อยละ",
                text="ร้อยละ",
                color="ระดับผลการเรียน",
                # ใช้สีเดียวกับที่กำหนด
                color_discrete_map={
                    "ระดับต่ำ": "#FCA5A5",
                    "ระดับปานกลาง": "#FDE68A",
                    "ระดับสูง": "#7BF3C3"
                },
                category_orders={
                    "ระดับผลการเรียน": [
                        "ระดับต่ำ",
                        "ระดับปานกลาง",
                        "ระดับสูง"
                    ]
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