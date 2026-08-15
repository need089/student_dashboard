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
                    "เพศชาย": "#4F8EF7",
                    "เพศหญิง": "#FF6B9A"
                }
            )
            # ========================================================
            # 🎨 ปรับแต่งกราฟวงกลม
            # ========================================================
            fig_gender.update_traces(
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
                    [0.00, "#F5D0FE"],
                    [0.25, "#E9A8F0"],
                    [0.50, "#D77AE5"],
                    [0.75, "#B84FCF"],
                    [1.00, "#9333A8"]
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
                    [0.0, "#93C5FD"],   # อ่อนสุด แต่ยังเป็นฟ้า
                    [0.5, "#3B82F6"],   # กลาง
                    [1.0, "#1E3A8A"]    # เข้มสุด
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
                        "#0B351D",
                        "#0F4224",
                        "#124C2A",
                        "#14532D",
                        "#166534",
                        "#15803D",
                        "#16A34A",
                        "#22C55E",
                        "#4ADE80",
                        "#86EFAC",
                        "#BBF7D0",
                        "#DCFCE7"
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
                    "ภาคการศึกษาที่ 1": "#3B82F6",
                    "ภาคการศึกษาที่ 2": "#8B5CF6"
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
                        color:#2563EB;
                        font-size:22px;
                        font-weight:700;
                        margin-top:3px;
                    ">
                        {semester1_count:,.0f} คน
                    </div>

                    <div style="
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
                        color:#7C3AED;
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
    
    # ------------------------------------------------------------------------
    # 📦 ROW 5: SEARCH & DATA TABLE (แบ่งแถวด้วย Container มีเส้นขอบ)
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
    # 2. โซนกลาง ( Donut Chart )
    # =====================================================
    with st.container(border=True):
        st.markdown("""
            <h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>การกระจายผลสัมฤทธิ์ทางการเรียน</h3>
            <p style='color: #64748B; font-size: 0.8rem; margin-top: 2px; margin-bottom: 10px;'>
                สัดส่วนจำแนกตามกลุ่มผลสัมฤทธิ์ (High / Medium / Low) • <i>เกณฑ์: H (70-100%), M (40-69%), L (&lt;40%)</i>
            </p>
        """, unsafe_allow_html=True)
        
        if "Class" in filtered_df.columns and not filtered_df.empty:
            total_students = len(filtered_df)
        
            # 1. คำนวณจำนวนนักเรียนและ % แต่ละกลุ่มแบบ Dynamic
            counts = filtered_df["Class"].value_counts()
            count_h = counts.get("H", 0)
            count_m = counts.get("M", 0)
            count_l = counts.get("L", 0)
        
            pct_h = (count_h / total_students * 100) if total_students > 0 else 0
            pct_m = (count_m / total_students * 100) if total_students > 0 else 0
            pct_l = (count_l / total_students * 100) if total_students > 0 else 0

            # Map ชื่อและจำนวนคนเข้า Legend
            class_map = {
                "H": f"ระดับสูง (H): {count_h:,} คน ({pct_h:.1f}%)",
                "M": f"ระดับปานกลาง (M): {count_m:,} คน ({pct_m:.1f}%)",
                "L": f"ระดับต่ำ (L): {count_l:,} คน ({pct_l:.1f}%)"
            }
        
            filtered_df["Class_Legend"] = filtered_df["Class"].map(class_map)

            # 2. สร้าง Donut Chart
            fig_donut = px.pie(
                filtered_df, names="Class_Legend", hole=0.55,
                color="Class",
                category_orders={"Class": ["H", "M", "L"]},
                color_discrete_map={"H": "#86EFAC", "M": "#FDE047", "L": "#F87171"}
            )
        
            fig_donut.update_traces(
                textposition="inside", 
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>สัดส่วน: <b>%{percent}</b><extra></extra>"
            )
        
            fig_donut.update_layout(
                height=280, 
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[{
                    "text": f"<b>นักเรียนทั้งหมด</b><br>{total_students:,} คน",
                    "x": 0.5, "y": 0.5, "font_size": 13, "font_color": "#1E293B", "showarrow": False
                }]
            )
        
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False}, key="chart_donut_academic")
        
            # 3. กล่องสรุปข้อสังเกตเชิงลึก (Insight Box)
            st.markdown(f"""
                <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; border-radius: 12px; padding: 12px 16px; margin-top: 10px; margin-bottom: 15px; box-shadow: 0px 1px 3px rgba(0,0,0,0.03);'>
                    <p style='color: #1E293B; font-size: 0.83rem; margin: 0; line-height: 1.5;'>
                        💡 <strong>ข้อสังเกตและการนำไปใช้:</strong><br>
                        • นักเรียนส่วนใหญ่อยู่ใน <strong>ระดับปานกลาง (M) {pct_m:.1f}% ({count_m:,} คน)</strong> ซึ่งเป็นกลุ่มเป้าหมายสำคัญที่มีศักยภาพในการยกระดับผลสัมฤทธิ์ขึ้นสู่ระดับสูง<br>
                        • กลุ่มที่ต้องได้รับการดูแลเร่งด่วนคือ <strong>ระดับต่ำ (L) {pct_l:.1f}% ({count_l:,} คน)</strong> เพื่อวางมาตรการช่วยเหลือและลดอัตราการเรียนตกค้าง
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ไม่พบข้อมูลผลสัมฤทธิ์ทางการเรียนสำหรับแสดงผล")
    # =====================================================
    # โซน: การมีส่วนร่วมรายชั้นปี แยกตามภาคเรียน (Semester Comparison)
    # =====================================================
    with st.container(border=True):
        st.markdown("""
            <h3 style='font-size: 1.1rem; font-weight: bold; color: #0A2540; margin-bottom: 0px;'>การเปรียบเทียบการมีส่วนร่วมรายชั้นปี จำแนกตามภาคเรียน</h3>
            <p style='color: #64748B; font-size: 0.8rem; margin-top: 2px; margin-bottom: 15px;'>
                แสดงคะแนนเฉลี่ยการมีส่วนร่วมในแต่ละชั้นปี เปรียบเทียบระหว่างภาคเรียนที่ 1 และ ภาคเรียนที่ 2
            </p>
        """, unsafe_allow_html=True)

        if {"GradeID", "Semester"}.issubset(filtered_df.columns) and not filtered_df.empty:
            activity_cols = [c for c in ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"] if c in filtered_df.columns]
        
            if activity_cols:
                grade_map = {
                    "G-01": {"name": "ป.2", "order": 1},
                    "G-02": {"name": "ป.2", "order": 2},
                    "G-04": {"name": "ป.4", "order": 3},
                    "G-05": {"name": "ป.5", "order": 4},
                    "G-06": {"name": "ป.6", "order": 5},
                    "G-07": {"name": "ม.1", "order": 6},
                    "G-08": {"name": "ม.2", "order": 7},
                    "G-09": {"name": "ม.3", "order": 8},
                    "G-10": {"name": "ม.4", "order": 9},
                    "G-11": {"name": "ม.5", "order": 10},
                    "G-12": {"name": "ม.6", "order": 11},
                }

                # Groupby ทั้ง GradeID และ Semester
                sem_grade_df = filtered_df.groupby(["GradeID", "Semester"]).agg(
                    Avg_Score=(activity_cols[0], lambda x: filtered_df.loc[x.index, activity_cols].mean(axis=1).mean()),
                    Student_Count=("GradeID", "count")
                ).reset_index()

                sem_grade_df["Grade_Name"] = sem_grade_df["GradeID"].map(lambda x: grade_map.get(x, {}).get("name", x))
                sem_grade_df["Order"] = sem_grade_df["GradeID"].map(lambda x: grade_map.get(x, {}).get("order", 99))
                sem_grade_df["Semester_Label"] = sem_grade_df["Semester"].map({"F": "ภาคเรียนที่ 1", "S": "ภาคเรียนที่ 2"}).fillna(sem_grade_df["Semester"])
            
                sem_grade_df = sem_grade_df.sort_values(by="Order")
                sem_grade_df["Label_Text"] = sem_grade_df.apply(lambda r: f"<b>{r['Avg_Score']:.0f}%</b>", axis=1)

                # สร้าง Grouped Bar Chart
                fig_sem_grade = px.bar(
                    sem_grade_df,
                    x="Grade_Name",
                    y="Avg_Score",
                    color="Semester_Label",
                    barmode="group",
                    text="Label_Text",
                    color_discrete_map={
                        "ภาคเรียนที่ 1": "#93C5FD",  # ฟ้าอ่อน
                        "ภาคเรียนที่ 2": "#1D4ED8"   # น้ำเงินเข้ม
                    }
                )

                fig_sem_grade.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{x} (%{fullData.name})</b><br>คะแนนเฉลี่ย: <b>%{y:.1f}%</b><extra></extra>"
                )

                fig_sem_grade.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title="",
                    yaxis_title="คะแนนเฉลี่ย (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        title_text=""
                    ),
                    xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#1E293B", weight="bold")),
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9', range=[0, 105])
                )

                st.plotly_chart(fig_sem_grade, use_container_width=True, config={"displayModeBar": False}, key="chart_sem_grade")

                # กล่องสรุปข้อสังเกต
                st.markdown("""
                    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #1D4ED8; border-radius: 12px; padding: 12px 16px; margin-top: 10px; margin-bottom: 15px; box-shadow: 0px 1px 3px rgba(0,0,0,0.03);'>
                        <p style='color: #1E293B; font-size: 0.85rem; margin: 0; line-height: 1.5;'>
                            📊 <strong>วิเคราะห์พัฒนาการ:</strong> การเปรียบเทียบระหว่างภาคเรียนช่วยให้เห็นการเติบโตของการมีส่วนร่วมในแต่ละระดับชั้น โดยชั้นปีที่มีคะแนนเพิ่มขึ้นในภาคเรียนที่ 2 สะท้อนถึงการปรับตัวและแรงจูงใจในการเรียนรู้ที่สูงขึ้น
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.warning("ไม่พบคอลัมน์ข้อมูลกิจกรรมการเรียนรู้")
        else:
            st.info("ไม่พบข้อมูลระดับชั้นหรือภาคเรียนสำหรับแสดงผล")

    # --- 3. โซนล่าง: คะแนนเฉลี่ยตามรายวิชา (Topic) ---
    with st.container(border=True):
        st.markdown("""
            <h3 style='font-size: 1.2rem; font-weight: bold; color: #1E293B; margin-bottom: 0px;'>คะแนนเฉลี่ยการมีส่วนร่วมตามรายวิชา</h3>
            <p style='color: #64748B; font-size: 0.85rem; margin-top: 2px; margin-bottom: 15px;'>
                คิดจากกิจกรรม: การยกมือตอบ, การเข้าชมบทเรียน, การดูประกาศ และการร่วมอภิปราย
            </p>
        """, unsafe_allow_html=True)

        if "Topic" in filtered_df.columns and not filtered_df.empty:
            activity_cols = [c for c in ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"] if c in filtered_df.columns]
    
            if activity_cols:
                # พจนานุกรมแปลชื่อรายวิชาเป็นภาษาไทย
                topic_th_map = {
                    "IT": "เทคโนโลยีสารสนเทศ (IT)",
                    "Math": "คณิตศาสตร์",
                    "Arabic": "ภาษาอาหรับ",
                    "English": "ภาษาอังกฤษ",
                    "French": "ภาษาฝรั่งเศส",
                    "Spanish": "ภาษาสเปน",
                    "Science": "วิทยาศาสตร์",
                    "Biology": "ชีววิทยา",
                    "Chemistry": "เคมี",
                    "Geology": "ธรณีวิทยา",
                    "History": "ประวัติศาสตร์",
                    "Quran": "อัลกุรอาน"
                }

                # คำนวณคะแนนเฉลี่ยและนับจำนวนผู้เรียน
                topic_df = filtered_df.groupby("Topic").agg(
                    Avg_Score=(activity_cols[0], lambda x: filtered_df.loc[x.index, activity_cols].mean(axis=1).mean()),
                    Student_Count=("Topic", "count")
                ).reset_index()

                # 1. [แก้ไข] แปลงชื่อวิชาเป็นภาษาไทย
                topic_df["Topic_TH"] = topic_df["Topic"].map(topic_th_map).fillna(topic_df["Topic"])

                # 2. [แก้ไข] เรียงลำดับ ascending=True เพื่อให้ Plotly เอาค่ามากไว้ด้านบน
                topic_df = topic_df.sort_values(by="Avg_Score", ascending=False)
        
                # สร้างข้อความสำหรับ Data Label (เช่น "62% (24 คน)")
                topic_df["Label_Text"] = topic_df.apply(lambda r: f" <b>{r['Avg_Score']:.0f}%</b> ({r['Student_Count']} คน)", axis=1)

                # 3. สร้าง Horizontal Bar Chart
                fig_topic = px.bar(
                    topic_df,
                    x="Avg_Score",
                    y="Topic_TH",  # [แก้ไขจุดที่ผิด] เปลี่ยนจาก "Topic" เป็น "Topic_TH"
                    orientation="h",
                    text="Label_Text",
                    color="Avg_Score",
                    color_continuous_scale=["#93C5FD", "#3B82F6", "#1D4ED8"]
                )

                fig_topic.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    width=0.6,
                    hovertemplate="<b>วิชา %{y}</b><br>คะแนนเฉลี่ย: <b>%{x:.1f}%</b><extra></extra>"
                )

                fig_topic.update_layout(
                    height=480,
                    margin=dict(l=10, r=90, t=10, b=10),
                    xaxis_title="",
                    yaxis_title="",
                    coloraxis_showscale=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#F1F5F9',
                        showticklabels=False,
                        range=[0, topic_df['Avg_Score'].max() * 1.25]
                    ),
                    yaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=13, color="#1E293B", weight="bold")
                    )
                )

                # 4. [แก้ไข] ใส่ key="chart_bar_topic" เพื่อป้องกันเออเรอร์ Duplicate ID
                st.plotly_chart(fig_topic, use_container_width=True, config={"displayModeBar": False}, key="chart_bar_topic")
                # [ปรับแก้ไข] เพิ่ม margin-top: 10px และ margin-bottom: 15px เว้นระยะห่างไม่ให้ติดกรอบล่าง
                st.markdown("""
                    <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #8B5CF6; border-radius: 12px; padding: 12px 16px; margin-top: 10px; margin-bottom: 15px; box-shadow: 0px 1px 3px rgba(0,0,0,0.03);'>
                        <p style='color: #1E293B; font-size: 0.85rem; margin: 0; line-height: 1.5;'>
                            🎯 <strong>ข้อสังเกต:</strong> รายวิชาที่มีระดับการมีส่วนร่วมสูงสะท้อนถึงความสนใจของนักเรียนผ่านกิจกรรมการเรียนรู้แบบโต้ตอบ 
                            ในขณะที่วิชาที่มีคะแนนเฉลี่ยต่ำกว่าอาจเป็นจุดที่ต้องได้รับการสนับสนุนหรือปรับรูปแบบกิจกรรมเพิ่มเติม
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("ไม่พบคอลัมน์ข้อมูลกิจกรรมการเรียนรู้")
        else:
            st.info("ไม่พบข้อมูลรายวิชาสำหรับแสดงผล")

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

    # ทำสำเนาข้อมูลเพื่อป้องกันการแก้ไข filtered_df เดิม
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

    behavior_names = {
        "raisedhands": "Raised Hands",
        "VisITedResources": "Visited Resources",
        "AnnouncementsView": "Announcements View",
        "Discussion": "Discussion"
    }

    # =================================================
    # CSS
    # =================================================
    st.markdown("""
    <style>

    .behavior-card {
        border: 1px solid #8BA4BE;
        border-radius: 25px;
        padding: 10px 6px;
        min-height: 90px;
        background: white;
        text-align: center;
    }

    .behavior-icon {
        font-size: 24px;
        margin-bottom: 2px;
    }

    .behavior-title {
        font-size: 12px;
        color: #2463A5;
        line-height: 1.25;
        min-height: 32px;
    }

    .behavior-value {
        font-size: 15px;
        font-weight: 600;
        color: #333333;
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

    avg_participation = behavior_df["ParticipationScore"].mean()

    # =================================================
    # 1. KPI CARDS
    # =================================================
    card_cols = st.columns(4)

    cards = [
        ("✋", "จำนวนการยกมือ", f"{avg_raised:.1f} ครั้ง/คน"),
        ("👁️", "จำนวนการเข้าดูสื่อ", f"{avg_resource:.1f} ครั้ง/คน"),
        ("📣", "จำนวนการดูประกาศ", f"{avg_announcement:.1f} ครั้ง/คน"),
        ("💬", "จำนวนการมีร่วมอภิปราย", f"{avg_discussion:.1f} ครั้ง/คน")
    ]

    for col, (icon, title, value) in zip(card_cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="behavior-card">
                    <div class="behavior-icon">{icon}</div>
                    <div class="behavior-title">{title}</div>
                    <div class="behavior-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Map ชื่อพฤติกรรมภาษาไทย
    thai_behavior_names = {
        "raisedhands": "การยกมือตอบคำถาม",
        "VisITedResources": "การเข้าดูสื่อการเรียน",
        "AnnouncementsView": "การดูประกาศ",
        "Discussion": "การร่วมอภิปราย"
    }

    # =================================================
    # 2. เปรียบเทียบค่าเฉลี่ยพฤติกรรมการเรียนรู้จำแนกตามมิติต่างๆ
    # =================================================
    with st.container(border=True):
        st.markdown("### 📊 เปรียบเทียบค่าเฉลี่ยพฤติกรรมการเรียนรู้จำแนกตามมิติต่าง ๆ")

        # Dictionary สำหรับแปลงชื่อระดับชั้น (GradeID) เป็นภาษาไทย
        grade_mapping = {
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

        # Order การเรียงลำดับระดับชั้นภาษาไทย
        grade_order = [
            "ป.1", "ป.2", "ป.3",
            "ป.4", "ป.5", "ป.6",
            "ม.1", "ม.2", "ม.3",
            "ม.4", "ม.5", "ม.6"
        ]

        # -------------------------------------------------
        # ส่วนที่ 1: 3 มิติหลัก (ระดับผลการเรียน, ช่วงชั้น, ภาคการศึกษา)
        # -------------------------------------------------
        main_dimensions = {
            "Class": ("ระดับผลการเรียน", {"L": "ระดับต่ำ", "M": "ระดับปานกลาง", "H": "ระดับสูง"}),
            "StageID": ("ช่วงชั้น", {"lowerlevel": "ประถม", "MiddleSchool": "มัธยมต้น", "HighSchool": "มัธยมปลาย"}),
            "Semester": ("ภาคการศึกษา", {"F": "ภาคการศึกษาที่ 1", "S": "ภาคการศึกษาที่ 2"})
        }

        main_list = []
        for dim_col, (dim_label, mapping) in main_dimensions.items():
            if dim_col in behavior_df.columns:
                avg_temp = behavior_df.groupby(dim_col)[behavior_cols].mean().reset_index()
                long_temp = avg_temp.melt(id_vars=dim_col, value_vars=behavior_cols, var_name="Behavior", value_name="Average")
                long_temp["Dimension"] = dim_label
                long_temp["Group"] = long_temp[dim_col].replace(mapping).astype(str)
                long_temp["Behavior"] = long_temp["Behavior"].replace(thai_behavior_names)
                main_list.append(long_temp[["Dimension", "Group", "Behavior", "Average"]])

        if main_list:
            main_df = pd.concat(main_list, ignore_index=True)

            # กำหนดลำดับการแสดงผล
            order_map = {
                "ระดับต่ำ": 1,
                "ระดับปานกลาง": 2,
                "ระดับสูง": 3,
                "ประถม": 1,
                "มัธยมต้น": 2,
                "มัธยมปลาย": 3,
                "ภาคการศึกษาที่ 1": 1,
                "ภาคการศึกษาที่ 2": 2
            }

            main_df["Order"] = main_df["Group"].map(order_map)
            main_df = main_df.sort_values(["Dimension", "Order"])

            fig_main = px.bar(
                main_df,
                x="Group",
                y="Average",
                color="Behavior",
                barmode="group",
                facet_col="Dimension",
                facet_col_spacing=0.06,
                text_auto=".1f",
                title="ค่าเฉลี่ยพฤติกรรมจำแนกตาม ระดับผลการเรียน, ช่วงชั้น และภาคการศึกษา",
                labels={
                    "Group": "",
                    "Average": "ค่าเฉลี่ย (ครั้ง)",
                    "Behavior": "พฤติกรรม",
                    "Dimension": ""
                },
                category_orders={
                    "Dimension": [
                    "ระดับผลการเรียน",
                    "ช่วงชั้น",
                    "ภาคการศึกษา"
                    ],
                    "Group": [
                        "ระดับต่ำ",
                        "ระดับปานกลาง",
                        "ระดับสูง",
                        "ประถม",
                        "มัธยมต้น",
                        "มัธยมปลาย",
                        "ภาคการศึกษาที่ 1",
                        "ภาคการศึกษาที่ 2"
                    ]
                },
                color_discrete_map={
                    "การยกมือตอบคำถาม": "#DDA4F1",
                    "การเข้าดูสื่อการเรียน": "#8EDB8A",
                    "การดูประกาศ": "#F7C363",
                    "การอภิปราย": "#FF8A4C"
                },
                template="plotly_white"
            )

            fig_main.for_each_annotation(
                lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>")
            )

            fig_main.update_xaxes(matches=None, showticklabels=True)
            fig_main.update_yaxes(range=[0, main_df["Average"].max() * 1.18])
            fig_main.update_traces(textposition="outside")

            fig_main.update_layout(
                height=380,
                margin=dict(l=30, r=20, t=60, b=60),
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                legend=dict(
                    orientation="h",
                    y=-0.22,
                    x=0.5,
                    xanchor="center"
                )
            )

            st.plotly_chart(
                fig_main,
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.markdown(
                """
                <div style="background-color:#F8EFFE; border-left:6px solid #DDA4F1;
                    padding:15px 20px; border-radius:10px; margin-top:20px; margin-bottom:20px;
                ">

                <div style="color:#1E40AF; font-size:22; margin-bottom:15px; font-weight:800;">
                🔍 จากข้อมูล</div>
                <p style="font-size:15px; color:#334155; line-height:1.8; margin-bottom:8px;">
                📈 <b>นักเรียนที่มีผลการเรียนระดับสูง</b> มีค่าเฉลี่ยพฤติกรรมการเรียนรู้สูงกว่ากลุ่มอื่นในทุกด้าน 
                โดยเฉพาะการเข้าดูสื่อการเรียนและการยกมือตอบคำถาม</p>

                <p style="font-size:15px; color:#334155; line-height:1.8; margin-bottom:8px;">
                🏫 <b>นักเรียนระดับมัธยมต้น</b> 
                มีการมีส่วนร่วมในการเรียนรู้สูงที่สุดเมื่อเทียบกับระดับประถมศึกษาและมัธยมปลาย</p>

                <p style="font-size:15px; color:#334155; line-height:1.8; margin-bottom:8px;">
                📚 <b>ภาคการศึกษาที่ 2</b> 
                มีค่าเฉลี่ยพฤติกรรมการเรียนรู้สูงกว่าภาคการศึกษาที่ 1 ในเกือบทุกด้าน</p>

                <p style="font-size:15px; color:#334155; line-height:1.8;">
                👁️ <b>การเข้าดูสื่อการเรียน</b> 
                เป็นพฤติกรรมที่มีค่าเฉลี่ยสูงที่สุดในทุกกลุ่ม สะท้อนถึงความสำคัญของทรัพยากรการเรียนรู้ต่อผลสัมฤทธิ์ทางการเรียน</p>

                </div>
                """, 
                    unsafe_allow_html=True
            )

        # -------------------------------------------------
        # ส่วนที่ 2: มิติ "ระดับชั้น (GradeID)" ภาษาไทย แยกออกมากราฟแนวยาว
        # -------------------------------------------------
        if "GradeID" in behavior_df.columns:
            st.markdown("---")
            avg_grade = behavior_df.groupby("GradeID")[behavior_cols].mean().reset_index()
            long_grade = avg_grade.melt(id_vars="GradeID", value_vars=behavior_cols, var_name="Behavior", value_name="Average")
            
            # แปลงชื่อระดับชั้นและพฤติกรรมเป็นภาษาไทย
            long_grade["Grade_Name"] = long_grade["GradeID"].replace(grade_mapping).astype(str)
            long_grade["Behavior"] = long_grade["Behavior"].replace(thai_behavior_names)

            fig_grade = px.bar(
                long_grade,
                x="Grade_Name",
                y="Average",
                color="Behavior",
                barmode="group",
                text_auto=".1f",
                title="ค่าเฉลี่ยพฤติกรรมการเรียนรู้จำแนกรายระดับชั้น",
                labels={"Grade_Name": "ระดับชั้น", "Average": "ค่าเฉลี่ย (ครั้ง)", "Behavior": "พฤติกรรม"},
                category_orders={"Grade_Name": grade_order}, # เรียงลำดับ ป.1 -> ม.6 ให้ถูกต้อง
                color_discrete_map={
                    "การยกมือตอบคำถาม": "#DDA4F1",
                    "การเข้าดูสื่อการเรียน": "#8EDB8A",
                    "การดูประกาศ": "#F7C363",
                    "การอภิปราย": "#FF8A4C"
                },
                template="plotly_white"
            )

            fig_grade.update_yaxes(range=[0, long_grade["Average"].max() * 1.18])
            fig_grade.update_traces(textposition="outside")
            
            fig_grade.update_layout(
                height=450, # ปรับความสูงเล็กน้อยเพื่อรองรับชื่อแกน X ภาษาไทยที่ยาวขึ้น
                margin=dict(l=30, r=20, t=60, b=80),
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                showlegend=False
            )

            st.plotly_chart(fig_grade, use_container_width=True, config={"displayModeBar": False})

        # -------------------------------------------------
        # ส่วนที่ 3: สรุป Insight
        # -------------------------------------------------
        all_data = pd.concat([main_df, long_grade.rename(columns={"Grade_Name": "Group"}).assign(Dimension="ระดับชั้น")], ignore_index=True)
        max_row = all_data.loc[all_data["Average"].idxmax()]
        min_row = all_data.loc[all_data["Average"].idxmin()]
        st.markdown(
            f"""
            <div style="
                background-color:#FFFBEB;
                border-left:6px solid #F7C363;
                padding:15px 20px;
                border-radius:10px;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <div style="color:#166534; font-size:16px;
                    font-weight:700; margin-bottom:10px;
                ">
                    📌 สรุปภาพรวมการกระจายตัว</div>
                <div style="color:#334155; line-height:1.9;">
                * พฤติกรรมที่มีค่าเฉลี่ยสูงสุดในภาพรวม คือ {max_row['Behavior']} ในกลุ่ม {max_row['Group']} 
                ({max_row['Dimension']}) เฉลี่ย {max_row['Average']:.1f} ครั้ง<br>
                * พฤติกรรมที่มีค่าเฉลี่ยต่ำสุดในภาพรวม คือ {min_row['Behavior']} ในกลุ่ม {min_row['Group']}
                ({min_row['Dimension']}) เฉลี่ย {min_row['Average']:.1f} ครั้ง
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
   
    # =================================================
    # 2. GROUPED BAR CHART: การกระจายความถี่ (นำโค้ดใหม่มาวางตรงนี้ได้เลย)
    # =================================================
    with st.container(border=True):
        st.markdown("### 📈 การกระจายความถี่ของพฤติกรรมการเรียนรู้ทั้งหมด")

        behavior_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
        bins = [-1, 10, 20, 30, 45, float("inf")]
        labels = ["0–10 ครั้ง", "11–20 ครั้ง", "21–30 ครั้ง", "31–45 ครั้ง", "มากกว่า 45 ครั้ง"]

        # 1. จัดกลุ่มช่วงคะแนนของทุกพฤติกรรมพร้อมกัน
        dist_list = []
        for col in behavior_cols:
            b_range = pd.cut(behavior_df[col], bins=bins, labels=labels, include_lowest=True)
            counts = b_range.value_counts().sort_index().reset_index()
            counts.columns = ["ช่วงจำนวนครั้ง", "จำนวนผู้เรียน"]
            counts["Behavior"] = thai_behavior_names[col]
            dist_list.append(counts)

        # รวม Dataframe เข้าด้วยกัน
        all_dist_df = pd.concat(dist_list, ignore_index=True)

        # 2. สร้าง Grouped Bar Chart รวมในกราฟเดียว
        fig_all_dist = px.bar(
            all_dist_df,
            x="ช่วงจำนวนครั้ง",
            y="จำนวนผู้เรียน",
            color="Behavior",
            barmode="group",
            text_auto=True,
            title="การเปรียบเทียบการกระจายความถี่จำแนกตามพฤติกรรมการเรียนรู้",
            labels={
                "ช่วงจำนวนครั้ง": "ช่วงความถี่การทำพฤติกรรม",
                "จำนวนผู้เรียน": "จำนวนผู้เรียน (คน)",
                "Behavior": "พฤติกรรมการเรียนรู้"
            },
            color_discrete_map={
                "การยกมือตอบคำถาม": "#DDA4F1",
                "การเข้าดูสื่อการเรียน": "#8EDB8A",
                "การดูประกาศ": "#F7C363",
                "การอภิปราย": "#FF8A4C"
            },
            template="plotly_white"
        )

        fig_all_dist.update_traces(textposition="outside")
        fig_all_dist.update_layout(
            height=450,
            margin=dict(l=40, r=30, t=60, b=80),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center"
            )
        )

        st.plotly_chart(fig_all_dist, use_container_width=True, config={"displayModeBar": False})

        # คำนวณ Insight คำอธิบายสรุปภาพรวม
        total_students = len(behavior_df)
        
        high_freq_df = all_dist_df[all_dist_df["ช่วงจำนวนครั้ง"] == "มากกว่า 45 ครั้ง"]
        max_high = high_freq_df.loc[high_freq_df["จำนวนผู้เรียน"].idxmax()]

        low_freq_df = all_dist_df[all_dist_df["ช่วงจำนวนครั้ง"] == "0–10 ครั้ง"]
        max_low = low_freq_df.loc[low_freq_df["จำนวนผู้เรียน"].idxmax()]
        st.markdown(
            f"""
            <div style="
                background-color:#F0FDF4;
                border-left:6px solid #16A34A;
                padding:15px 20px;
                border-radius:10px;
                margin-top:20px;
                margin-bottom:20px;
            ">
                <div style="color:#166534; font-size:16px;
                    font-weight:700; margin-bottom:10px;
                ">
                    📌 สรุปภาพรวมการกระจายตัว</div>
                <div style="color:#334155; line-height:1.9;">
                *พฤติกรรมที่มีผู้เรียนทำในระดับสูงมาก (มากกว่า 45 ครั้ง) มากที่สุด คือ {max_high['Behavior']}
                จำนวน {max_high['จำนวนผู้เรียน']} คน
                (คิดเป็น {(max_high['จำนวนผู้เรียน']/total_students)*100:.1f}%)<br>
                * พฤติกรรมที่มีผู้เรียนทำในระดับต่ำ (0–10 ครั้ง) มากที่สุด คือ {max_low['Behavior']}
                จำนวน {max_low['จำนวนผู้เรียน']} คน 
                (คิดเป็น {(max_low['จำนวนผู้เรียน']/total_students)*100:.1f}%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

#---------------------------------------------------------------------------------------------------------
# การมาเรียน
#---------------------------------------------------------------------------------------------------------        
elif menu == "การเข้าเรียน":
    
    st.title("🗓️ Attendance Analysis")
    st.caption("แสดงสัดส่วนการขาดเรียนจำแนกตาม ผลการเรียน, ช่วงชั้น, ชั้นเรียน และภาคการศึกษา")
    st.markdown("---")
    
    # Mapping Dictionary สำหรับแปลงชื่อกลุ่มเป็นภาษาไทย
    absence_map = {"Under-7": "ขาดเรียน < 7 วัน", "Above-7": "ขาดเรียน ≥ 7 วัน"}
    class_map = {"L": "ระดับต่ำ (L)", "M": "ระดับปานกลาง (M)", "H": "ระดับสูง (H)"}
    stage_map = {"lowerlevel": "ประถมศึกษา", "MiddleSchool": "มัธยมศึกษาตอนต้น", "HighSchool": "มัธยมศึกษาตอนปลาย"}
    semester_map = {"F": "ภาคเรียนที่ 1", "S": "ภาคเรียนที่ 2"}
    grade_map = {
        "G-01": "ป.1", "G-02": "ป.2", "G-03": "ป.3", "G-04": "ป.4", "G-05": "ป.5", "G-06": "ป.6",
        "G-07": "ม.1", "G-08": "ม.2", "G-09": "ม.3", "G-10": "ม.4", "G-11": "ม.5", "G-12": "ม.6"
    }

    with st.container(border=True):
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 16px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 20px; font-weight: 700;'>
                    🚨 ภาพรวมการขาดเรียน (Student Absence Days)
                </h3>
                <p style='color: #64748B; font-size: 14px; margin: 2px 0 0 0;'>
                    เปรียบเทียบสัดส่วนนักเรียนขาดเรียนน้อย (< 7 วัน) และขาดเรียนมาก (≥ 7 วัน)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # 1. การคำนวณสถิติภาพรวม (KPI)
        absence_counts = filtered_df["StudentAbsenceDays"].value_counts().reset_index()
        absence_counts.columns = ["AbsenceCategory", "Count"]
        absence_counts["Absence_TH"] = absence_counts["AbsenceCategory"].map(absence_map)
        
        total_filtered = len(filtered_df)
        absence_counts["Percent"] = (absence_counts["Count"] / total_filtered * 100) if total_filtered > 0 else 0
    
        under_7_row = absence_counts[absence_counts["AbsenceCategory"] == "Under-7"]
        above_7_row = absence_counts[absence_counts["AbsenceCategory"] == "Above-7"]
    
        count_under7 = under_7_row["Count"].values[0] if not under_7_row.empty else 0
        pct_under7 = under_7_row["Percent"].values[0] if not under_7_row.empty else 0
        count_above7 = above_7_row["Count"].values[0] if not above_7_row.empty else 0
        pct_above7 = above_7_row["Percent"].values[0] if not above_7_row.empty else 0
    
        # KPI Cards
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.markdown(
                f"""
                <div style='background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 12px; text-align: center;'>
                    <span style='color: #166534; font-size: 13px; font-weight: 600;'>🟢 ขาดเรียน < 7 วัน</span>
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
                    <span style='color: #991B1B; font-size: 13px; font-weight: 600;'>🔴 ขาดเรียน ≥ 7 วัน</span>
                    <div style='color: #DC2626; font-size: 22px; font-weight: 800; margin-top: 2px;'>
                        {count_above7:,} คน <span style='font-size: 14px; font-weight: 500;'>({pct_above7:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # 2. กราฟจำแนกตามมิติต่างๆ (Descriptive Bar Charts)
        # -------------------------------------------------
        
        # ฟังก์ชันช่วยสร้าง Bar Chart แบบ Grouped
        def create_absence_bar(df, group_col, title_text, label_map=None, category_order=None):
            temp_df = df.groupby([group_col, "StudentAbsenceDays"]).size().reset_index(name="Count")
            temp_df["Absence_TH"] = temp_df["StudentAbsenceDays"].map(absence_map)
            if label_map:
                temp_df["Group_TH"] = temp_df[group_col].map(label_map).fillna(temp_df[group_col])
            else:
                temp_df["Group_TH"] = temp_df[group_col]

            fig = px.bar(
                temp_df,
                x="Group_TH",
                y="Count",
                color="Absence_TH",
                barmode="group",
                text="Count",
                title=f"<b>{title_text}</b>",
                labels={"Group_TH": "", "Count": "จำนวน (คน)", "Absence_TH": "การขาดเรียน"},
                color_discrete_map={"ขาดเรียน < 7 วัน": "#22C55E", "ขาดเรียน ≥ 7 วัน": "#EF4444"},
                category_orders={"Group_TH": category_order} if category_order else {}
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0, temp_df["Count"].max() * 1.25])
            fig.update_layout(
                template="plotly_white",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
            )
            return fig

        # แถวที่ 1: ผลการเรียน (Class) & ภาคการศึกษา (Semester)
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if "Class" in filtered_df.columns:
                fig_class = create_absence_bar(filtered_df, "Class", "จำแนกตามระดับผลการเรียน", class_map, ["ระดับต่ำ (L)", "ระดับปานกลาง (M)", "ระดับสูง (H)"])
                st.plotly_chart(fig_class, use_container_width=True, config={"displayModeBar": False})

        with row1_col2:
            if "Semester" in filtered_df.columns:
                fig_sem = create_absence_bar(filtered_df, "Semester", "จำแนกตามภาคการศึกษา", semester_map, ["ภาคเรียนที่ 1", "ภาคเรียนที่ 2"])
                st.plotly_chart(fig_sem, use_container_width=True, config={"displayModeBar": False})

        # แถวที่ 2: ช่วงชั้น (StageID) & ระดับชั้นเรียน (GradeID)
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if "StageID" in filtered_df.columns:
                fig_stage = create_absence_bar(filtered_df, "StageID", "จำแนกตามช่วงชั้น", stage_map, ["ประถมศึกษา", "มัธยมศึกษาตอนต้น", "มัธยมศึกษาตอนปลาย"])
                st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})

        with row2_col2:
            if "GradeID" in filtered_df.columns:
                grade_order_list = ["ป.1", "ป.2", "ป.3", "ป.4", "ป.5", "ป.6", "ม.1", "ม.2", "ม.3", "ม.4", "ม.5", "ม.6"]
                fig_grade = create_absence_bar(filtered_df, "GradeID", "จำแนกตามชั้นเรียน (Grade)", grade_map, grade_order_list)
                st.plotly_chart(fig_grade, use_container_width=True, config={"displayModeBar": False})

        # -------------------------------------------------
        # 3. สรุปสถิติเชิงพรรณนา (Descriptive Summary)
        # -------------------------------------------------
        st.markdown(
            f"""
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #0EA5E9; border-radius: 8px; 
                        padding: 12px; font-size: 13px; color: #334155; padding: 12px 16px; margin-top: 10px; margin-bottom: 15px; '>
                📌 <b>สรุปภาพรวม:</b> จากนักเรียนทั้งหมด <b>{total_filtered:,} คน</b> 
                มีผู้เรียนที่อยู่ในกลุ่มขาดเรียนน้อยกว่า 7 วัน จำนวน <b>{count_under7:,} คน ({pct_under7:.1f}%)</b> 
                และกลุ่มขาดเรียนตั้งแต่ 7 วันขึ้นไป จำนวน <b>{count_above7:,} คน ({pct_above7:.1f}%)</b> 
                โดยสามารถแยกดูสัดส่วนตามมิติต่างๆ ในกราฟด้านบนเพื่อเปรียบเทียบจำนวนนักเรียนในแต่ละกลุ่มได้ทันที
            </div>
            """,
            unsafe_allow_html=True
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
                color_discrete_map={"บิดา": "#3B82F6", "มารดา": "#8B5CF6"} # สีน้ำเงินฟ้า และ สีม่วงอ่อน ตามรูปตัวอย่าง
            )
            
            fig_donut.update_traces(
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
                    <div style='background-color: #EFF6FF; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid #3B82F6;'>
                        <span style='color: #1E3A8A; font-weight: bold; font-size: 14px;'>🟦 บิดา (ผู้ดูแลหลัก)</span><br>
                        <b style='color: #2563EB; font-size: 22px;'>{father_cnt:,} คน</b><br>
                        <span style='color: #64748B; font-size: 12px;'>คิดเป็น {father_pct:.1f}%</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # บล็อกที่ 2: มารดา
                st.markdown(
                    f"""
                    <div style='background-color: #F5F3FF; padding: 12px 16px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #8B5CF6;'>
                        <span style='color: #4C1D95; font-weight: bold; font-size: 14px;'>🟪 มารดา (ผู้ดูแลหลัก)</span><br>
                        <b style='color: #7C3AED; font-size: 22px;'>{mother_cnt:,} คน</b><br>
                        <span style='color: #64748B; font-size: 12px;'>คิดเป็น {mother_pct:.1f}%</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # บล็อกข้อสังเกตเชิงสถิติ ด้านล่างสุด
                st.markdown(
                    f"""
                    <div style='background-color: #FEF3C7; padding: 12px 14px; border-radius: 8px; 
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
                color_discrete_map={"บิดา": "#3B82F6", "มารดา": "#8B5CF6"}, # คุมโทนสีตามแถวแรก
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
                    <div style='background-color: #FEF3C7; padding: 10px 12px; border-radius: 8px; 
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
                color_discrete_map={"บิดา": "#3B82F6", "มารดา": "#8B5CF6"}
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
                <div style='background-color: #FEF3C7; padding: 10px 12px; border-radius: 8px; 
                border: 1px solid #FCD34D; margin-top: 0px; margin-bottom: 10px'>
                    <span style='color: #92400E; font-size: 12px; line-height: 1.4; display: block;'>
                        💡 <b>ข้อสังเกต:</b> นักเรียนที่มี <b>มารดา</b> ดูแลหลัก มีอัตราการเข้าดูสื่อการเรียน (69.1 ครั้ง) และยกมือตอบคำถามสูงกว่ามีบิดาดูแลหลักอย่างมีนัยสำคัญ
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # 👉 กราฟฝั่งขวา: แก้ไขอักษรซ้อนทับ
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
                title="<b>🎓 ความสัมพันธ์กับผลการเรียน</b>",
                color_discrete_map={"ตอบแบบสำรวจ": "#10B981", "ไม่ตอบแบบสำรวจ": "#F59E0B"}
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
                <div style='background-color: #EFF6FF; padding: 10px 12px; border-radius: 8px; 
                border: 1px solid #93C5FD; margin-top: 0px; margin-bottom: 10px;'>
                    <span style='color: #1E3A8A; font-size: 12px; line-height: 1.4; display: block;'>
                        🎯 <b>ข้อสังเกต:</b> กลุ่ม <b>ระดับสูง (High)</b> ผู้ปกครองตอบแบบสำรวจสูงถึง 80.3% ในขณะที่กลุ่ม <b>ระดับต่ำ (Low)</b> ผู้ปกครองไม่ตอบแบบสำรวจสูงถึง 78.0%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

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