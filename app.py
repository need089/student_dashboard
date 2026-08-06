import streamlit as st
import pandas as pd
import plotly.express as px

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

import streamlit as st

# ------------------------
# SIDEBAR NAVIGATION & FILTERS
# ------------------------
with st.sidebar:
    # 🎨 Custom CSS ระดับพรีเมียมสำหรับ Sidebar และ ตัวกรอง
    st.markdown("""
        <style>
            /* 1. ปรับพื้นหลัง Sidebar หลัก */
            [data-testid="stSidebar"] {
                background-color: #F8FAFC;
                border-right: 1px solid #E2E8F0;
            }
            
            /* 🔒 ล็อคไม่ให้พิมพ์ใน Selectbox (คลิกเลือกได้อย่างเดียว) */
            div[data-baseweb="select"] input {
                caret-color: transparent !important;
                pointer-events: none !important;
            }

            /* 2. หัวข้อแดชบอร์ดหลัก */
            .sidebar-title {
                text-align: center;
                margin-top: -10px;
                font-size: 1.2rem;
                font-weight: 700;
                color: #0F172A;
                letter-spacing: -0.3px;
            }

            /* 3. หัวข้อ Section */
            .filter-header {
                color: #334155;
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            /* 4. กล่องการ์ดสำหรับรวมตัวกรองทั้งหมด (Filter Section Wrapper) */
            .filter-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            }

            /* 5. ปรับแต่ง Label ของ Multiselect ใน Streamlit */
            div[data-testid="stMultiSelect"] label p {
                font-weight: 600 !important;
                color: #475569 !important;
                font-size: 0.85rem !important;
            }

            /* 6. แต่งสไตล์ แท็กตัวเลือก (Multiselect Selected Tags) ให้ดูโปร */
            span[data-baseweb="tag"] {
                background-color: #EFF6FF !important; /* ฟ้าอ่อนนุ่มนวล */
                border: 1px solid #BFDBFE !important;  /* เส้นขอบฟ้า */
                border-radius: 6px !important;
                padding: 2px 6px !important;
            }
            span[data-baseweb="tag"] span {
                color: #1E40AF !important; /* ตัวอักษรน้ำเงินเข้ม */
                font-weight: 600 !important;
                font-size: 0.8rem !important;
            }

            /* 7. กล่องสรุปจำนวนข้อมูลด้านล่างสุด (Badge) */
            .data-info-box {
                background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
                border: 1px solid #BFDBFE;
                border-left: 4px solid #2563EB;
                padding: 12px;
                border-radius: 10px;
                font-size: 0.85rem;
                color: #334155;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
            }
        </style>
    """, unsafe_allow_html=True)

    # --- ส่วน Logo ---
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=75)
    
    st.markdown(
        "<h2 class='sidebar-title'>การวิเคราะห์ผลการเรียนรู้</h2>", 
        unsafe_allow_html=True
    )
    st.markdown("<hr style='margin: 15px 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

    # --- เมนูนำทาง ---
    st.markdown("<div class='filter-header'>🧭 เมนูนำทาง</div>", unsafe_allow_html=True)
    menu_options = [
        "ภาพรวม",
        "การเข้าเรียน",
        "พฤติกรรมการเรียนรู้",
        "ผลสัมฤทธิ์ทางการเรียน",
        "การมีส่วนร่วมของผู้ปกครอง",
        "การวิเคราะห์ความสัมพันธ์ของข้อมูล",
        "การทำนายผลการเรียนรู้"
    ]

    menu = st.selectbox(
        "เลือกหน้าต่างแดชบอร์ด",
        options=menu_options,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- ตัวกรองข้อมูล (รวมอยู่ใน Card มินิมอล) ---
    st.markdown("<div class='filter-header'>🔍 ตัวกรองข้อมูล</div>", unsafe_allow_html=True)
    
    # 💡 ใช้ container หรือจัดวางให้อยู่ในกลุ่มที่ดูสะอาดตา
    with st.container():
        gender = st.multiselect(
            "👤 เพศ",
            options=df["gender"].unique(),
            default=df["gender"].unique(),
            placeholder="เลือกเพศ..."
        )

        stage = st.multiselect(
            "🏫 ระดับการศึกษา",
            options=df["StageID"].unique(),
            default=df["StageID"].unique(),
            placeholder="เลือกระดับชั้น..."
        )

        semester = st.multiselect(
            "📅 ภาคเรียน",
            options=df["Semester"].unique(),
            default=df["Semester"].unique(),
            placeholder="เลือกเทอม..."
        )

    # Filter Data
    filtered_df = df[
        (df["gender"].isin(gender))
        & (df["StageID"].isin(stage))
        & (df["Semester"].isin(semester))
    ]

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # --- กล่องแสดงจำนวนข้อมูล (Status Badge) ---
    st.markdown(
        f"""
        <div class='data-info-box'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span>📊 <b>แสดงผลข้อมูล</b></span>
                <span style='background-color:#DBEAFE; color:#1E40AF; padding:2px 8px; border-radius:12px; font-weight:bold; font-size:0.8rem;'>
                    {len(filtered_df):,} / {len(df):,}
                </span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
# =====================================================
# OVERVIEW
# =====================================================
if menu == "ภาพรวม":

    st.title("📌 ภาพรวมของข้อมูล")
    st.caption("ภาพรวมของขผลสัมฤทธิ์ทางการเรียนของผู้เรียน")
    st.markdown("---")

    # ROW 1: GENDER DISTRIBUTION
    chart1, chart2 = st.columns(2)
    
    with chart1:
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 24px; font-weight: 700;'>👤 สัดส่วนของผู้เรียน</h3>
                <p style='color: #64748B; font-size: 15px; margin: 2px 0 0 0;'>ผู้เรียนจำแนกตามเพศ</p>
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
            textfont=dict(size=15, color="white", family="Arial"),
            hovertemplate="<b>%{label}</b><br>จำนวน : %{value:,}<br>สัดส่วน : %{percent}<extra></extra>",
            marker=dict(line=dict(color="white", width=3))
        )

        fig_gender.update_layout(
            annotations=[
                dict(
                    x=0.5, y=0.5,
                    text=f"<span style='color:#64748B; font-size:12px;'>จำนวนนักเรียนทั้งหมด</span><br><br>"
                         f"<b style='font-size:24px; color:#1E293B;'>{total_gender_students:,} คน</b>",
                    showarrow=False
                )
            ],
            showlegend=True,
            legend=dict(
                orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=14)
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=40),
            height=420
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
                        padding: 8px; border-radius: 8px; font-weight: bold; margin-bottom: 12px;'>
                สรุปภาพรวม
            </div>
            
            <div style='background-color: #EFF6FF; border-radius: 12px; padding: 15px; margin-bottom: 12px;'>
                <div style='display: flex; align-items: center;'>
                    <div style='background-color: #4F8EF7; color: white; width: 45px; height: 45px;
                                border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                font-size: 22px; margin-right: 12px;'>👨</div>
                    <div>
                        <div style='color: #1E293B; font-weight: bold; font-size: 15px;'>เพศชาย</div>
                        <div style='color: #4F8EF7; font-size: 24px; font-weight: bold;'>{males:,}
                          <span style='font-size: 14px; font-weight: normal; color: #475569;'>คน</span></div>
                    </div>
                </div>
                <hr style='margin: 8px 0; border-top: 1px solid #DBEAFE;'>
                <div style='color: #4F8EF7; font-weight: bold; font-size: 14px;'>{male_pct:.1f}%</div>
            </div>

            <div style='background-color: #FFF1F2; border-radius: 12px; padding: 15px; margin-bottom: 12px;'>
                <div style='display: flex; align-items: center;'>
                    <div style='background-color: #FF6B9A; color: white; width: 45px; 
                                height: 45px; border-radius: 50%; display: flex; 
                                align-items: center; justify-content: center; 
                                font-size: 22px; margin-right: 12px;'>👩</div>
                    <div>
                        <div style='color: #1E293B; font-weight: bold; font-size: 15px;'>เพศหญิง</div>
                        <div style='color: #FF6B9A; font-size: 24px; font-weight: bold;'>{females:,} 
                            <span style='font-size: 14px; font-weight: normal; color: #475569;'>คน</span></div>
                    </div>
                </div>
                <hr style='margin: 8px 0; border-top: 1px solid #FFE4E6;'>
                <div style='color: #FF6B9A; font-weight: bold; font-size: 14px;'>{female_pct:.1f}%</div>
            </div>

            <div style='background-color: #DCFCE7; border: 1px solid #BBF7D0; border-radius: 12px; padding: 15px;'>
                <div style='display: flex; align-items: center;'>
                    <div style='background-color: #064E3B; width: 45px; height: 45px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; 
                                margin-right: 12px; box-shadow: 0 2px 6px rgba(6, 78, 59, 0.3);'>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M16 21V19C16 17.9391 15.5786 16.9217 14.8284 16.1716C14.0783 15.4214 13.0609 15 12 15C10.9391 15 9.92172 15.4214 9.17157 16.1716C8.42143 16.9217 8 17.9391 8 19V21" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M12 11C14.2091 11 16 9.20914 16 7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7C8 9.20914 9.79086 11 12 11Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div>
                        <div style='color: #1E293B; font-weight: bold; font-size: 15px;'>รวมทั้งหมด</div>
                        <div style='color: #059669; font-size: 24px; font-weight: bold;'>{total_curr_gender:,} 
                            <span style='font-size: 14px; font-weight: normal; color: #475569;'>คน</span></div>
                    </div>
                </div>
                <hr style='margin: 8px 0; border-top: 1px solid #E2E8F0;'>
                <div style='color: #059669; font-weight: bold; font-size: 14px;'>100.0%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style='background-color: #F8FAFC; color: #64748B; padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-top: 10px;'>
                ℹ️ ข้อมูลนี้แสดงสัดส่วนของนักเรียนจำแนกตามเพศ จากข้อมูลทั้งหมด {total_curr_gender:,} คน
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 2: SEMESTER DISTRIBUTION
    chart3, chart4 = st.columns(2)

    with chart3:
        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 24px; font-weight: 700;'>📅 จำนวนนักเรียน</h3>
                <p style='color: #64748B; font-size: 15px; margin: 2px 0 0 0;'>จำนวนนักเรียนแยกตามภาคเรียน</p>
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
            textfont=dict(size=18, color="white"),
            marker=dict(line=dict(color="white", width=4)),
            hovertemplate="<b>%{label}</b><br>Students : %{value:,} คน<br>Percentage : %{percent}<extra></extra>"
        )

        fig_semester.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h", y=-0.05, x=0.5, xanchor="center", font=dict(size=14)
            ),
            annotations=[
                dict(
                    x=0.5, y=0.5,
                    text=f"<span style='color:#64748B; font-size:12px;'>จำนวนนักเรียนทั้งหมด</span><br><br>"
                         f"<b style='font-size:24px; color:#1E293B;'>{total_semester_students:,} คน</b>",
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
            <div style='display: flex; flex-direction: column; justify-content: center; height: 100%;'>
                <div style='background-color: #EFF6FF; border-radius: 12px; padding: 15px; margin-bottom: 12px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: #3B82F6; color: white; width: 45px; height: 45px; 
                                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                    font-size: 22px; margin-right: 12px;'>📘</div>
                        <div>
                            <div style='color: #1E293B; font-weight: bold; font-size: 15px;'>ภาคเรียนที่ 1</div>
                            <div style='color: #2563EB; font-size: 24px; font-weight: bold;'>{semester1:,} <span style='font-size: 14px; 
                                        font-weight: normal; color: #475569;'>คน</span></div>
                        </div>
                    </div>
                    <hr style='margin: 8px 0; border: none; border-top: 1px solid #DBEAFE;'>
                    <div style='color: #2563EB; font-weight: bold; font-size: 14px;'>{percent1:.1f}%</div>
                </div>
                <div style='background-color: #F5F3FF; border-radius: 12px; padding: 15px; margin-bottom: 12px;'>
                    <div style='display: flex; align-items: center;'>
                        <div style='background-color: #8B5CF6; color: white; width: 45px; height: 45px; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; font-size: 22px; margin-right: 12px;'>📙</div>
                        <div>
                            <div style='color: #1E293B; font-weight: bold; font-size: 15px;'>ภาคเรียนที่ 2</div>
                            <div style='color: #7C3AED; font-size: 24px; font-weight: bold;'>{semester2:,} <span style='font-size: 14px; 
                                        font-weight: normal; color: #475569;'>คน</span></div>
                        </div>
                    </div>
                    <hr style='margin: 8px 0; border: none; border-top: 1px solid #DDD6FE;'>
                    <div style='color: #7C3AED; font-weight: bold; font-size: 14px;'>{percent2:.1f}%</div>
                </div>
                <div style='background-color: #F8FAFC; border-left: 4px solid #3B82F6; border-radius: 10px; 
                            padding: 12px 15px; font-size: 13px; color: #334155; box-shadow: 0 1px 4px rgba(0,0,0,.05);'> 
                    <b>💡 ข้อสังเกต (Insight):</b> นักเรียนส่วนใหญ่ลงทะเบียนใน <b>{major_semester}</b> คิดเป็น <b>{major_percent:.1f}%</b> ของทั้งหมด
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 3: STAGE DISTRIBUTION
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 10px;'>
            <h3 style='color: #1E293B; margin: 0; font-size: 24px; font-weight: 700;'>🏫 นักเรียนตามระดับชั้น</h3>
            <p style='color: #64748B; font-size: 15px; margin: 2px 0 0 0;'>แสดงจำนวนผู้เรียนในแต่ละชั้น</p>
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

    fig_stage.update_traces(textposition="outside", marker_line_width=0)

    fig_stage.update_layout(
        height=400,
        xaxis_title="ระดับการศึกษา",
        yaxis_title="จำนวนนักเรียน",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#E2E8F0", zeroline=False),
        xaxis=dict(linecolor="#CBD5E1"),
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})

    if not stage_count.empty:
        most_stage = stage_count.iloc[0]["Stage"]
        most_count = stage_count.iloc[0]["Count"]

        st.markdown(
            f"""
            <div style='background-color:#F8FAFC; border:1px solid #BFDBFE; border-left:4px solid #3B82F6;
                        padding:12px 14px; border-radius:10px; font-size:13px; color:#334155; box-shadow:0 2px 6px rgba(0,0,0,.05);'>
                <b>💡 ข้อสังเกต (Insight):</b> ระดับการศึกษาที่มีนักเรียนมากที่สุดคือ <b>{most_stage}</b> จำนวน <b>{most_count:,}</b> คนจากข้อมูลทั้งหมด
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ROW 4: ACADEMIC PERFORMANCE
    class_order = ["H", "M", "L"]
    class_df = (
        filtered_df["Class"]
        .value_counts()
        .reindex(class_order, fill_value=0)
        .reset_index()
    )
    class_df.columns = ["Class", "Count"]

    total_students_perf = class_df["Count"].sum()

    class_df["Percent"] = (
        (class_df["Count"] / total_students_perf * 100)
        if total_students_perf > 0
        else 0
    )

    labels_map = {
        "H": "H<br>(ผลการเรียนสูง)",
        "M": "M<br>(ผลการเรียนปานกลาง)",
        "L": "L<br>(ผลการเรียนต่ำ)",
    }
    class_df["Label"] = class_df["Class"].map(labels_map)

    st.markdown(
        """
        <div style='text-align:left; margin-bottom: 10px;'>
            <h3 style='color:#1E293B; margin:0; font-size:24px; font-weight:700;'>
                📊 ผลสัมฤทธิ์ทางการเรียนของผู้เรียน
            </h3>
            <div style='color:#64748B; font-size:15px; margin-top:4px;'>
                จำนวนและร้อยละของนักเรียนในแต่ละระดับผลการเรียน
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    fig_perf = px.bar(
        class_df,
        x="Label",
        y="Count",
        text="Count",
        color="Class",
        color_discrete_map={
            "H": "#68C0CF",
            "M": "#6F8FB6",
            "L": "#F29ED4"
        }
    )

    fig_perf.update_traces(
        texttemplate="<b>%{y:,}</b>",
        textposition="outside",
        textfont=dict(size=14, color="#1E293B"),
        marker_line_width=0
    )

    for idx, row in class_df.iterrows():
        if row["Count"] > 0:
            fig_perf.add_annotation(
                x=row["Label"],
                y=row["Count"] / 2,
                text=f"<b>{row['Percent']:.1f}%</b>",
                showarrow=False,
                font=dict(size=13, color="#1E293B")
            )

    max_val = class_df["Count"].max()

    fig_perf.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Arial", size=13, color="#334155")
    )

    fig_perf.update_xaxes(title="", showgrid=False)
    fig_perf.update_yaxes(
        title="จำนวนนักเรียน (คน)",
        range=[0, max_val * 1.25 if max_val > 0 else 100],
        gridcolor="#E2E8F0"
    )

    st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})

    if total_students_perf > 0:
        top_row = class_df.sort_values(by="Count", ascending=False).iloc[0]
        top_class = top_row["Class"]
        top_count = top_row["Count"]
        top_percent = top_row["Percent"]

        class_name = {
            "H": "ผลการเรียนสูง",
            "M": "ผลการเรียนปานกลาง",
            "L": "ผลการเรียนต่ำ"
        }

        st.markdown(
            f"""
            <div style='background-color:#F8FAFC; border:1px solid #BFDBFE; border-left:4px solid #3B82F6;
                        border-radius:10px; padding:12px 15px; font-size:13px; color:#334155; box-shadow:0 2px 6px rgba(0,0,0,.05);'>
                <b>💡 ข้อสังเกต (Insight):</b> นักเรียนส่วนใหญ่อยู่ในกลุ่ม <b>{class_name[top_class]}</b> จำนวน <b>{top_count:,}</b> คน คิดเป็น <b>{top_percent:.1f}%</b> ของทั้งหมด
            </div>
            """,
            unsafe_allow_html=True
        )
    # -------------------------------------------------------------------------
    # TOP NATIONALITY 
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Header ส่วนหัวข้อ ---
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 15px;'>
            <h3 style='color: #1E293B; margin: 0; font-size: 24px; font-weight: 700; display: flex; align-items: center; gap: 8px;'>
                🌍 สัญชาติของผู้เรียนทั้งหมด
            </h3>
            <p style='color: #64748B; font-size: 14px; margin: 4px 0 0 0;'>
                วิเคราะห์สัดส่วนการกระจายตัวของนักเรียนจำแนกตามประเทศสัญชาติทั้งหมด
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- การเตรียมข้อมูล ---
    nationality_count = (
        filtered_df["NationalITy"]
        .value_counts()
        .reset_index()
    )
    nationality_count.columns = ["Nationality", "Count"]
    
    total_students = len(filtered_df)
    nationality_count["Percent"] = (nationality_count["Count"] / total_students * 100) if total_students > 0 else 0

    # 🎮 [ลูกเล่นที่ 1] แผงควบคุม Interactive Controls
    ctrl_col1, ctrl_col2 = st.columns([2, 3])

    with ctrl_col1:
        sort_order = st.selectbox(
            "↕️ การเรียงลำดับ:",
            options=["มากไปน้อย", "น้อยไปมาก"],
            index=0,
            key="nat_sort"
        )


    with ctrl_col2:
        # แสดง KPI แบบสรุปย่อความหลากหลายทางสัญชาติ
        total_countries = len(nationality_count)
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); 
                        border-radius: 12px; padding: 8px 15px; text-align: center; margin-top: 5px;'>
                <span style='color: #1E40AF; font-size: 12px; font-weight: 600;'>ความหลากหลายทางสัญชาติ</span><br>
                <b style='color: #1D4ED8; font-size: 18px;'>{total_countries:,} ประเทศ</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ปรับการเรียงลำดับข้อมูลตามที่ผู้ใช้เลือก
    is_ascending = True if sort_order == "น้อยไปมาก" else False
    category_order_setting = "total ascending" if not is_ascending else "total descending"

    dynamic_height = max(400, len(nationality_count) * 38)

    # --- 📊 [ลูกเล่นที่ 2] สร้างแผนภูมิ Plotly พร้อมการไล่เฉดสี และ Hover พิเศษ ---
    fig_nat = px.bar(
        nationality_count,
        x="Count",
        y="Nationality",
        orientation="h",
        text="Count",
        color="Count",
        color_continuous_scale="Cividis", # ใช้โทนสี Cividis/Blues ที่มีมิติ
        custom_data=["Percent"]
    )

    # ตกแต่งตัวแท่งกราฟ (Traces)
    fig_nat.update_traces(
        texttemplate="<b>%{x:,} คน</b> (%{customdata[0]:.1f}%)",
        textposition="outside",
        textfont=dict(size=12, color="#334155"),
        marker=dict(
            line=dict(width=0),
            cornerradius=8 # โค้งมนแบบ Modern UI
        ),
        hovertemplate="<b>🌐 ประเทศ:</b> %{y}<br>" +
                      "<b>👥 จำนวน:</b> %{x:,} คน<br>" +
                      "<b>📊 คิดเป็น:</b> %{customdata[0]:.2f}% ของทั้งหมด<extra></extra>"
    )

    max_count = nationality_count["Count"].max() if not nationality_count.empty else 100

    # ตกแต่ง Layout รวม
    fig_nat.update_layout(
        template="plotly_white",
        height=dynamic_height,
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=80, t=20, b=20),
        yaxis=dict(
            categoryorder=category_order_setting, 
            title="",
            tickfont=dict(size=13, color="#1E293B", family="Arial"),
            showline=False
        ),
        xaxis=dict(
            title="จำนวนนักเรียน (คน)",
            title_font=dict(size=12, color="#64748B"), # แก้ตรงนี้แล้ว
            gridcolor="#F1F5F9",
            zeroline=False,
            range=[0, max_count * 1.28]
        )
    )

    st.plotly_chart(fig_nat, use_container_width=True, config={"displayModeBar": False})

    # --- 💡 [ลูกเล่นที่ 3] Insight Box ตอบสนองตามตัวเลือก ---
    if not nationality_count.empty:
        top_nat = nationality_count.iloc[0]["Nationality"]
        top_count = nationality_count.iloc[0]["Count"]
        top_pct = nationality_count.iloc[0]["Percent"]

        st.markdown(
            f"""
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6;
                        border-radius: 10px; padding: 12px 16px; font-size: 13.5px; color: #334155; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.03); margin-top: 5px;'>
                💡 <b>ข้อสังเกตเชิงสถิติ:</b> ผู้เรียนส่วนใหญ่เป็นสัญชาติ <b>{top_nat}</b> 
                ครองสัดส่วนสูงสุดถึง <b>{top_count:,} คน ({top_pct:.1f}%)</b> จากผู้เรียนทั้งหมด {total_students:,} คน
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # --------------------------------------------
    # SEARCH & DATA TABLE
    # -------------------------------------------
    st.markdown("### 🔍 ค้นหาและดูข้อมูลนักเรียน")
    
    keyword = st.text_input("ระบุคำค้นหา (ค้นหาได้ทุกคอลัมน์):", placeholder="พิมพ์คำค้นหา...")

    if keyword:
        display_df = filtered_df[
            filtered_df.astype(str)
            .apply(lambda x: x.str.contains(keyword, case=False))
            .any(axis=1)
        ]
    else:
        display_df = filtered_df

    st.dataframe(display_df, use_container_width=True, height=400)

    csv = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="⬇️ ดาวน์โหลดรายงาน CSV",
        data=csv,
        file_name="student_dashboard_report.csv",
        mime="text/csv"
    )

# =====================================================
# ผลสัมฤทธิ์ทางการเรียน
# =====================================================
elif menu == "ผลสัมฤทธิ์ทางการเรียน":

    st.title("📈 การวิเคราะห์ผลสัมฤทธิ์ทางการเรียน")
    st.caption("การกระจายตัวของผลการเรียนแบ่งตามกลุ่ม Low (L), Medium (M) และ High (H)")
    st.markdown("---")

    col_chart, col_summary = st.columns([2, 1])

    with col_chart:
        fig = px.histogram(
            filtered_df,
            x="Class",
            color="Class",
            category_orders={"Class": ["L", "M", "H"]},
            title="สัดส่วนระดับผลการเรียน",
            color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
            template="plotly_white"
        ) 
        fig.update_layout(
            bargap=0.3,
            showlegend=False,
            height=400,
            xaxis_title="ระดับผลการเรียน (Class)",
            yaxis_title="จำนวนนักเรียน (คน)",
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_summary:
        st.markdown("### 💡 สรุปข้อมูล")
        class_counts = filtered_df["Class"].value_counts().reindex(["H", "M", "L"]).fillna(0)
    
        st.markdown(
            f"""
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px;'>
                <div style='margin-bottom: 12px; font-weight: 600; color: #1E293B;'>จำนวนนักเรียนจำแนกตามระดับ:</div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span>🟢 <b>High (H):</b></span>
                    <span style='color: #059669; font-weight: bold;'>{int(class_counts.get('H', 0)):,} คน</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span>🟡 <b>Medium (M):</b></span>
                    <span style='color: #D97706; font-weight: bold;'>{int(class_counts.get('M', 0)):,} คน</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span>🔴 <b>Low (L):</b></span>
                    <span style='color: #DC2626; font-weight: bold;'>{int(class_counts.get('L', 0)):,} คน</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# พฤติกรรมการเรียนรู้
# =====================================================
elif menu == "พฤติกรรมการเรียนรู้":

    st.title("📚 การวิเคราะห์พฤติกรรมการเรียนรู้")
    st.caption("วิเคราะห์ความสัมพันธ์ระหว่างการมีส่วนร่วมในห้องเรียนกับระดับผลการเรียน")
    st.markdown("---")

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
        height=520,
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =====================================================
# การเข้าเรียน
# =====================================================
elif menu == "การเข้าเรียน":

    st.title("🗓️ การวิเคราะห์การขาดเรียน")
    st.caption("เปรียบเทียบผลกระทบของวันขาดเรียน (น้อยกว่า 7 วัน vs มากกว่า 7 วัน) ต่อระดับผลการเรียน")
    st.markdown("---")

    fig = px.histogram(
        filtered_df,
        x="StudentAbsenceDays",
        color="Class",
        barmode="group",
        title="จำนวนวันขาดเรียนเทียบกับระดับผลการเรียน",
        category_orders={"Class": ["L", "M", "H"]},
        color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
        template="plotly_white",
        labels={"StudentAbsenceDays": "กลุ่มวันขาดเรียน", "Class": "ผลการเรียน"}
    )

    fig.update_layout(
        bargap=0.2,
        bargroupgap=0.1,
        height=450,
        xaxis_title="ระยะเวลาการขาดเรียน",
        yaxis_title="จำนวนนักเรียน (คน)",
        font=dict(family="Plus Jakarta Sans, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =====================================================
# การมีส่วนร่วมของผู้ปกครอง
# =====================================================
elif menu == "การมีส่วนร่วมของผู้ปกครอง":

    st.title("👨‍👩‍👧 การวิเคราะห์การมีส่วนร่วมของผู้ปกครอง")
    st.caption("วิเคราะห์สัดส่วนความร่วมมือของผู้ดูแลหลัก ความพึงพอใจต่อโรงเรียน และผลการเรียนของผู้เรียน")
    st.markdown("---")

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

    st.title("🔗 การวิเคราะห์ความสัมพันธ์เชิงพฤติกรรม (Correlation)")
    st.caption("เมทริกซ์ความสัมพันธ์เชิงตัวเลขระหว่างพฤติกรรมการเรียนรู้ด้านต่าง ๆ ของผู้เรียน")
    st.markdown("---")

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
        height=480,
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

    # Prepare ML Model
    model_df = df.copy()

    encoders = {}
    feature_cols = [c for c in model_df.columns if c != "Class"]

    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            encoders[col] = le

    X = model_df[feature_cols]
    y = model_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    tab1, tab2 = st.tabs(["📊 ประสิทธิภาพโมเดล (Model Performance)", "🔮 ทำนายผลการเรียนผู้เรียนใหม่ (Live Prediction)"])

    # TAB 1: MODEL PERFORMANCE
    with tab1:
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
                height=350,
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                template="plotly_white"
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

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
            height=400,
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    # TAB 2: LIVE PREDICTION
    with tab2:
        st.markdown("### 📝 ป้อนข้อมูลพฤติกรรมของนักเรียนใหม่")
        st.caption("กรอกข้อมูลการเรียนและการมีส่วนร่วมด้านล่างเพื่อประเมินเกรดคาดการณ์")

        with st.form("prediction_form"):
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                st.markdown("**👤 ข้อมูลทั่วไป**")
                input_gender = st.selectbox("Gender (เพศ)", df["gender"].unique())
                input_national = st.selectbox("NationalITy (สัญชาติ)", df["NationalITy"].unique())
                input_place = st.selectbox("PlaceofBirth (สถานที่เกิด)", df["PlaceofBirth"].unique())
                input_stage = st.selectbox("StageID (ระดับชั้น)", df["StageID"].unique())
                input_grade = st.selectbox("GradeID (ชั้นปี)", df["GradeID"].unique())
                input_section = st.selectbox("SectionID (ห้องเรียน)", df["SectionID"].unique())

            with col_f2:
                st.markdown("**📚 พฤติกรรมการเรียน**")
                input_hands = st.number_input("Raised Hands (จำนวนครั้งที่ยกมือ)", min_value=0, max_value=100, value=25)
                input_resources = st.number_input("Visited Resources (การเข้าดูสื่อ)", min_value=0, max_value=100, value=50)
                input_announcements = st.number_input("Announcements View (การดูประกาศ)", min_value=0, max_value=100, value=20)
                input_discussion = st.number_input("Discussion (การพูดคุยแลกเปลี่ยน)", min_value=0, max_value=100, value=15)
                input_topic = st.selectbox("Topic (วิชาเรียน)", df["Topic"].unique())

            with col_f3:
                st.markdown("**👨‍👩‍👧 ข้อมูลการขาดเรียน & ผู้ปกครอง**")
                input_semester = st.selectbox("Semester (ภาคเรียน)", df["Semester"].unique())
                input_relation = st.selectbox("Relation (ผู้ดูแลหลัก)", df["Relation"].unique())
                input_absence = st.selectbox("StudentAbsenceDays (วันขาดเรียน)", df["StudentAbsenceDays"].unique())
                input_parent_sat = st.selectbox("ParentschoolSatisfaction (ความพึงพอใจผู้ปกครอง)", df["ParentschoolSatisfaction"].unique())
                input_parent_survey = st.selectbox("ParentAnsweringSurvey (การตอบแบบสอบถาม)", df["ParentAnsweringSurvey"].unique())

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
            input_data = input_data[feature_cols]

            for col in input_data.columns:
                if col in encoders:
                    try:
                        input_data[col] = encoders[col].transform(input_data[col].astype(str))
                    except ValueError:
                        input_data[col] = 0

            prediction_encoded = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            predicted_class = encoders["Class"].inverse_transform([prediction_encoded])[0]

            st.markdown("---")
            st.markdown("### 🎯 ผลการทำนาย (Prediction Result)")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                if predicted_class == "H":
                    st.success("🎉 **ระดับผลการเรียนคาดการณ์: High (H)**")
                elif predicted_class == "M":
                    st.warning("⚡ **ระดับผลการเรียนคาดการณ์: Medium (M)**")
                else:
                    st.error("🚨 **ระดับผลการเรียนคาดการณ์: Low (L)**")

            with res_col2:
                prob_df = pd.DataFrame({
                    "Class": encoders["Class"].classes_,
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
                fig_prob.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})