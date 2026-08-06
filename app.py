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
# PROFESSIONAL CUSTOM CSS
# ------------------------
st.markdown("""
    <style>
    /* 1. Font & Overall Layout */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* 2. Premium Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.03);
        border-color: #CBD5E1;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* 3. Sleek Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; /* Dark Slate Theme */
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #F8FAFC !important;
        font-weight: 600;
    }

    /* Sidebar Multiselect & Selectbox */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }

    /* 4. Headings & Divider */
    h1 {
        font-weight: 800 !important;
        color: #0F172A !important;
        letter-spacing: -0.025em;
    }

    h2, h3, .stSubheader {
        font-weight: 700 !important;
        color: #1E293B !important;
        letter-spacing: -0.02em;
    }

    hr {
        margin: 1.5rem 0;
        border-color: #E2E8F0;
    }

    /* 5. Modern Dataframe / Table */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* 6. Custom Alert Info Box */
    div.stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# COLOR PALETTE
# ------------------------
# ใช้โทนสีสไตล์ Modern Dashboard (ดึงความพรีเมียมของสี)
COLOR_SEQUENCE = px.colors.qualitative.Bold

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
    # 1. Header & Brand Logo
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    
    st.markdown(
        "<h2 style='text-align: center; margin-top: -10px; font-size: 1.25rem;'>การวิเคราะห์ผลการเรียนรู้ของผู้เรียน</h2>", 
        unsafe_allow_html=True
    )
    st.markdown("---")

    # 2. Main Navigation Menu
    st.markdown("##### 🧭 เมนูนำทาง")
    menu = st.selectbox(
        "เลือกหน้าต่างแดชบอร์ด",
        [
            "ภาพรวม",
            "ผลสัมฤทธิ์ทางการเรียน",
            "พฤติกรรมการเรียนรู้",
            "การเข้าเรียน",
            "การมีส่วนร่วมของผู้ปกครอง",
            "การวิเคราะห์ความสัมพันธ์ของข้อมูล",
            "การทำนายผลการเรียนรู้"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Data Filters
    st.markdown("##### 🔍 ตัวกรองทั้งหมด")
    
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

    # 4. Filter Query Logic
    filtered_df = df[
        (df["gender"].isin(gender))
        & (df["StageID"].isin(stage))
        & (df["Semester"].isin(semester))
    ]

    # 5. Data Summary Mini Badge
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"📊 แสดงผลข้อมูล **{len(filtered_df):,}** จากทั้งหมด **{len(df):,}** รายการ")

# =====================================================
# OVERVIEW
# =====================================================
if menu == "ภาพรวม":

    st.title("📌 แดชบอร์ดภาพรวมผู้เรียน")
    st.caption("ภาพรวมข้อมูลนักเรียน ผลการเรียน พฤติกรรมการเรียน และข้อมูลเชิงสถิติ")
    st.markdown("---")
    # =====================================================
    # ROW 1
    # =====================================================

    chart1, chart2 = st.columns(2)
    with chart1:

        st.markdown(
            """
            <div style='text-align: left; margin-bottom: 10px;'>
                <h3 style='color: #1E293B; margin: 0; font-size: 30px; font-weight: 700;'>👤 สัดส่วนของผู้เรียน</h3>
                <p style='color: #64748B; font-size: 17px; margin: 2px 0 0 0;'>สัดส่วนนักเรียนจำแนกตามเพศ</p>
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

        # เปลี่ยนชื่อให้สวยขึ้น (ถ้าข้อมูลเป็น M/F)
        gender_count["Gender"] = gender_count["Gender"].replace({
            "M": "เพศชาย",
            "F": "เพศหญิง"
        })

        total_students = gender_count["Count"].sum()

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
            textfont=dict(
                size=15,
                color="white",
                family="Arial"
            ),

            hovertemplate=
            "<b>%{label}</b><br>" +
            "จำนวน : %{value:,}<br>" +
            "สัดส่วน : %{percent}<extra></extra>",

            marker=dict(
                line=dict(
                    color="white",
                    width=3
                )
            )
        )

        fig_gender.update_layout(
            
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    text=f"<span style='color:#64748B; font-size:12px;'>จำนวนนักเรียนทั้งหมด</span><br><br>"
                         f"<b style='font-size:28px; color:#1E293B;'>{total_students:,} คน</b><br><br>",
                    showarrow=False
                )
            ],

            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.05,
                x=0.5,
                xanchor="center",
                font=dict(size=14)
            ),

            paper_bgcolor="white",
            plot_bgcolor="white",

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=40
            ),

            height=420
        )

        st.plotly_chart(
            fig_gender,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with chart2:

        # คำนวณตัวเลขสำหรับแสดงในการ์ดฝั่งขวา
        males = len(filtered_df[filtered_df["gender"] == "M"])
        females = len(filtered_df[filtered_df["gender"] == "F"])
        male_pct = (males / total_students * 100) if total_students > 0 else 0
        female_pct = (females / total_students * 100) if total_students > 0 else 0

        st.markdown(
            f"""
            <div style='background-color: #DBEAFE; color: #1E40AF; text-align: center; 
                        padding: 8px; border-radius: 8px; font-weight: bold; margin-bottom: 12px;'>
                สรุปภาพรวม
            </div>
        
            <!-- การ์ดเพศชาย -->
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
            <!-- การ์ดเพศหญิง -->
            <div style='background-color: #FFF1F2; border-radius: 12px; padding: 15px;'>
                <div style='display: flex; align-items: center;'>
                    <div style='background-color: #FF6B9A; color: white; width: 45px; 
                                height: 45px; border-radius: 50%; display: flex; 
                                align-items: center; justify-content: center; 
                                font-size: 22px; margin-right: 12px;'>👩</div>
                    <div>
                        <div style='color: #1E293B; font-weight: bold; 
                                    font-size: 15px;'>เพศหญิง</div>
                        <div style='color: #FF6B9A; font-size: 24px; 
                                    font-weight: bold;'>{females:,} 
                                    <span style='font-size: 14px; font-weight: normal; 
                                    color: #475569;'>คน</span></div>
                    </div>
                </div>
                <hr style='margin: 8px 0; border-top: 1px solid #FFE4E6;'>
                <div style='color: #FF6B9A; font-weight: bold; font-size: 14px;'>{female_pct:.1f}%</div>
            </div>

            <!-- การ์ดรวมทั้งหมด -->
            <div style='background-color: #DCFCE7; border: 1px solid #BBF7D0; border-radius: 12px; padding: 15px; margin-bottom: 12px;'>
                <div style='display: flex; align-items: center;'>
                    <!-- วงกลมสีกรมท่าเข้ม + ไอคอนคนสีขาวสว่าง -->
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
                        <div style='color: #059669; font-size: 24px; font-weight: bold;'>{(males + females):,} 
                                    <span style='font-size: 14px; font-weight: normal; color: #475569;'>คน</span></div>
                    </div>
                </div>
                <hr style='margin: 8px 0; border-top: 1px solid #E2E8F0;'>
                <div style='color: #059669; font-weight: bold; font-size: 14px;'>100.0%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # แถบหมายเหตุ Footer ด้านล่างสุด
    st.markdown(
        f"""
        <div style='background-color: #F8FAFC; color: #64748B; padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-top: 10px;'>
            ℹ️ ข้อมูลนี้แสดงสัดส่วนของนักเรียนจำแนกตามเพศ จากข้อมูลทั้งหมด {total_students:,} คน
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # ROW 2
    # =====================================================

    chart3, chart4 = st.columns(2)

    with chart3:
        st.markdown("## 📅 จำนวนนักเรียน")
        st.caption("จำนวนนักเรียนแยกตามภาคเรียน")

        semester_count = (
            filtered_df["Semester"]
            .value_counts()
            .reset_index()
        )

        semester_count.columns = ["Semester", "Count"]

        # เปลี่ยนชื่อ F / S เป็น Semester 1 / Semester 2
        semester_count["Semester"] = semester_count["Semester"].replace({
            "F": "ภาคเรียนที่ 1",
            "S": "ภาคเรียนที่ 2"
        })

        total_students = semester_count["Count"].sum()

        semester1 = semester_count.loc[
            semester_count["Semester"] == "ภาคเรียนที่ 1",
            "Count"
        ].sum()

        semester2 = semester_count.loc[
            semester_count["Semester"] == "ภาคเรียนที่ 2",
            "Count"
        ].sum()

        percent1 = (semester1 / total_students * 100) if total_students > 0 else 0
        percent2 = (semester2 / total_students * 100) if total_students > 0 else 0

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
            textfont=dict(
                size=18,
                color="white"
            ),
            marker=dict(
                line=dict(
                    color="white",
                    width=4
                )
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Students : %{value:,} คน<br>"
                "Percentage : %{percent}<extra></extra>"
            )   
        )

        fig_semester.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            legend=dict(
                orientation="h",
                y=-0.05,
                x=0.5,
                xanchor="center",
                font=dict(size=14)
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    text=f"<span style='color:#64748B; font-size:12px;'>จำนวนนักเรียนทั้งหมด</span><br><br>"
                            f"<b style='font-size:28px; color:#1E293B;'>{total_students:,} คน</b><br><br>",
                            showarrow=False
                     )
            ]
        )

        st.plotly_chart(
            fig_semester,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # ==========================================================
    # Chart 4: Summary Cards & Insight
    # ==========================================================
    with chart4:
        # 1. คำนวณข้อมูล (ย่อหน้าต้องตรงกับระดับใน with)
        major_semester = "ภาคเรียนที่ 1" if semester1 >= semester2 else "ภาคเรียนที่ 2"
        major_percent = max(percent1, percent2)

        # 2. Render HTML (เขียนชิดกันเพื่อป้องกัน Streamlit อ่าน CSS พลาด)
        st.markdown(
            f"""
        <div style='display: flex; flex-direction: column; justify-content: center; min-height: 450px;'>
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
    # =====================================================
    # ROW 3
    # =====================================================
    chart5 = st.columns(1)[0]
    with chart5:
        # ---------------------------------------------------------
        # ส่วนที่ 1: เตรียมและคำนวณข้อมูล (Data Preparation)
        # ---------------------------------------------------------
        # นับจำนวนนักเรียนตามระดับผลการเรียน (H, M, L)
        class_order = ["H", "M", "L"]
        class_df = (
            filtered_df["Class"]
            .value_counts()
            .reindex(class_order, fill_value=0)
            .reset_index()
        )
        class_df.columns = ["Class", "Count"]

        total_students_perf = class_df["Count"].sum()

        # คำนวณ % และสร้างป้ายชื่อแกน X
        class_df["Percent"] = (
            (class_df["Count"] / total_students_perf * 100)
            if total_students_perf > 0
            else 0
        )

        labels_map = {
            "H": "H<br>(High Performance)",
            "M": "M<br>(Medium Performance)",
            "L": "L<br>(Low Performance)",
        }
        class_df["Label"] = class_df["Class"].map(labels_map)

        # สีประจำแต่ละระดับ (เขียว, น้ำเงิน, ส้ม)
        colors = ["#4CAF50", "#2962FF", "#FFA726"]

        # ---------------------------------------------------------
        # ส่วนที่ 2: ส่วนหัวและ การ์ดจำนวนรวม (Header & Card)
        # ---------------------------------------------------------
        col_title, col_card = st.columns([2.5, 1])

        with col_title:
            st.markdown(
                """
                <h2 style='margin:0; font-size:22px;'>ผลสัมฤทธิ์ทางการเรียนของผู้เรียน</h2>
                <p style='margin:0; color:#64748B; font-size:14px;'>
                            จำนวนและร้อยละของนักเรียนในแต่ละระดับผลการเรียน (H, M, L)</p>
            """,
                unsafe_allow_html=True,
            )

        with col_card:
            st.markdown(
                f"""
                <div style='background-color: #EFF6FF; border-radius: 10px; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between;'>
                    <span style='font-size: 20px;'>👥</span>
                    <div style='text-align: right;'>
                        <div style='color: #64748B; font-size: 11px; font-weight: bold;'>จำนวนนักเรียนทั้งหมด</div>
                        <div style='color: #1E3A8A; font-size: 18px; font-weight: bold;'>{total_students_perf:,}</div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------
        # ส่วนที่ 3: สร้างกราฟแท่ง (ใช้ px.bar แทน go.Figure)
        # ---------------------------------------------------------
        fig_perf = px.bar(
            class_df,
            x="Label",
            y="Count",
            text="Count",
            color="Class",
            color_discrete_map={
                "H": "#4CAF50",
                "M": "#2962FF",
                "L": "#FFA726"
            }
        )

        # จัดสไตล์ตัวเลขบนแท่ง และ % กลางแท่ง
        fig_perf.update_traces(
            texttemplate='<b>%{y:,}</b>',
            textposition='outside',
            textfont=dict(size=14, color='#1E293B')
        )

        # ใส่ % กลางแท่งกราฟด้วย Annotation
        for idx, row in class_df.iterrows():
            fig_perf.add_annotation(
                x=row["Label"],
                y=row["Count"] / 2, # ปักไว้ตรงกลางความสูงของแท่ง
                text=f"<b>{row['Percent']:.1f}%</b>",
                showarrow=False,
                font=dict(size=13, color="white")
            )

        # ปรับการแสดงผล Layout
        max_val = class_df["Count"].max()
        fig_perf.update_layout(
            template="plotly_white",
            height=380,
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40)
        )
        fig_perf.update_xaxes(title="")
        fig_perf.update_yaxes(
            title="จำนวนนักเรียน (คน)",
            range=[0, max_val * 1.25 if max_val > 0 else 100],
            gridcolor="#F1F5F9"
        )

        # แสดงกราฟลง Streamlit
        st.plotly_chart(fig_perf, use_container_width=True, 
                        config={"displayModeBar": False}
        )
    # =====================================================
    # TOP NATIONALITY
    # =====================================================

    nationality_count = (
        filtered_df["NationalITy"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    nationality_count.columns = [
        "Nationality",
        "Count"
    ]

    fig_nat = px.bar(
        nationality_count,
        x="Count",
        y="Nationality",
        orientation="h",
        text_auto=True,
        title="🌍 10 อันดับสัญชาติที่มีจำนวนมากที่สุด",
        color="Count",
        color_continuous_scale="Blues"
    )

    fig_nat.update_layout(
        template="plotly_white",
        height=500,
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig_nat,
        use_container_width=True
    )

    # =====================================================
    # SEARCH
    # =====================================================

    st.subheader("🔍 Search Student Data")

    keyword = st.text_input(
        "ค้นหาข้อมูลในตาราง"
    )

    if keyword:

        display_df = filtered_df[
            filtered_df.astype(str)
            .apply(
                lambda x: x.str.contains(
                    keyword,
                    case=False
                )
            )
            .any(axis=1)
        ]

    else:
        display_df = filtered_df

    # =====================================================
    # DATA TABLE
    # =====================================================

    st.subheader("📋 Student Dataset")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv = display_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download CSV Report",
        data=csv,
        file_name="student_dashboard_report.csv",
        mime="text/csv"
    )
    
# =====================================================
# ACADEMIC PERFORMANCE
# =====================================================
elif menu == "Academic Performance":

    st.title("📈 Academic Performance Analysis")
    st.caption("การกระจายตัวของผลการเรียนแบ่งตามกลุ่ม Low (L), Medium (M) และ High (H)")
    st.markdown("---")

    col_chart, col_summary = st.columns([2, 1])

    with col_chart:
        fig = px.histogram(
            filtered_df,
            x="Class",
            color="Class",
            category_orders={"Class": ["L", "M", "H"]},
            title="Academic Performance Distribution",
            color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"}, # Red, Amber, Emerald
            template="plotly_white"
        )
        fig.update_layout(
            bargap=0.3,
            showlegend=False,
            height=400,
            xaxis_title="Performance Class",
            yaxis_title="Student Count",
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.subheader("💡 Summary")
        class_counts = filtered_df["Class"].value_counts().reindex(["H", "M", "L"]).fillna(0)
        
        st.write("จำนวนนักเรียนแบ่งตามระดับ:")
        st.write(f"🟢 **High (H):** {int(class_counts.get('H', 0))} คน")
        st.write(f"🟡 **Medium (M):** {int(class_counts.get('M', 0))} คน")
        st.write(f"🔴 **Low (L):** {int(class_counts.get('L', 0))} คน")

# =====================================================
# LEARNING BEHAVIOR
# =====================================================
elif menu == "Learning Behavior":

    st.title("📚 Learning Behavior Analysis")
    st.caption("วิเคราะห์พฤติกรรมการมีส่วนร่วมในห้องเรียนกับระดับผลการเรียน")
    st.markdown("---")

    fig = px.scatter(
        filtered_df,
        x="raisedhands",
        y="VisITedResources",
        color="Class",
        size="Discussion",
        hover_data=["AnnouncementsView"],
        title="Raised Hands vs Visited Resources (Size = Discussion)",
        labels={
            "raisedhands": "Raised Hands (ยกมือตอบคำถาม)",
            "VisITedResources": "Visited Resources (เข้าดูสื่อการเรียน)",
            "Class": "Performance"
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

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ATTENDANCE
# =====================================================
elif menu == "Attendance":

    st.title("🗓 Attendance Analysis")
    st.caption("เปรียบเทียบผลกระทบของวันขาดเรียน (Under-7 Days vs Above-7 Days) ต่อผลการเรียน")
    st.markdown("---")

    fig = px.histogram(
        filtered_df,
        x="StudentAbsenceDays",
        color="Class",
        barmode="group",
        title="Student Absence Days vs Academic Performance",
        category_orders={"Class": ["L", "M", "H"]},
        color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
        template="plotly_white",
        labels={"StudentAbsenceDays": "Absence Duration"}
    )

    fig.update_layout(
        bargap=0.2,
        bargroupgap=0.1,
        height=450,
        xaxis_title="Absence Days Group",
        yaxis_title="Student Count",
        font=dict(family="Plus Jakarta Sans, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PARENT INVOLVEMENT
# =====================================================
elif menu == "Parent Involvement":

    st.title("👨‍👩‍👧 Parent Involvement Analysis")
    st.caption("วิเคราะห์สัดส่วนความร่วมมือของผู้ปกครอง (ผู้ดูแลหลัก) ความพึงพอใจต่อโรงเรียน และผลการเรียนของนักเรียน")
    st.markdown("---")

    col_chart, col_info = st.columns([2, 1])

    with col_chart:

        # Copy data และแปลง Class เป็นข้อความ ป้องกัน Plotly Categorical Error
        plot_df = filtered_df.copy()

        plot_df["Class"] = plot_df["Class"].astype(str)
        plot_df["Relation"] = plot_df["Relation"].astype(str)
        plot_df["ParentschoolSatisfaction"] = plot_df["ParentschoolSatisfaction"].astype(str)

        fig = px.sunburst(
            plot_df,
            path=[
                "Relation",
                "ParentschoolSatisfaction",
                "Class"
            ],
            title="Parent Hierarchy & School Satisfaction Breakdown",
            color="Class",
            color_discrete_map={
                "L": "#EF4444",
                "M": "#F59E0B",
                "H": "#10B981"
            }
        )

        fig.update_layout(
            margin=dict(t=40, l=0, r=0, b=0),
            height=500,
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_info:

        st.subheader("💡 Key Insights")

        st.info("""
**โครงสร้างข้อมูลแผนภูมิ Sunburst**

🟢 วงในสุด (Inner):
ผู้รับผิดชอบหลัก (Father / Mother)

🟡 วงกลาง (Middle):
ความพึงพอใจต่อโรงเรียน (Good / Bad)

🔴 วงนอกสุด (Outer):
ผลการเรียนของนักเรียน (H / M / L)

📌 คลิกที่แต่ละส่วนของกราฟเพื่อเจาะลึกข้อมูลแต่ละกลุ่มได้
""")

# =====================================================
# RELATIONSHIP ANALYSIS
# =====================================================
elif menu == "Relationship Analysis":

    st.title("🔗 Relationship Analysis (Correlation Matrix)")
    st.caption("ความสัมพันธ์เชิงตัวเลขระหว่างพฤติกรรมการเรียนรู้ด้านต่างๆ ของนักเรียน")
    st.markdown("---")

    corr_cols = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
    corr_matrix = filtered_df[corr_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Behavioral Metrics Correlation Heatmap",
        labels=dict(color="Correlation"),
        x=["Raised Hands", "Visited Resources", "Announcements View", "Discussion"],
        y=["Raised Hands", "Visited Resources", "Announcements View", "Discussion"]
    )

    fig.update_layout(
        height=480,
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 ดูแผนภูมิ Scatter Matrix แบบโต้ตอบเพิ่มเติม"):
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
# RANDOM FOREST PREDICTION
# =====================================================
elif menu == "Random Forest Prediction":

    st.title("🤖 Random Forest Prediction Model")
    st.caption("การประเมินประสิทธิภาพโมเดล และระบบจำแนกระดับผลการเรียนสำหรับนักเรียนรายบุคคล")
    st.markdown("---")

    # 1. Model Training & Data Preprocessing
    model_df = df.copy()

    encoders = {}
    for col in model_df.columns:
        if model_df[col].dtype == "object":
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            encoders[col] = le

    X = model_df.drop("Class", axis=1)
    y = model_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    # 2. Tabs Layout
    tab1, tab2 = st.tabs(["📊 Model Performance & Insights", "🔮 Live Student Prediction (นักเรียนใหม่)"])

    # -------------------------------------------------
    # TAB 1: MODEL PERFORMANCE
    # -------------------------------------------------
    with tab1:
        col_acc, col_cm = st.columns([1, 2])
        
        with col_acc:
            st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
            st.info("""
            **เกี่ยวกับการทำนาย:**
            โมเดลประมวลผลข้อมูลจากปัจจัย 16 ตัวแปร (เช่น วันขาดเรียน, การเข้าดูสื่อ, การมีส่วนร่วมในห้อง) เพื่อจำแนกเกรดออกเป็น:
            * **H (High):** คะแนนสูง
            * **M (Medium):** คะแนนปานกลาง
            * **L (Low):** คะแนนต่ำ
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
                labels=dict(x="Predicted Class", y="Actual Class")
            )
            fig_cm.update_layout(
                height=350,
                font=dict(family="Plus Jakarta Sans, sans-serif"),
                template="plotly_white"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")

        # Feature Importance
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig_imp = px.bar(
            importance.tail(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Key Factors Affecting Student Grades",
            color="Importance",
            color_continuous_scale="Blugrn",
            template="plotly_white"
        )
        fig_imp.update_layout(
            height=400,
            font=dict(family="Plus Jakarta Sans, sans-serif")
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("📄 ดูตารางค่า Feature Importance ทั้งหมด"):
            st.dataframe(importance.sort_values(by="Importance", ascending=False), use_container_width=True)

    # -------------------------------------------------
    # TAB 2: LIVE PREDICTION FOR NEW STUDENT
    # -------------------------------------------------
    with tab2:
        st.subheader("📝 ป้อนข้อมูลพฤติกรรมของนักเรียนใหม่")
        st.caption("กรอกข้อมูลการเรียนและการมีส่วนร่วมด้านล่างเพื่อประเมินระดับผลการเรียนคาดการณ์")

        with st.form("prediction_form"):
            col_f1, col_f2, col_f3 = st.columns(3)

            with col_f1:
                st.markdown("**👤 ข้อมูลทั่วไป**")
                input_gender = st.selectbox("Gender", df["gender"].unique())
                input_national = st.selectbox("NationalITy", df["NationalITy"].unique())
                input_place = st.selectbox("PlaceofBirth", df["PlaceofBirth"].unique())
                input_stage = st.selectbox("StageID", df["StageID"].unique())
                input_grade = st.selectbox("GradeID", df["GradeID"].unique())
                input_section = st.selectbox("SectionID", df["SectionID"].unique())

            with col_f2:
                st.markdown("**📚 พฤติกรรมการเรียน**")
                input_hands = st.number_input("Raised Hands (จำนวนครั้งที่ยกมือ)", min_value=0, max_value=100, value=25)
                input_resources = st.number_input("Visited Resources (จำนวนครั้งที่เข้าดูสื่อ)", min_value=0, max_value=100, value=50)
                input_announcements = st.number_input("Announcements View (จำนวนครั้งที่เปิดดูประกาศ)", min_value=0, max_value=100, value=20)
                input_discussion = st.number_input("Discussion (จำนวนครั้งที่พูดคุยร่วมกัน)", min_value=0, max_value=100, value=15)
                input_topic = st.selectbox("Topic (วิชาที่เรียน)", df["Topic"].unique())

            with col_f3:
                st.markdown("**👨‍👩‍👧 การขาดเรียน & ผู้ปกครอง**")
                input_semester = st.selectbox("Semester", df["Semester"].unique())
                input_relation = st.selectbox("Relation (ผู้ดูแลหลัก)", df["Relation"].unique())
                input_absence = st.selectbox("StudentAbsenceDays (วันขาดเรียน)", df["StudentAbsenceDays"].unique())
                input_parent_sat = st.selectbox("ParentschoolSatisfaction", df["ParentschoolSatisfaction"].unique())
                input_parent_survey = st.selectbox("ParentAnsweringSurvey", df["ParentAnsweringSurvey"].unique())

            submit_button = st.form_submit_button("🔮 ทำนายผลการเรียน (Predict Performance)", use_container_width=True)

        if submit_button:
            # Construct input dataframe
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

            # Apply same LabelEncoding
            for col in input_data.columns:
                if col in encoders:
                    # กรณีเจอ Category ใหม่ที่โมเดลไม่เคยเรียนรู้ ให้ตั้งค่า fallback
                    try:
                        input_data[col] = encoders[col].transform(input_data[col])
                    except ValueError:
                        input_data[col] = 0

            # Prediction
            prediction_encoded = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]

            predicted_class = encoders["Class"].inverse_transform([prediction_encoded])[0]

            # Display Prediction Result Card
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
                # Probability Distribution
                prob_df = pd.DataFrame({
                    "Class": encoders["Class"].classes_,
                    "Probability": probabilities
                })
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    title="Prediction Confidence / Probability",
                    text_auto=".1%",
                    color="Class",
                    color_discrete_map={"L": "#EF4444", "M": "#F59E0B", "H": "#10B981"},
                    template="plotly_white"
                )
                fig_prob.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)