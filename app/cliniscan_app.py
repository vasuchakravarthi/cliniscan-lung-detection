import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - Lung Abnormality Detection",
    layout="wide"
)

# --------------------------------------------------
# PAGE STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page):
    st.session_state.page = page
    st.rerun()


# --------------------------------------------------
# HEADER (TITLE + SIDEBAR)
# --------------------------------------------------

def show_header():

    st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

    st.markdown("""
    Upload a **Chest X-ray** image to:

    - 🎯 Detect **14 lung abnormalities** with bounding boxes (YOLOv8-M, mAP: 0.4305)  
    - 📊 Get **overall classification**: Abnormal vs Normal (EfficientNet-B3, Acc: 95.20%)  
    - 🧠 View **Grad-CAM heatmap** showing model focus areas  

    **Note**: Classification trained on 512×512 images, optimized for chest X-ray analysis.
    """)

    with st.sidebar:

        st.header("ℹ️ About CliniScan")

        st.markdown("""
        **14 Detectable Abnormalities**:

        1. Aortic enlargement  
        2. Atelectasis  
        3. Calcification  
        4. Cardiomegaly  
        5. Consolidation  
        6. ILD  
        7. Infiltration  
        8. Lung Opacity  
        9. Nodule/Mass  
        10. Other lesion  
        11. Pleural effusion  
        12. Pleural thickening  
        13. Pneumothorax  
        14. Pulmonary fibrosis  

        **Classification Classes**

        - Abnormal (Class 0)  
        - Normal (Class 1)

        **⚠️ Disclaimer**: Educational purposes only.
        """)

        st.markdown("---")

        st.markdown("**Developer**: Vasu Chakravarthi")
        st.markdown("**Institution**: SRKR Engineering College")

        st.markdown(
            "[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)"
        )


# --------------------------------------------------
# FOOTER (DISCLAIMER)
# --------------------------------------------------

def show_footer():

    st.markdown("---")

    st.markdown(
    """
    <div style='text-align: center; color: gray;'>

    <p><strong>⚠️ DISCLAIMER</strong></p>

    <p>This system is for <strong>educational and research purposes only</strong>.</p>

    <p>It should NOT be used for clinical diagnosis or medical decision-making.</p>

    <p>Always consult a qualified radiologist for medical interpretation of chest X-rays.</p>

    <hr>

    <p><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>

    <p>
    <a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>
    GitHub Repository
    </a>
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

def home_page():

    show_header()

    st.subheader("🏠 Welcome to CliniScan")

    st.write(
        """
        **CliniScan** is an AI-powered system designed to assist in
        detecting lung abnormalities from chest X-ray images.

        This system combines:

        - **YOLOv8 detection** for 14 lung abnormalities
        - **EfficientNet classification**
        - **Grad-CAM explainability**

        Please login or start a free trial to test the system.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔐 Login"):
            go_to("login")

    with col2:
        if st.button("🧪 Free Trial"):
            go_to("trial")

    show_footer()


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    show_header()

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "cliniscan":
            go_to("dashboard")

        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back to Home"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# FREE TRIAL PAGE
# --------------------------------------------------

def trial_page():

    show_header()

    st.subheader("🧪 Free Trial")

    st.info("Upload a chest X-ray to test the AI system.")

    if st.button("Start Trial"):
        go_to("dashboard")

    if st.button("⬅ Back to Home"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

def dashboard_page():

    show_header()

    st.subheader("📤 Upload Chest X-ray")

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        from PIL import Image
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded X-ray")

        st.success("Your AI detection + GradCAM code goes here.")

    if st.button("Logout"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "login":
    login_page()

elif st.session_state.page == "trial":
    trial_page()

elif st.session_state.page == "dashboard":
    dashboard_page()
