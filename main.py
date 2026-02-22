# main.py

import streamlit as st
import cv2
import numpy as np
import time
import json
from pathlib import Path

from config import (
    LAPTOP_CAM_INDEX,
    PHONE_CAM_URL,
    SESSION_DURATION_SECONDS,
    EVIDENCE_DIR
)
from modules.dual_camera import (
    DualCameraSystem,
    CameraFrame,
    create_dual_view
)
from modules.face_analyzer import FaceAnalyzer
from modules.body_analyzer import BodyAnalyzer
from modules.avatar_3d import (
    create_pose_avatar_3d,
    create_face_avatar_3d
)
from modules.risk_engine import RiskEngine
from modules.report_generator import ReportGenerator
from modules.chatbot import AutismScreeningChatbot


# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="NeuroLens AI - Autism Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2980b9, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #2980b9;
        margin: 5px 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #641E16, #922B21);
        border-left: 4px solid #E74C3C;
        color: white;
        padding: 15px;
        border-radius: 12px;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #7D6608, #9A7D0A);
        border-left: 4px solid #F1C40F;
        color: white;
        padding: 15px;
        border-radius: 12px;
    }
    .risk-low {
        background: linear-gradient(135deg, #0E6251, #148F77);
        border-left: 4px solid #2ECC71;
        color: white;
        padding: 15px;
        border-radius: 12px;
    }
    .evidence-tag-high {
        background-color: #E74C3C;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    .evidence-tag-medium {
        background-color: #F1C40F;
        color: black;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
    }
    .stChatMessage {
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# SESSION STATE INITIALIZATION
# =============================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'app_state': 'setup',      # setup, screening, results
        'camera_system': None,
        'face_analyzer': None,
        'body_analyzer': None,
        'risk_engine': None,
        'chatbot': None,
        'session_start_time': None,
        'is_dual_mode': False,
        'phone_url': '',
        'subject_id': 'Anonymous',
        'face_stats': {},
        'body_stats': {},
        'assessment': None,
        'report_path': None,
        'chat_messages': [],
        'frame_count': 0,
        'last_face_result': None,
        'last_body_result': None,
        'screening_active': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
def render_sidebar():
    """Render the sidebar configuration panel."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Subject Info
        st.markdown("### 👤 Subject")
        st.session_state.subject_id = st.text_input(
            "Subject ID (optional)",
            value=st.session_state.subject_id,
            placeholder="e.g., CHILD-001"
        )

        st.markdown("---")

        # Camera Config
        st.markdown("### 📷 Camera Setup")

        use_dual = st.checkbox(
            "Enable Dual Camera (Phone + Laptop)",
            value=st.session_state.is_dual_mode,
            help="Use IP Webcam app on your phone as a body camera"
        )

        if use_dual:
            st.session_state.phone_url = st.text_input(
                "Phone Camera URL",
                value=st.session_state.phone_url or PHONE_CAM_URL,
                placeholder="http://192.168.x.x:8080/video",
                help=(
                    "Install 'IP Webcam' app on Android, "
                    "start server, and enter the URL here"
                )
            )
            st.info(
                "📱 **Setup:** Install 'IP Webcam' on Android → "
                "Start Server → Enter the IP address shown"
            )

        st.session_state.is_dual_mode = use_dual

        st.markdown("---")

        # Session Settings
        st.markdown("### ⏱️ Session")
        session_duration = st.slider(
            "Duration (seconds)",
            min_value=30,
            max_value=300,
            value=SESSION_DURATION_SECONDS,
            step=15
        )

        st.markdown("---")

        # System Status
        st.markdown("### 📊 Status")
        state = st.session_state.app_state
        if state == 'setup':
            st.info("🔧 Ready to configure")
        elif state == 'screening':
            st.success("🔴 Screening in progress")
            st.metric(
                "Frames",
                st.session_state.frame_count
            )
        elif state == 'results':
            st.success("✅ Results available")

        # Reset button
        if st.button("🔄 Reset Everything", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Clear evidence
            for f in EVIDENCE_DIR.glob("evidence_*.jpg"):
                f.unlink()
            st.rerun()

        return session_duration


# =============================================
# SETUP PAGE
# =============================================
def render_setup_page():
    """Render the initial setup and start page."""
    st.markdown(
        '<h1 class="main-header">🧠 NeuroLens AI</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        'AI-Powered Autism Spectrum Behavioral Screening Tool'
        '</p>',
        unsafe_allow_html=True
    )

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👁️ Gaze & Expression")
        st.markdown(
            "Real-time eye contact tracking, blink rate "
            "analysis, facial expression variance detection, "
            "and head pose estimation using 468 3D facial "
            "landmarks."
        )

    with col2:
        st.markdown("### 🏃 Body Movement")
        st.markdown(
            "Dual-camera body analysis detects repetitive "
            "behaviors including body rocking, hand flapping, "
            "and unusual stillness using 33-point pose "
            "estimation."
        )

    with col3:
        st.markdown("### 📄 Evidence Reports")
        st.markdown(
            "Generate clinical-grade PDF reports with "
            "timestamped visual evidence, explainable AI "
            "risk scores, and actionable recommendations."
        )

    st.markdown("---")

    # Quick start
    st.markdown("## 🚀 Start Screening Session")

    col_start1, col_start2 = st.columns([2, 1])

    with col_start1:
        st.markdown(
            """
            **Pre-Screening Checklist:**
            - ✅ Good lighting on the subject's face
            - ✅ Webcam positioned at eye level
            - ✅ Subject seated ~2 feet from camera
            - ✅ Minimal background movement
            """
        )
        if st.session_state.is_dual_mode:
            st.markdown(
                """
                **Dual Camera Setup:**
                - ✅ Phone running IP Webcam app
                - ✅ Phone positioned to capture full body (side view)
                - ✅ Both devices on same WiFi network
                """
            )

    with col_start2:
        st.markdown("")
        st.markdown("")
        if st.button(
            "▶️ START SCREENING",
            type="primary",
            use_container_width=True
        ):
            start_screening_session()


def start_screening_session():
    """Initialize all components and start screening."""
    with st.spinner("Initializing cameras and AI models..."):
        # Initialize camera system
        phone_url = (
            st.session_state.phone_url
            if st.session_state.is_dual_mode
            else None
        )
        camera_system = DualCameraSystem(
            laptop_source=LAPTOP_CAM_INDEX,
            phone_url=phone_url
        )
        status = camera_system.start()

        if not status['face_cam']:
            st.error(
                "❌ Failed to open laptop camera. "
                "Check permissions."
            )
            return

        if (
            st.session_state.is_dual_mode
            and not status['body_cam']
        ):
            st.warning(
                "⚠️ Phone camera not connected. "
                "Running in single-camera mode."
            )
            st.session_state.is_dual_mode = False

        # Initialize analyzers
        face_analyzer = FaceAnalyzer()
        body_analyzer = (
            BodyAnalyzer()
            if st.session_state.is_dual_mode
            else None
        )

        # Initialize risk engine
        start_time = time.time()
        risk_engine = RiskEngine(start_time)

        # Initialize chatbot
        chatbot = AutismScreeningChatbot()

        # Store in session state
        st.session_state.camera_system = camera_system
        st.session_state.face_analyzer = face_analyzer
        st.session_state.body_analyzer = body_analyzer
        st.session_state.risk_engine = risk_engine
        st.session_state.chatbot = chatbot
        st.session_state.session_start_time = start_time
        st.session_state.app_state = 'screening'
        st.session_state.screening_active = True
        st.session_state.frame_count = 0

    st.rerun()


# =============================================
# SCREENING PAGE
# =============================================
def render_screening_page(session_duration: int):
    """Render the live screening interface."""
    st.markdown(
        '<h2 style="text-align:center;">🔴 Live Screening</h2>',
        unsafe_allow_html=True
    )

    # Timer
    elapsed = time.time() - st.session_state.session_start_time
    remaining = max(0, session_duration - elapsed)

    timer_col1, timer_col2, timer_col3 = st.columns(3)
    with timer_col1:
        st.metric(
            "⏱️ Elapsed",
            f"{int(elapsed)}s"
        )
    with timer_col2:
        st.metric(
            "⏳ Remaining",
            f"{int(remaining)}s"
        )
    with timer_col3:
        st.metric(
            "📊 Frames",
            st.session_state.frame_count
        )

    # Stop button
    col_stop1, col_stop2, col_stop3 = st.columns([1, 1, 1])
    with col_stop2:
        if st.button(
            "⏹️ STOP SCREENING",
            type="primary",
            use_container_width=True
        ):
            stop_screening()
            return

    st.markdown("---")

    # Main content area
    if st.session_state.is_dual_mode:
        video_col, avatar_col = st.columns([3, 2])
    else:
        video_col, avatar_col = st.columns([3, 2])

    # Placeholders for dynamic content
    with video_col:
        st.markdown("### 📹 Camera Feed")
        video_placeholder = st.empty()
        face_metrics_placeholder = st.empty()

    with avatar_col:
        st.markdown("### 🧍 3D Digital Twin")
        avatar_placeholder = st.empty()
        body_metrics_placeholder = st.empty()

    # Evidence log
    st.markdown("### 📋 Live Evidence Log")
    evidence_placeholder = st.empty()

    # ---- PROCESSING LOOP ----
    camera = st.session_state.camera_system
    face_az = st.session_state.face_analyzer
    body_az = st.session_state.body_analyzer
    risk_eng = st.session_state.risk_engine

    # Auto-stop if duration exceeded
    if remaining <= 0:
        stop_screening()
        return

    # Get frames
    face_frame_data, body_frame_data = camera.get_frames()

    if face_frame_data.is_valid:
        raw_face_frame = face_frame_data.frame

        # Analyze face
        face_result = face_az.analyze_frame(raw_face_frame)
        st.session_state.last_face_result = face_result
        st.session_state.frame_count += 1

        # Process through risk engine
        if face_result.face_detected:
            risk_eng.process_face_result(
                face_result, raw_face_frame
            )

        # Display face feed
        with video_col:
            if face_result.annotated_frame is not None:
                display_frame = cv2.cvtColor(
                    face_result.annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

                # If dual mode, create side-by-side
                if (
                    body_frame_data is not None
                    and body_frame_data.is_valid
                ):
                    body_result = body_az.analyze_frame(
                        body_frame_data.frame
                    )
                    st.session_state.last_body_result = body_result

                    risk_eng.process_body_result(
                        body_result, body_frame_data.frame
                    )

                    if body_result.annotated_frame is not None:
                        combined = create_dual_view(
                            face_result.annotated_frame,
                            body_result.annotated_frame
                        )
                        display_frame = cv2.cvtColor(
                            combined, cv2.COLOR_BGR2RGB
                        )

                video_placeholder.image(
                    display_frame,
                    channels="RGB",
                    use_container_width=True
                )

            # Face metrics
            face_metrics_placeholder.markdown(
                f"""
                | Metric | Value |
                |--------|-------|
                | 👁️ Eye Contact | {'✅ Yes' if face_result.gaze.is_looking_at_camera else '❌ No'} |
                | 🔍 Gaze Direction | {face_result.gaze.gaze_direction} |
                | 😐 Expression | {face_result.emotion.expression_label} |
                | 💧 Blinks | {face_az.blink_total} |
                | 🔄 Head Yaw | {face_result.gaze.head_pose_yaw:.1f}° |
                """
            )

        # --- 3D Avatar ---
        with avatar_col:
            # Face 3D Avatar
            if (
                face_result.face_detected
                and face_result.landmarks_3d is not None
            ):
                face_avatar_fig = create_face_avatar_3d(
                    face_result.landmarks_3d,
                    gaze_direction=face_result.gaze.gaze_direction,
                    expression=face_result.emotion.expression_label
                )
                avatar_placeholder.plotly_chart(
                    face_avatar_fig,
                    use_container_width=True,
                    key=f"face_avatar_{st.session_state.frame_count}"
                )

            # Body 3D Avatar (if dual mode)
            if (
                st.session_state.is_dual_mode
                and st.session_state.last_body_result is not None
                and st.session_state.last_body_result.pose_detected
                and st.session_state.last_body_result.landmarks_3d
                is not None
            ):
                body_result = st.session_state.last_body_result
                movement_flags = {
                    'is_rocking': body_result.is_rocking,
                    'is_hand_flapping': body_result.is_hand_flapping,
                }
                body_avatar_fig = create_pose_avatar_3d(
                    body_result.landmarks_3d,
                    title="3D Body Avatar",
                    movement_flags=movement_flags
                )
                body_metrics_placeholder.plotly_chart(
                    body_avatar_fig,
                    use_container_width=True,
                    key=f"body_avatar_{st.session_state.frame_count}"
                )

                # Body metrics display
                st.markdown(
                    f"""
                    | Motor Metric | Status |
                    |-------------|--------|
                    | 🔄 Rocking | {'⚠️ DETECTED' if body_result.is_rocking else '✅ None'} |
                    | 👋 Flapping | {'⚠️ DETECTED' if body_result.is_hand_flapping else '✅ None'} |
                    | 📊 Repetitive Score | {body_result.repetitive_motion_score:.2f} |
                    | 🧘 Stillness | {body_result.stillness_score:.2f} |
                    """
                )

        # --- Live Evidence Log ---
        current_evidence = risk_eng.evidence
        if current_evidence:
            evidence_md = "| Time | Category | Description | Severity |\n"
            evidence_md += "|------|----------|-------------|----------|\n"

            for ev in current_evidence[-5:]:  # Show last 5
                sev_emoji = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(ev.severity, '⚪')

                # Truncate description for table
                short_desc = (
                    ev.description[:80] + "..."
                    if len(ev.description) > 80
                    else ev.description
                )

                evidence_md += (
                    f"| {ev.session_time_str} "
                    f"| {ev.category.upper()} "
                    f"| {short_desc} "
                    f"| {sev_emoji} {ev.severity.upper()} |\n"
                )

            evidence_placeholder.markdown(evidence_md)
        else:
            evidence_placeholder.info(
                "No behavioral markers flagged yet. "
                "Monitoring in progress..."
            )

    # Auto-refresh for continuous streaming
    time.sleep(0.05)  # ~20 FPS target
    st.rerun()


def stop_screening():
    """Stop the screening session and compute results."""
    with st.spinner("Computing risk assessment..."):
        # Stop cameras
        if st.session_state.camera_system is not None:
            st.session_state.camera_system.stop()

        # Get final stats
        face_stats = {}
        body_stats = None

        if st.session_state.face_analyzer is not None:
            face_stats = (
                st.session_state.face_analyzer.get_session_stats()
            )
            st.session_state.face_stats = face_stats

        if st.session_state.body_analyzer is not None:
            body_stats = (
                st.session_state.body_analyzer.get_session_stats()
            )
            st.session_state.body_stats = body_stats

        # Compute risk assessment
        if st.session_state.risk_engine is not None:
            assessment = st.session_state.risk_engine.compute_assessment(
                face_stats=face_stats,
                body_stats=body_stats
            )
            st.session_state.assessment = assessment

            # Generate PDF Report
            report_gen = ReportGenerator()
            session_info = {
                'session_id': f"SCR-{int(time.time())}",
                'duration': face_stats.get(
                    'session_duration_seconds', 0
                ),
                'camera_mode': (
                    'Dual (Face + Body)'
                    if st.session_state.is_dual_mode
                    else 'Single (Face)'
                ),
                'subject_id': st.session_state.subject_id,
            }

            report_path = report_gen.generate_report(
                assessment=assessment,
                session_info=session_info,
                face_stats=face_stats,
                body_stats=body_stats
            )
            st.session_state.report_path = report_path

            # Inject context into chatbot
            if st.session_state.chatbot is not None:
                st.session_state.chatbot.inject_session_context(
                    assessment_summary=assessment.summary,
                    domain_scores=assessment.domain_scores,
                    evidence_count=len(assessment.evidence_items)
                )

        st.session_state.app_state = 'results'
        st.session_state.screening_active = False

    st.rerun()


# =============================================
# RESULTS PAGE
# =============================================
def render_results_page():
    """Render the post-screening results dashboard."""
    st.markdown(
        '<h1 class="main-header">📊 Screening Results</h1>',
        unsafe_allow_html=True
    )

    assessment = st.session_state.assessment
    face_stats = st.session_state.face_stats
    body_stats = st.session_state.body_stats

    if assessment is None:
        st.error("No assessment data available.")
        return

    # ---- OVERALL RISK BANNER ----
    risk_class = {
        'Low': 'risk-low',
        'Moderate': 'risk-moderate',
        'High': 'risk-high',
        'Very High': 'risk-high'
    }.get(assessment.risk_level, 'risk-low')

    risk_emoji = {
        'Low': '🟢',
        'Moderate': '🟡',
        'High': '🔴',
        'Very High': '🔴'
    }.get(assessment.risk_level, '⚪')

    st.markdown(
        f"""
        <div class="{risk_class}">
            <h2 style="text-align:center; margin:0;">
                {risk_emoji} Overall Risk: {assessment.risk_level}
            </h2>
            <h3 style="text-align:center; margin:5px 0 0 0;">
                Score: {assessment.overall_risk_score}/100
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("")

    # ---- TABS FOR DIFFERENT VIEWS ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔍 Evidence Timeline",
        "📄 Download Report",
        "🧍 3D Review",
        "💬 AI Chat"
    ])

    # ========== TAB 1: DASHBOARD ==========
    with tab1:
        render_dashboard_tab(assessment, face_stats, body_stats)

    # ========== TAB 2: EVIDENCE TIMELINE ==========
    with tab2:
        render_evidence_tab(assessment)

    # ========== TAB 3: PDF REPORT ==========
    with tab3:
        render_report_tab()

    # ========== TAB 4: 3D REVIEW ==========
    with tab4:
        render_3d_review_tab()

    # ========== TAB 5: CHATBOT ==========
    with tab5:
        render_chatbot_tab()


def render_dashboard_tab(assessment, face_stats, body_stats):
    """Render the main dashboard with domain scores."""
    st.markdown("## Domain Scores")

    # Domain score cards
    cols = st.columns(4)
    domain_emojis = {
        'Social Attention': '👁️',
        'Facial Expression': '😐',
        'Motor Behavior': '🏃',
        'Physiological': '💓'
    }

    for i, (domain, score) in enumerate(
        assessment.domain_scores.items()
    ):
        with cols[i]:
            emoji = domain_emojis.get(domain, '📊')

            if score < 25:
                delta_color = "normal"
                level = "Typical"
            elif score < 50:
                delta_color = "off"
                level = "Borderline"
            elif score < 75:
                delta_color = "inverse"
                level = "Atypical"
            else:
                delta_color = "inverse"
                level = "High Risk"

            st.metric(
                label=f"{emoji} {domain}",
                value=f"{score:.0f}/100",
                delta=level,
                delta_color=delta_color
            )

    st.markdown("---")

    # Session Statistics
    st.markdown("## 📈 Session Statistics")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown("### Face Analysis")
        st.markdown(
            f"""
            | Metric | Value |
            |--------|-------|
            | Session Duration | {face_stats.get('session_duration_seconds', 0):.0f}s |
            | Total Blinks | {face_stats.get('total_blinks', 0)} |
            | Blinks/Minute | {face_stats.get('blinks_per_minute', 0):.1f} |
            | Gaze Away Time | {face_stats.get('total_gaze_away_seconds', 0):.1f}s |
            | Gaze Away % | {face_stats.get('gaze_away_percentage', 0):.1f}% |
            | Expression Variance | {face_stats.get('expression_variance', 0):.4f} |
            | Gaze Events (>3s) | {len(face_stats.get('gaze_events', []))} |
            """
        )

    with stat_col2:
        st.markdown("### Body Analysis")
        if body_stats:
            st.markdown(
                f"""
                | Metric | Value |
                |--------|-------|
                | Total Frames | {body_stats.get('total_frames_analyzed', 0)} |
                | Rocking % | {body_stats.get('rocking_percentage', 0):.1f}% |
                | Flapping % | {body_stats.get('flapping_percentage', 0):.1f}% |
                | Rocking Frames | {body_stats.get('rocking_detected_frames', 0)} |
                | Flapping Frames | {body_stats.get('flapping_detected_frames', 0)} |
                | Duration | {body_stats.get('session_duration', 0):.0f}s |
                """
            )
        else:
            st.info(
                "Body analysis was not active "
                "(single camera mode)."
            )

    st.markdown("---")

    # Recommendations
    st.markdown("## 💡 Recommendations")
    for i, rec in enumerate(assessment.recommendations):
        st.markdown(f"**{i+1}.** {rec}")

    # Summary
    st.markdown("---")
    st.markdown("## 📝 Summary")
    st.info(assessment.summary)


def render_evidence_tab(assessment):
    """Render the evidence timeline with screenshots."""
    st.markdown("## 🔍 Visual Evidence Timeline")
    st.markdown(
        "*Each entry represents a moment where the AI detected "
        "an atypical behavioral marker.*"
    )

    if not assessment.evidence_items:
        st.success(
            "✅ No significant behavioral markers were flagged "
            "during this session."
        )
        return

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    with filter_col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=list(set(
                e.category for e in assessment.evidence_items
            )),
            default=list(set(
                e.category for e in assessment.evidence_items
            ))
        )

    filtered_evidence = [
        e for e in assessment.evidence_items
        if e.severity in severity_filter
        and e.category in category_filter
    ]

    st.markdown(
        f"**Showing {len(filtered_evidence)} of "
        f"{len(assessment.evidence_items)} evidence items**"
    )
    st.markdown("---")

    # Render each evidence item
    for i, evidence in enumerate(filtered_evidence):
        severity_colors = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }
        sev_emoji = severity_colors.get(evidence.severity, '⚪')

        # Evidence card
        with st.container():
            ev_col1, ev_col2 = st.columns([3, 2])

            with ev_col1:
                st.markdown(
                    f"### {sev_emoji} "
                    f"[{evidence.session_time_str}] "
                    f"{evidence.category.upper()}"
                )
                st.markdown(
                    f"**Severity:** {evidence.severity.upper()} "
                    f"| **Confidence:** "
                    f"{evidence.confidence * 100:.0f}%"
                )
                st.markdown(evidence.description)

                if evidence.metric_name:
                    st.caption(
                        f"📏 Metric: `{evidence.metric_name}` = "
                        f"**{evidence.metric_value:.1f}** "
                        f"(threshold: {evidence.threshold_value:.1f})"
                    )

            with ev_col2:
                if (
                    evidence.screenshot_path
                    and Path(evidence.screenshot_path).exists()
                ):
                    screenshot = cv2.imread(
                        evidence.screenshot_path
                    )
                    if screenshot is not None:
                        screenshot_rgb = cv2.cvtColor(
                            screenshot, cv2.COLOR_BGR2RGB
                        )
                        st.image(
                            screenshot_rgb,
                            caption=(
                                f"Evidence @ "
                                f"{evidence.session_time_str}"
                            ),
                            use_container_width=True
                        )
                else:
                    st.markdown(
                        "*Screenshot not available*"
                    )

        st.markdown("---")


def render_report_tab():
    """Render the PDF report download section."""
    st.markdown("## 📄 Diagnostic Evidence Report")
    st.markdown(
        "Download the comprehensive PDF report containing "
        "all screening data, visual evidence, and "
        "recommendations."
    )

    report_path = st.session_state.report_path

    if report_path and Path(report_path).exists():
        st.success(
            f"✅ Report generated successfully!"
        )

        # Report preview info
        assessment = st.session_state.assessment
        st.markdown(
            f"""
            **Report Contents:**
            - 📋 Executive Summary
            - 📊 Domain Risk Scores (4 domains)
            - 🔍 Visual Evidence Timeline ({len(assessment.evidence_items)} items)
            - 💡 Clinical Recommendations ({len(assessment.recommendations)} items)
            - 🔬 Technical Methodology
            - ⚠️ Clinical Disclaimers
            """
        )

        # Download button
        with open(report_path, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=Path(report_path).name,
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.markdown("---")

        # Also show raw data
        with st.expander("🔧 Raw Session Data (JSON)"):
            raw_data = {
                "assessment": {
                    "risk_score": assessment.overall_risk_score,
                    "risk_level": assessment.risk_level,
                    "domain_scores": assessment.domain_scores,
                    "summary": assessment.summary,
                    "recommendations": assessment.recommendations,
                    "evidence_count": len(
                        assessment.evidence_items
                    )
                },
                "face_stats": st.session_state.face_stats,
                "body_stats": st.session_state.body_stats,
            }
            st.json(raw_data)

    else:
        st.error(
            "❌ Report file not found. Try running the "
            "screening session again."
        )


def render_3d_review_tab():
    """
    Render the 3D model review tab showing the
    last captured 3D avatars.
    """
    st.markdown("## 🧍 3D Digital Twin Review")
    st.markdown(
        "Interactive 3D models reconstructed from the "
        "screening session. Rotate and zoom to inspect "
        "posture and landmarks."
    )

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        st.markdown("### Face Mesh (3D)")
        last_face = st.session_state.last_face_result
        if (
            last_face is not None
            and last_face.face_detected
            and last_face.landmarks_3d is not None
        ):
            fig = create_face_avatar_3d(
                last_face.landmarks_3d,
                gaze_direction=last_face.gaze.gaze_direction,
                expression=last_face.emotion.expression_label
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="results_face_3d"
            )
        else:
            st.info("No face 3D data captured.")

    with model_col2:
        st.markdown("### Body Skeleton (3D)")
        last_body = st.session_state.last_body_result
        if (
            last_body is not None
            and last_body.pose_detected
            and last_body.landmarks_3d is not None
        ):
            movement_flags = {
                'is_rocking': last_body.is_rocking,
                'is_hand_flapping': last_body.is_hand_flapping,
            }
            fig = create_pose_avatar_3d(
                last_body.landmarks_3d,
                title="Last Captured Body Pose",
                movement_flags=movement_flags
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="results_body_3d"
            )
        else:
            st.info(
                "No body 3D data captured. "
                "(Enable dual camera mode for body tracking)"
            )

    st.markdown("---")
    st.markdown(
        """
        **About the Digital Twin:**
        - Face Mesh: 468 3D landmarks from MediaPipe Face Mesh 
          with iris tracking refinement
        - Body Skeleton: 33 3D landmarks from MediaPipe Pose
        - All coordinates are in normalized space (0-1)
        - Z-axis represents estimated depth relative to camera
        - Interactive: Click and drag to rotate, scroll to zoom
        """
    )


def render_chatbot_tab():
    """Render the AI chatbot interface."""
    st.markdown("## 💬 NeuroLens AI Assistant")
    st.markdown(
        "Ask questions about the screening results, autism "
        "spectrum disorder, next steps, or child development."
    )

    chatbot = st.session_state.chatbot

    if chatbot is None:
        st.error("Chatbot not initialized.")
        return

    if not chatbot.is_configured:
        st.warning(
            "⚠️ Gemini API key not configured. Using basic "
            "responses. Set the `GEMINI_API_KEY` environment "
            "variable for full AI chat capabilities."
        )

    # Chat history display
    chat_container = st.container()

    with chat_container:
        # Display existing messages
        for msg in st.session_state.chat_messages:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(content)

    # Chat input
    user_input = st.chat_input(
        "Ask about the screening results...",
        key="chat_input"
    )

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': user_input
        })

        # Get AI response
        with st.spinner("Thinking..."):
            response = chatbot.send_message(user_input)

        # Add assistant message
        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': response
        })

        st.rerun()

    # Quick question buttons
    st.markdown("---")
    st.markdown("**Quick Questions:**")

    quick_col1, quick_col2 = st.columns(2)

    with quick_col1:
        if st.button(
            "📊 Explain my results",
            use_container_width=True
        ):
            _send_quick_question(
                chatbot,
                "Can you explain what my screening results mean "
                "in simple terms?"
            )

        if st.button(
            "🏥 What are the next steps?",
            use_container_width=True
        ):
            _send_quick_question(
                chatbot,
                "Based on these results, what should my "
                "next steps be?"
            )

    with quick_col2:
        if st.button(
            "👁️ What does 'gaze avoidance' mean?",
            use_container_width=True
        ):
            _send_quick_question(
                chatbot,
                "What does gaze avoidance mean in the context "
                "of autism screening?"
            )

        if st.button(
            "🧒 Early signs of autism",
            use_container_width=True
        ):
            _send_quick_question(
                chatbot,
                "What are the early signs of autism in toddlers "
                "that parents should watch for?"
            )


def _send_quick_question(chatbot, question: str):
    """Handle quick question button clicks."""
    st.session_state.chat_messages.append({
        'role': 'user',
        'content': question
    })
    response = chatbot.send_message(question)
    st.session_state.chat_messages.append({
        'role': 'assistant',
        'content': response
    })
    st.rerun()


# =============================================
# MAIN APP ROUTER
# =============================================
def main():
    """Main application entry point and router."""
    session_duration = render_sidebar()

    if st.session_state.app_state == 'setup':
        render_setup_page()

    elif st.session_state.app_state == 'screening':
        render_screening_page(session_duration)

    elif st.session_state.app_state == 'results':
        render_results_page()


if __name__ == "__main__":
    main()