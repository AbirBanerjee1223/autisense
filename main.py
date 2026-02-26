# main.py (COMPLETE REWRITE)

import streamlit as st
import cv2
import numpy as np
import time
import base64
from pathlib import Path

from config import (
    LAPTOP_CAM_INDEX,
    SESSION_DURATION_SECONDS,
    EVIDENCE_DIR,
    NAME_CALL_AUDIO,
    BASELINES
)
from modules.face_analyzer import FaceAnalyzer
from modules.body_analyzer import BodyAnalyzer
from modules.stimulus_engine import StimulusEngine, StimulusPhase
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
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #2980b9, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .phase-banner {
        padding: 8px 16px; border-radius: 8px;
        font-weight: bold; text-align: center;
        margin-bottom: 10px;
    }
    .phase-baseline { background: #ecf0f1; color: #2c3e50; }
    .phase-social { background: #3498db; color: white; }
    .phase-namecall { background: #e67e22; color: white; }
    .phase-smile { background: #e91e63; color: white; }
    .phase-cooldown { background: #95a5a6; color: white; }
    .deviation-typical { 
        background: #d5f5e3; border-left: 4px solid #2ecc71;
        padding: 10px; margin: 5px 0; border-radius: 4px;
    }
    .deviation-borderline {
        background: #fef9e7; border-left: 4px solid #f1c40f;
        padding: 10px; margin: 5px 0; border-radius: 4px;
    }
    .deviation-atypical {
        background: #fdedec; border-left: 4px solid #e74c3c;
        padding: 10px; margin: 5px 0; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# SESSION STATE
# =============================================
def init_state():
    defaults = {
        'app_state': 'setup',
        'face_analyzer': None,
        'stimulus_engine': None,
        'risk_engine': None,
        'chatbot': None,
        'session_start': None,
        'face_stats': {},
        'assessment': None,
        'report_path': None,
        'chat_messages': [],
        'frame_count': 0,
        'subject_id': 'Anonymous',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =============================================
# AUDIO HELPER
# =============================================
def get_audio_html(audio_path: str) -> str:
    """Create auto-playing audio HTML from a wav file."""
    if not Path(audio_path).exists():
        return ""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """


# =============================================
# SIDEBAR
# =============================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        st.session_state.subject_id = st.text_input(
            "Subject ID", value=st.session_state.subject_id
        )

        st.markdown("---")
        st.markdown("### Protocol Phases")
        st.markdown("""
        1. **Baseline** (0-15s) - Natural behavior
        2. **Social/Geometric** (15-45s) - Visual preference
        3. **Name-Call** (45-55s) - Audio response
        4. **Smile Prompt** (55-75s) - Reciprocity
        5. **Cooldown** (75-90s) - Post-stimulus
        """)

        st.markdown("---")
        state = st.session_state.app_state
        if state == 'setup':
            st.info("🔧 Ready")
        elif state == 'screening':
            st.success("🔴 Screening active")
        elif state == 'results':
            st.success("✅ Results ready")

        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            for f in EVIDENCE_DIR.glob("evidence_*.jpg"):
                f.unlink()
            st.rerun()


# =============================================
# SETUP PAGE
# =============================================
def render_setup():
    st.markdown(
        '<h1 class="main-header">🧠 NeuroLens AI</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#7f8c8d;'>"
        "Structured Clinical Stimulus Behavioral Screening"
        "</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👁️ Social Preference")
        st.markdown(
            "Split-screen test measures visual preference "
            "for social vs. geometric stimuli "
            "(EarliPoint method)."
        )
    with col2:
        st.markdown("### 🔊 Name-Call Latency")
        st.markdown(
            "Audio stimulus measures orienting response "
            "time in milliseconds "
            "(SenseToKnow method)."
        )
    with col3:
        st.markdown("### 😊 Emotional Reciprocity")
        st.markdown(
            "Smile prompt measures social mirroring "
            "and affect synchrony."
        )

    st.markdown("---")

    # Check stimuli exist
    from config import SOCIAL_GEOMETRIC_VIDEO, SMILE_PROMPT_VIDEO
    stimuli_ready = (
        Path(SOCIAL_GEOMETRIC_VIDEO).exists() and
        Path(SMILE_PROMPT_VIDEO).exists() and
        Path(NAME_CALL_AUDIO).exists()
    )

    if not stimuli_ready:
        st.warning(
            "⚠️ Stimulus files not found. "
            "Run `python stimulus_creator.py` first."
        )
        if st.button("Generate Stimuli Now"):
            with st.spinner("Generating stimulus files..."):
                from stimulus_creator import create_all_stimuli
                create_all_stimuli()
            st.success("✅ Stimuli generated!")
            st.rerun()
        return

    st.success("✅ All stimulus files ready")

    if st.button(
        "▶️ START SCREENING PROTOCOL",
        type="primary",
        use_container_width=True
    ):
        start_session()


def start_session():
    """Initialize everything and begin."""
    with st.spinner("Initializing..."):
        face_analyzer = FaceAnalyzer()
        stimulus_engine = StimulusEngine()
        start_time = time.time()
        risk_engine = RiskEngine(start_time)
        chatbot = AutismScreeningChatbot()

        stimulus_engine.start_session()

        st.session_state.face_analyzer = face_analyzer
        st.session_state.stimulus_engine = stimulus_engine
        st.session_state.risk_engine = risk_engine
        st.session_state.chatbot = chatbot
        st.session_state.session_start = start_time
        st.session_state.app_state = 'screening'
        st.session_state.frame_count = 0

    st.rerun()


# =============================================
# SCREENING PAGE (NO FLICKERING)
# =============================================
# main.py - Replace render_screening() completely

def render_screening():
    st.markdown(
        "<h2 style='text-align:center;'>🔴 Live Screening</h2>",
        unsafe_allow_html=True
    )

    face_az = st.session_state.face_analyzer
    stim = st.session_state.stimulus_engine
    risk_eng = st.session_state.risk_engine

    # Stop button (OUTSIDE the loop)
    stop_col1, stop_col2, stop_col3 = st.columns([1, 1, 1])
    with stop_col2:
        stop_button = st.button(
            "⏹️ STOP SCREENING",
            type="primary",
            use_container_width=True
        )

    if stop_button:
        stop_session()
        return

    st.markdown("---")

    # ALL placeholders created ONCE, OUTSIDE the loop
    phase_placeholder = st.empty()
    timer_placeholder = st.empty()
    progress_placeholder = st.empty()

    st.markdown("---")

    # Two columns for camera and stimulus
    cam_col, stim_col = st.columns(2)
    with cam_col:
        st.markdown("### 📹 Subject Camera")
        video_placeholder = st.empty()
        metrics_placeholder = st.empty()

    with stim_col:
        st.markdown("### 🖥️ Stimulus Display")
        stimulus_placeholder = st.empty()
        stim_info_placeholder = st.empty()

    # Audio (hidden)
    audio_placeholder = st.empty()

    # Evidence
    st.markdown("### 📋 Live Evidence")
    evidence_placeholder = st.empty()

    # ---- CAMERA + MAIN LOOP ----
    cap = cv2.VideoCapture(LAPTOP_CAM_INDEX)
    if not cap.isOpened():
        st.error("❌ Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            elapsed = time.time() - st.session_state.session_start

            # Auto-stop
            if elapsed >= SESSION_DURATION_SECONDS:
                break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # ---- ANALYZE ----
            face_result = face_az.analyze_frame(frame)
            st.session_state.frame_count += 1

            if face_result.face_detected:
                risk_eng.process_face_result(face_result, frame)

            # Extract behavioral data for stimulus engine
            gaze_dir = (
                face_result.gaze.gaze_direction
                if face_result.face_detected else "unknown"
            )
            is_looking = (
                face_result.gaze.is_looking_at_camera
                if face_result.face_detected else False
            )
            head_yaw = (
                face_result.gaze.head_pose_yaw
                if face_result.face_detected else 0.0
            )
            smile_score = (
                face_result.emotion.smile_score
                if face_result.face_detected else 0.0
            )

            stim_result = stim.update(
                gaze_direction=gaze_dir,
                is_looking=is_looking,
                head_yaw=head_yaw,
                smile_score=smile_score,
            )

            # ---- UPDATE UI (all using .empty() placeholders) ----

            phase = stim_result["phase"]
            remaining = max(0, SESSION_DURATION_SECONDS - elapsed)

            # Phase banner
            phase_classes = {
                "baseline": ("phase-baseline", "🔬"),
                "social_geo": ("phase-social", "👁️"),
                "name_call": ("phase-namecall", "🔊"),
                "smile_prompt": ("phase-smile", "😊"),
                "cooldown": ("phase-cooldown", "⏸️"),
            }
            css_class, phase_icon = phase_classes.get(
                phase, ("phase-baseline", "📊")
            )
            phase_placeholder.markdown(
                f'<div class="phase-banner {css_class}">'
                f'{phase_icon} {stim_result["instruction"]}</div>',
                unsafe_allow_html=True
            )

            # Timer - single row using markdown table
            phase_display = phase.replace("_", " ").title()
            timer_placeholder.markdown(
                f"| ⏱️ Elapsed | ⏳ Remaining | 📊 Phase | 🎞️ Frames |\n"
                f"|:---:|:---:|:---:|:---:|\n"
                f"| **{int(elapsed)}s** | **{int(remaining)}s** | "
                f"**{phase_display}** | **{st.session_state.frame_count}** |"
            )

            # Progress bar
            progress_placeholder.progress(
                min(elapsed / SESSION_DURATION_SECONDS, 1.0),
                text=(
                    f"Protocol: {elapsed:.0f}/"
                    f"{SESSION_DURATION_SECONDS}s"
                )
            )

            # Camera feed
            if face_result.annotated_frame is not None:
                display = cv2.cvtColor(
                    face_result.annotated_frame, cv2.COLOR_BGR2RGB
                )
                video_placeholder.image(
                    display, channels="RGB",
                    use_container_width=True
                )

            # Face metrics
            if face_result.face_detected:
                contact = (
                    "✅ Yes"
                    if face_result.gaze.is_looking_at_camera
                    else "❌ No"
                )
                metrics_placeholder.markdown(
                    f"**Eye Contact:** {contact} | "
                    f"**Gaze:** {gaze_dir} | "
                    f"**Expr:** "
                    f"{face_result.emotion.expression_label} | "
                    f"**Blinks:** {face_az.blink_total} | "
                    f"**Smile:** {smile_score:.2f}"
                )
            else:
                metrics_placeholder.warning("No face detected")

            # Stimulus display
            stim_frame = stim_result.get("stimulus_frame")
            if stim_frame is not None:
                stim_rgb = cv2.cvtColor(
                    stim_frame, cv2.COLOR_BGR2RGB
                )
                stimulus_placeholder.image(
                    stim_rgb, channels="RGB",
                    use_container_width=True
                )

            # Stimulus-specific info
            if phase == "social_geo":
                sg = stim.social_geo_metrics
                stim_info_placeholder.markdown(
                    f"**Social:** "
                    f"{sg.social_preference_pct:.0f}% | "
                    f"**Geometric:** "
                    f"{sg.geometric_preference_pct:.0f}% | "
                    f"**Away:** {sg.gaze_away_frames} frames"
                )
            elif phase == "name_call":
                nc = stim.name_call_metrics
                if nc.responded:
                    stim_info_placeholder.success(
                        f"✅ Response! Latency: "
                        f"{nc.response_latency_ms:.0f}ms"
                    )
                elif stim._name_call_audio_played:
                    stim_info_placeholder.warning(
                        "⏳ Waiting for head-turn response..."
                    )
                else:
                    stim_info_placeholder.info(
                        "Preparing audio stimulus..."
                    )
            elif phase == "smile_prompt":
                if hasattr(stim, 'reciprocity_tracker'):
                    rc = stim.reciprocity_tracker.get_live_metrics()
                    info_text = (
                        f"**Smile-back:** "
                        f"{rc['smile_reciprocity_pct']:.0f}% | "
                        f"**Peak:** {rc['peak_intensity']:.2f} | "
                        f"**Episodes:** {rc['smile_episodes']}"
                    )
                    if rc['is_currently_smiling']:
                        info_text += (
                            f"\n\n😊 **SMILING NOW** "
                            f"({rc['current_smile_duration_ms']:.0f}ms)"
                        )
                    stim_info_placeholder.markdown(info_text)
                else:
                    rc_m = stim.reciprocity_metrics
                    stim_info_placeholder.markdown(
                        f"**Smile-back:** "
                        f"{rc_m.smile_reciprocity_pct:.0f}% | "
                        f"**Peak:** "
                        f"{rc_m.peak_smile_score:.2f}"
                    )
            elif phase == "cooldown":
                stim_info_placeholder.info(
                    f"Post-stimulus observation. "
                    f"Ending in {int(remaining)}s..."
                )
            elif phase == "baseline":
                stim_info_placeholder.info(
                    "Observing natural behavior (no stimulus)"
                )

            # Audio trigger
            if stim_result.get("play_audio"):
                audio_html = get_audio_html(NAME_CALL_AUDIO)
                if audio_html:
                    audio_placeholder.markdown(
                        audio_html, unsafe_allow_html=True
                    )

            # Evidence log
            if risk_eng.evidence:
                ev_lines = []
                for ev in risk_eng.evidence[-4:]:
                    sev_icon = {
                        'high': '🔴',
                        'medium': '🟡',
                        'low': '🟢'
                    }.get(ev.severity, '⚪')
                    short_desc = (
                        ev.description[:90] + "..."
                        if len(ev.description) > 90
                        else ev.description
                    )
                    ev_lines.append(
                        f"{sev_icon} **[{ev.session_time_str}]** "
                        f"{ev.category}: {short_desc}"
                    )
                evidence_placeholder.markdown(
                    "\n\n".join(ev_lines)
                )
            else:
                evidence_placeholder.info(
                    "No markers flagged yet. Monitoring..."
                )

            # Frame rate control (~20 fps)
            time.sleep(0.05)

    except Exception as e:
        st.error(f"Screening error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()

    # Session ended naturally (timer ran out)
    stop_session()


def stop_session():
    """Compute results and transition to results page."""
    with st.spinner("Computing clinical assessment..."):
        face_stats = {}
        if st.session_state.face_analyzer:
            face_stats = st.session_state.face_analyzer.get_session_stats()
            st.session_state.face_stats = face_stats

        stimulus_metrics = None
        if st.session_state.stimulus_engine:
            stimulus_metrics = (
                st.session_state.stimulus_engine.get_all_metrics()
            )
            st.session_state.stimulus_engine.cleanup()

        if st.session_state.risk_engine:
            assessment = st.session_state.risk_engine.compute_assessment(
                face_stats=face_stats,
                stimulus_metrics=stimulus_metrics,
            )
            st.session_state.assessment = assessment

            # Generate report
            gen = ReportGenerator()
            session_info = {
                'session_id': f"SCR-{int(time.time())}",
                'duration': face_stats.get(
                    'session_duration_seconds', 0
                ),
                'camera_mode': 'Single (Face + Stimulus Protocol)',
                'subject_id': st.session_state.subject_id,
            }
            report_path = gen.generate_report(
                assessment=assessment,
                session_info=session_info,
                face_stats=face_stats,
            )
            st.session_state.report_path = report_path

            # Inject context into chatbot
            if st.session_state.chatbot:
                st.session_state.chatbot.inject_session_context(
                    assessment_summary=assessment.summary,
                    domain_scores=assessment.domain_scores,
                    evidence_count=len(assessment.evidence_items)
                )

        if st.session_state.face_analyzer:
            st.session_state.face_analyzer.close()

        st.session_state.app_state = 'results'
    st.rerun()


# =============================================
# RESULTS PAGE
# =============================================
def render_results():
    st.markdown(
        '<h1 class="main-header">📊 Clinical Results</h1>',
        unsafe_allow_html=True
    )

    assessment = st.session_state.assessment
    if not assessment:
        st.error("No assessment data.")
        return

    # Risk level banner
    level_styles = {
        'Typical': ('deviation-typical', '🟢'),
        'Borderline': ('deviation-borderline', '🟡'),
        'Elevated': ('deviation-atypical', '🟠'),
        'High': ('deviation-atypical', '🔴'),
    }
    css, emoji = level_styles.get(
        assessment.risk_level, ('deviation-typical', '⚪')
    )
    st.markdown(
        f'<div class="{css}" style="text-align:center; font-size:1.3em;">'
        f'{emoji} Overall: <b>{assessment.risk_level}</b></div>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Deviations",
        "🔍 Evidence",
        "📄 Report",
        "💬 AI Chat"
    ])

    # ========== TAB 1: CLINICAL DEVIATIONS ==========
    with tab1:
        render_deviations_tab(assessment)

    # ========== TAB 2: EVIDENCE TIMELINE ==========
    with tab2:
        render_evidence_tab(assessment)

    # ========== TAB 3: PDF REPORT ==========
    with tab3:
        render_report_tab()

    # ========== TAB 4: CHATBOT ==========
    with tab4:
        render_chatbot_tab()


def render_deviations_tab(assessment):
    """Render clinical deviation analysis."""
    st.markdown("## Clinical Deviation Analysis")
    st.markdown(
        "*Each domain is measured against published neurotypical "
        "baselines. Deviations are expressed in Standard "
        "Deviations (SD).*"
    )
    st.markdown("---")

    if not hasattr(assessment, 'deviations') or not assessment.deviations:
        st.info("No deviation data available.")
        return

    # Summary metrics row
    atypical = [
        d for d in assessment.deviations
        if d.clinical_significance == "atypical"
    ]
    borderline = [
        d for d in assessment.deviations
        if d.clinical_significance == "borderline"
    ]
    typical = [
        d for d in assessment.deviations
        if d.clinical_significance == "typical"
    ]

    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric(
            "Total Domains",
            len(assessment.deviations)
        )
    with sum_col2:
        st.metric(
            "🟢 Typical",
            len(typical),
            help="Within 1 SD of baseline"
        )
    with sum_col3:
        st.metric(
            "🟡 Borderline",
            len(borderline),
            help="1-2 SD from baseline"
        )
    with sum_col4:
        st.metric(
            "🔴 Atypical",
            len(atypical),
            help=">2 SD from baseline"
        )

    st.markdown("---")

    # Individual deviation cards
    for dev in assessment.deviations:
        css_class = f"deviation-{dev.clinical_significance}"
        significance_emoji = {
            "typical": "🟢",
            "borderline": "🟡",
            "atypical": "🔴"
        }.get(dev.clinical_significance, "⚪")

        z_display = f"{dev.z_score:+.1f} SD"

        # Build the deviation card
        st.markdown(
            f'<div class="{css_class}">'
            f'<b>{significance_emoji} {dev.domain_name}</b> '
            f'&nbsp;|&nbsp; DSM-5: <code>{dev.dsm5_code}</code> '
            f'&nbsp;|&nbsp; Z-Score: <b>{z_display}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Expandable details
        with st.expander(
            f"Details: {dev.domain_name}", expanded=False
        ):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                # Value vs baseline
                if dev.metric_value == -1:
                    val_str = "No response"
                elif dev.metric_value < 1:
                    val_str = f"{dev.metric_value:.4f}"
                else:
                    val_str = f"{dev.metric_value:.1f}"

                st.markdown(f"**Measured Value:** {val_str}")
                st.markdown(
                    f"**Baseline (Neurotypical):** "
                    f"{dev.baseline_mean:.1f} ± {dev.baseline_std:.1f}"
                )
                st.markdown(f"**Z-Score:** {dev.z_score:+.2f} SD")
                st.markdown(
                    f"**Clinical Significance:** "
                    f"{dev.clinical_significance.upper()}"
                )

            with detail_col2:
                st.markdown(f"**Interpretation:**")
                st.markdown(dev.interpretation)

            # Visual z-score bar
            render_z_score_bar(dev.z_score, dev.domain_name)

        st.markdown("")

    # Recommendations
    st.markdown("---")
    st.markdown("## 💡 Clinical Recommendations")

    if assessment.recommendations:
        for i, rec in enumerate(assessment.recommendations):
            st.markdown(f"**{i + 1}.** {rec}")
    else:
        st.success(
            "No specific recommendations. All domains within "
            "typical range."
        )

    # Summary
    st.markdown("---")
    st.markdown("## 📝 Session Summary")
    st.info(assessment.summary)


def render_z_score_bar(z_score: float, label: str = ""):
    """
    Render a visual z-score bar using Streamlit columns.
    Shows position on a -3 to +3 SD scale.
    """
    # Clamp z-score for display
    z_clamped = max(-3.0, min(3.0, z_score))

    # Create a visual representation
    # Map z-score to 0-100 percentage (center = 50)
    position_pct = ((z_clamped + 3.0) / 6.0) * 100

    # Color based on significance
    z_abs = abs(z_score)
    if z_abs < 1.0:
        color = "#2ecc71"
        bg_zone = "Typical"
    elif z_abs < 2.0:
        color = "#f1c40f"
        bg_zone = "Borderline"
    else:
        color = "#e74c3c"
        bg_zone = "Atypical"

    # Build HTML bar
    bar_html = f"""
    <div style="
        position: relative;
        height: 30px;
        background: linear-gradient(to right, 
            #fdedec 0%, #fdedec 16.7%,
            #fef9e7 16.7%, #fef9e7 33.3%,
            #d5f5e3 33.3%, #d5f5e3 66.7%,
            #fef9e7 66.7%, #fef9e7 83.3%,
            #fdedec 83.3%, #fdedec 100%
        );
        border-radius: 4px;
        margin: 5px 0;
        border: 1px solid #ddd;
    ">
        <!-- Center line -->
        <div style="
            position: absolute; left: 50%; top: 0;
            width: 2px; height: 100%;
            background: #888;
        "></div>
        
        <!-- SD markers -->
        <div style="position:absolute; left:16.7%; top:0; width:1px; height:100%; background:#ccc;"></div>
        <div style="position:absolute; left:33.3%; top:0; width:1px; height:100%; background:#ccc;"></div>
        <div style="position:absolute; left:66.7%; top:0; width:1px; height:100%; background:#ccc;"></div>
        <div style="position:absolute; left:83.3%; top:0; width:1px; height:100%; background:#ccc;"></div>
        
        <!-- Marker -->
        <div style="
            position: absolute;
            left: {position_pct}%;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 18px; height: 18px;
            background: {color};
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        "></div>
        
        <!-- Labels -->
        <div style="position:absolute; left:2px; bottom:-16px; font-size:10px; color:#999;">-3σ</div>
        <div style="position:absolute; left:16%; bottom:-16px; font-size:10px; color:#999;">-2σ</div>
        <div style="position:absolute; left:32%; bottom:-16px; font-size:10px; color:#999;">-1σ</div>
        <div style="position:absolute; left:49%; bottom:-16px; font-size:10px; color:#999;">0</div>
        <div style="position:absolute; left:65%; bottom:-16px; font-size:10px; color:#999;">+1σ</div>
        <div style="position:absolute; left:82%; bottom:-16px; font-size:10px; color:#999;">+2σ</div>
        <div style="position:absolute; right:2px; bottom:-16px; font-size:10px; color:#999;">+3σ</div>
    </div>
    <div style="height: 20px;"></div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def render_evidence_tab(assessment):
    """Render evidence timeline with screenshots."""
    st.markdown("## 🔍 Visual Evidence Timeline")

    if not assessment.evidence_items:
        st.success(
            "✅ No behavioral markers were flagged "
            "during this session."
        )
        return

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        severity_filter = st.multiselect(
            "Severity",
            options=["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    with filter_col2:
        categories = list(set(
            e.category for e in assessment.evidence_items
        ))
        cat_filter = st.multiselect(
            "Category",
            options=categories,
            default=categories
        )

    filtered = [
        e for e in assessment.evidence_items
        if e.severity in severity_filter
        and e.category in cat_filter
    ]

    st.markdown(
        f"**Showing {len(filtered)} of "
        f"{len(assessment.evidence_items)} items**"
    )
    st.markdown("---")

    for evidence in filtered:
        sev_icons = {
            'high': '🔴', 'medium': '🟡', 'low': '🟢'
        }
        icon = sev_icons.get(evidence.severity, '⚪')

        with st.container():
            ev_col1, ev_col2 = st.columns([3, 2])

            with ev_col1:
                st.markdown(
                    f"### {icon} [{evidence.session_time_str}] "
                    f"{evidence.category.upper()}"
                )
                st.markdown(
                    f"**Severity:** {evidence.severity.upper()} | "
                    f"**Confidence:** "
                    f"{evidence.confidence * 100:.0f}%"
                )

                if evidence.z_score != 0:
                    st.markdown(
                        f"**Z-Score:** {evidence.z_score:+.1f} SD"
                    )

                st.markdown(evidence.description)

                if evidence.metric_name:
                    st.caption(
                        f"Metric: `{evidence.metric_name}` = "
                        f"**{evidence.metric_value:.1f}**"
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
                            caption=f"@ {evidence.session_time_str}",
                            use_container_width=True
                        )

        st.markdown("---")


def render_report_tab():
    """PDF report download section."""
    st.markdown("## 📄 Diagnostic Evidence Report")

    report_path = st.session_state.report_path
    assessment = st.session_state.assessment

    if report_path and Path(report_path).exists():
        st.success("✅ Report generated!")

        st.markdown(
            f"""
            **Report Contents:**
            - Clinical Deviation Analysis 
              ({len(assessment.deviations)} domains)
            - Visual Evidence Timeline 
              ({len(assessment.evidence_items)} items)
            - Recommendations 
              ({len(assessment.recommendations)} items)
            - DSM-5-TR Mapping
            - Technical Methodology with Citations
            """
        )

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

        # Raw data export
        with st.expander("🔧 Raw Session Data (JSON)"):
            import json

            # Build exportable data
            deviation_data = []
            for d in assessment.deviations:
                deviation_data.append({
                    "domain": d.domain_name,
                    "dsm5": d.dsm5_code,
                    "value": d.metric_value,
                    "baseline_mean": d.baseline_mean,
                    "baseline_std": d.baseline_std,
                    "z_score": d.z_score,
                    "significance": d.clinical_significance,
                    "interpretation": d.interpretation,
                })

            raw = {
                "risk_level": assessment.risk_level,
                "deviations": deviation_data,
                "recommendations": assessment.recommendations,
                "summary": assessment.summary,
                "face_stats": st.session_state.face_stats,
                "evidence_count": len(assessment.evidence_items),
            }
            st.json(raw)

            # Download JSON
            json_str = json.dumps(raw, indent=2)
            st.download_button(
                "📥 Download JSON Data",
                data=json_str,
                file_name="neurolens_session_data.json",
                mime="application/json"
            )

    else:
        st.error("❌ Report not found. Try screening again.")


def render_chatbot_tab():
    """AI chatbot interface."""
    st.markdown("## 💬 NeuroLens AI Assistant")
    st.markdown(
        "Ask about the screening results, autism, "
        "next steps, or child development."
    )

    chatbot = st.session_state.chatbot
    if not chatbot:
        st.error("Chatbot not initialized.")
        return

    if not chatbot.is_configured:
        st.warning(
            "⚠️ Gemini API not configured. "
            "Using basic responses. "
            "Set GEMINI_API_KEY in .env for full AI chat."
        )

    # Display chat history
    for msg in st.session_state.chat_messages:
        avatar = "👤" if msg['role'] == 'user' else "🧠"
        with st.chat_message(msg['role'], avatar=avatar):
            st.markdown(msg['content'])

    # Chat input
    user_input = st.chat_input(
        "Ask about the screening results..."
    )

    if user_input:
        # Add user message
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': user_input
        })

        # Get response
        with st.spinner("Thinking..."):
            response = chatbot.send_message(user_input)

        st.session_state.chat_messages.append({
            'role': 'assistant',
            'content': response
        })
        st.rerun()

    # Quick questions
    st.markdown("---")
    st.markdown("**Quick Questions:**")

    q_col1, q_col2 = st.columns(2)

    with q_col1:
        if st.button(
            "📊 Explain my results",
            use_container_width=True
        ):
            _quick_q(
                chatbot,
                "Explain my screening results in simple terms. "
                "What do the z-scores and standard deviations mean?"
            )

        if st.button(
            "🏥 What are the next steps?",
            use_container_width=True
        ):
            _quick_q(
                chatbot,
                "Based on these results, what should my next "
                "steps be? Who should I consult?"
            )

    with q_col2:
        if st.button(
            "🔊 What does name-call latency mean?",
            use_container_width=True
        ):
            _quick_q(
                chatbot,
                "What does the name-call response latency "
                "test measure and why is it important for "
                "autism screening?"
            )

        if st.button(
            "😊 Explain emotional reciprocity",
            use_container_width=True
        ):
            _quick_q(
                chatbot,
                "What is emotional reciprocity and why is "
                "the smile-back test important in autism "
                "screening?"
            )


def _quick_q(chatbot, question: str):
    """Handle quick question button."""
    st.session_state.chat_messages.append({
        'role': 'user', 'content': question
    })
    response = chatbot.send_message(question)
    st.session_state.chat_messages.append({
        'role': 'assistant', 'content': response
    })
    st.rerun()


# =============================================
# MAIN ROUTER
# =============================================
def main():
    render_sidebar()

    if st.session_state.app_state == 'setup':
        render_setup()
    elif st.session_state.app_state == 'screening':
        render_screening()
    elif st.session_state.app_state == 'results':
        render_results()


if __name__ == "__main__":
    main()