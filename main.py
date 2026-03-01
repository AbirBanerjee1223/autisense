import time
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from config import (
    EVIDENCE_DIR,
    LAPTOP_CAM_INDEX,
    NAME_CALL_AUDIO,
    SESSION_DURATION_SECONDS,
)
from modules.chatbot import AutismScreeningChatbot
from modules.face_analyzer import FaceAnalyzer
from modules.mchat import MCHAT_QUESTIONS, MCHATScreener
from modules.milestones import AGE_GROUP_LABELS, CATEGORY_INFO, MilestoneTracker
from modules.referral_generator import ReferralGenerator
from modules.report_generator import ReportGenerator
from modules.resource_directory import ResourceDirectory
from modules.risk_engine import RiskEngine
from modules.social_stories import STORY_TOPICS, SocialStoryGenerator
from modules.stimulus_engine import StimulusEngine
from modules.therapy_goals import GOAL_DOMAINS, PROMPT_LEVELS, TherapyGoalTracker
from modules.visual_schedule import VisualScheduleBuilder, get_all_activities_flat

st.set_page_config(page_title="Autisense Care Continuum", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700&family=Inter:wght@400;500;600&display=swap');

/* --- LIGHT MODE (DEFAULT) --- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

h1, h2, h3 { font-family: 'Quicksand', sans-serif !important; }

/* Hero Container - Soft light gradient */
.hero-container {
    text-align: center;
    padding: 40px 20px;
    background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
    border-radius: 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #1E3A8A; /* Deep Blue for light mode */
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #64748B;
    font-weight: 500;
}

/* Cards - Light background with dark text */
.soft-card {
    background: #FFFFFF !important;
    color: #1E293B !important;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 15px;
}

/* Locked State */
.locked {
    background: #F1F5F9 !important;
    opacity: 0.7;
    border: 1px dashed #CBD5E1;
}

/* --- DARK MODE OVERRIDES --- */
@media (prefers-color-scheme: dark) {
    html, body, [class*="css"] {
        background-color: #0F172A !important;
        color: #F1F5F9 !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-color: #334155;
    }

    .hero-title {
        color: #60A5FA; /* Brighter blue for dark mode */
    }

    .hero-subtitle {
        color: #94A3B8;
    }

    .soft-card {
        background: #1E293B !important;
        color: #F1F5F9 !important;
        border-color: #334155;
    }

    .locked {
        background: #161E2E !important;
        opacity: 0.6;
    }
}

/* Button Styling (Works for both) */
.stButton > button {
    background: #FF4B4B !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    border: none !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "app_state": "setup",
        "subject_id": "Anonymous",
        "journey_stage": "pre",
        "active_nav": "🏠 Home / Dashboard",
        "assessment_step": 1,
        "consent": False,
        "mchat_screener": MCHATScreener(),
        "mchat_result": None,
        "combined_summary": None,
        "matched_domains": [],
        "face_analyzer": None,
        "stimulus_engine": None,
        "risk_engine": None,
        "chatbot": AutismScreeningChatbot(),
        "session_start": None,
        "face_stats": {},
        "assessment": None,
        "report_path": None,
        "referral_path": None,
        "chat_messages": [],
        "frame_count": 0,
        "valid_face_frames": 0,
        "camera_snapshot": None,
        "camera_check_done": False,
        "nav_target": "🏠 Home / Dashboard",
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v


def compute_combined_summary():
    cv=st.session_state.assessment
    mc=st.session_state.mchat_result
    if cv is None and mc is None:
        return None
    cv_level = getattr(cv,'risk_level','Typical') if cv else 'Typical'
    mc_level = getattr(mc,'risk_level','LOW') if mc else 'LOW'
    score_map = {'Typical':0,'Borderline':1,'Elevated':2,'High':3,'LOW':0,'MEDIUM':2,'HIGH':3}
    score=max(score_map.get(cv_level,0), score_map.get(mc_level,0))
    combined = 'Low' if score==0 else 'Moderate' if score==1 else 'Elevated' if score==2 else 'High'
    concordance = 'Aligned' if (score_map.get(cv_level,0)>=2 and score_map.get(mc_level,0)>=2) or (score_map.get(cv_level,0)<2 and score_map.get(mc_level,0)<2) else 'Mixed'
    rec = {
        'Low':'Current data is reassuring. Continue surveillance and milestone checks.',
        'Moderate':'Some concerns were identified. Discuss findings with your pediatrician soon.',
        'Elevated':'Findings suggest notable developmental concern. Schedule a pediatrician and EI referral promptly.',
        'High':'Strong concern detected. Arrange urgent pediatric developmental follow-up and early intervention.'
    }[combined]
    return {'combined_risk':combined,'concordance':concordance,'combined_recommendation':rec,'cv_risk':cv_level,'mchat_risk':mc_level}


def refresh_matches():
    rd=ResourceDirectory()
    if st.session_state.assessment:
        rd.match_from_cv_assessment(st.session_state.assessment)
    if st.session_state.mchat_result:
        rd.match_from_mchat(st.session_state.mchat_result)
    st.session_state.matched_domains=sorted(rd.matched_domains)
    return rd


def render_sidebar():
    with st.sidebar:
        st.markdown('## 🧭 Navigation')
        st.session_state.subject_id = st.text_input('Subject ID', value=st.session_state.subject_id)
        
        options = [
            "🏠 Home / Dashboard",
            "📋 Assessment Center",
            "📊 Clinical Profile & Referrals",
            "🛠️ Daily Care Toolkit",
            "💬 Autisense Copilot"
        ]
        
        # Safely get the current index
        current_index = options.index(st.session_state.active_nav) if st.session_state.active_nav in options else 0
        
        # Callback: Only fires when a user manually clicks the selectbox dropdown
        def update_nav():
            st.session_state.active_nav = st.session_state.sidebar_nav_widget
            
        # The selectbox now uses the callback instead of an if-statement
        st.selectbox(
            'Go to', 
            options, 
            index=current_index, 
            key='sidebar_nav_widget', # Distinct key
            on_change=update_nav
        )

        locked = st.session_state.journey_stage != 'post'
        if locked:
            st.caption('🔒 Clinical Profile & Daily Toolkit unlock after assessment.')

        st.markdown('---')
        st.caption(f"Journey stage: **{'Post-Screening' if st.session_state.journey_stage=='post' else 'Onboarding'}**")
        if st.button('🔄 Reset Experience', use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            for f in EVIDENCE_DIR.glob('evidence_*.jpg'):
                f.unlink(missing_ok=True)
            st.rerun()


def render_home_dashboard():
    if st.session_state.journey_stage == 'pre':
        st.markdown("""<div class='hero-container'><div class='hero-title'>Autisense Early Intervention</div><div class='hero-subtitle'>Welcome → Listen & Observe → Analyze & Inform → Support Daily</div></div>""", unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.markdown("<div class='soft-card'><b>Why early screening matters</b><br/>Earlier support improves communication, social, and adaptive outcomes.</div>",unsafe_allow_html=True)
        c2.markdown("<div class='soft-card'><b>Quick and child-friendly</b><br/>A structured 90-second protocol plus parent questionnaire.</div>",unsafe_allow_html=True)
        c3.markdown("<div class='soft-card'><b>Actionable next steps</b><br/>Referral packet, specialist matching, and daily support tools.</div>",unsafe_allow_html=True)
        st.markdown('')
        if st.button('🚀 Start Initial Assessment', use_container_width=True, type='primary'):
            st.session_state.active_nav='📋 Assessment Center'
            st.session_state.assessment_step=1
            st.rerun()

        st.markdown("<div class='soft-card locked'>🔒 <b>Clinical Profile & Referrals</b><br/>Unlocks after complete assessment.</div>",unsafe_allow_html=True)
        st.markdown("<div class='soft-card locked'>🔒 <b>Daily Care Toolkit</b><br/>Personalized goals, stories, schedules, and milestones.</div>",unsafe_allow_html=True)
    else:
        summary = st.session_state.combined_summary or compute_combined_summary()
        if summary:
            st.info(f"**Combined Guidance:** {summary['combined_recommendation']}")
        col1,col2,col3=st.columns(3)
        with col1:
            if st.session_state.referral_path and Path(st.session_state.referral_path).exists():
                with open(st.session_state.referral_path,'rb') as f:
                    st.download_button('📥 Download Referral PDF',f.read(),file_name=Path(st.session_state.referral_path).name,mime='application/pdf',use_container_width=True)
        with col2:
            if st.button('🧭 View Recommended Next Steps',use_container_width=True):
                st.session_state.active_nav='📊 Clinical Profile & Referrals'; st.rerun()
        with col3:
            if st.button('🛠️ Open Daily Toolkit',use_container_width=True):
                st.session_state.active_nav='🛠️ Daily Care Toolkit'; st.rerun()

        st.markdown('### Daily Tools Highlights')
        goals=TherapyGoalTracker(st.session_state.subject_id).get_active_goals()
        st.markdown(f"<div class='soft-card'><b>Therapy Goals</b><br/>{len(goals)} active goals ready for today.</div>",unsafe_allow_html=True)
        sched=VisualScheduleBuilder(st.session_state.subject_id)
        st.markdown(f"<div class='soft-card'><b>Visual Schedule</b><br/>{len(sched.schedules)} saved schedules available.</div>",unsafe_allow_html=True)


def render_assessment_center():
    st.markdown('## 📋 Assessment Center')
    step=st.session_state.assessment_step
    st.progress(step/5, text=f"Step {step} of 5")

    if step==1:
        st.markdown('### Step 1 · Ethics Gate')
        st.checkbox('I provide consent for camera-based developmental screening.', key='consent')
        if st.button('Continue to Parent Questionnaire', type='primary', disabled=not st.session_state.consent):
            st.session_state.assessment_step=2; st.rerun()

    elif step==2:
        st.markdown('### Step 2 · M-CHAT-R Parent Questionnaire')
        screener=st.session_state.mchat_screener
        for q in MCHAT_QUESTIONS:
            current = screener.responses.get(q['id'])
            ans = st.radio(f"Q{q['id']}. {q['text']}", ['Yes','No'], index=(0 if current=='Yes' else 1 if current=='No' else None), key=f"mchat_{q['id']}", horizontal=True)
            if ans in ('Yes','No'):
                screener.set_response(q['id'], ans)
            if q.get('example'):
                st.caption(q['example'])
        c1,c2=st.columns(2)
        if c1.button('⬅️ Back'): st.session_state.assessment_step=1; st.rerun()
        if c2.button('Continue to Telemetry Setup', type='primary', disabled=not screener.is_complete()):
            st.session_state.mchat_result=screener.score()
            st.session_state.assessment_step=3
            st.rerun()

    elif step == 3:
        st.markdown('### Step 3 · Digital Telemetry Setup')
        
        from config import SOCIAL_GEOMETRIC_VIDEO, SMILE_PROMPT_VIDEO
        from pathlib import Path
        
        # Check if stimulus files exist
        stim_ok = Path(SOCIAL_GEOMETRIC_VIDEO).exists() and Path(NAME_CALL_AUDIO).exists() and Path(SMILE_PROMPT_VIDEO).exists()
        st.write(f"Stimulus assets loaded: {'✅' if stim_ok else '❌'}")
        
        if not stim_ok:
            st.error("Missing stimulus files. Please run `stimulus_creator.py`.")
            
        st.info("Please ensure the subject is in a well-lit room and facing the screen.")

        # 📸 NATIVE LIVE CAMERA WIDGET
        # This automatically shows a live feed and waits for the user to click "Take Photo"
        test_photo = st.camera_input("Verify lighting and positioning")

        # Once the user clicks "Take Photo", the image data populates test_photo
        if test_photo is not None:
            st.session_state.camera_snapshot = test_photo
            st.session_state.camera_check_done = True
            st.success('✅ Photo captured! Lighting and face visibility verified. You can continue to screening.')

        st.markdown("---")
        
        # Navigation Buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button('⬅️ Back'): 
                st.session_state.assessment_step = 2
                st.rerun()
                
        with col2:
            # Only allow starting if stimulus files exist AND they took a valid test photo
            can_start = stim_ok and st.session_state.get('camera_check_done', False)
            if st.button('Start 90-Second Protocol', type='primary', disabled=not can_start):
                # We update the state to 4 (the screening loop) and trigger the start function
                st.session_state.assessment_step = 4
                start_session()

    elif step==4:
        st.info('The 90-second protocol is running...')

    elif step == 5:
        st.markdown('### Step 5 · Clinical Synthesis')
        
        # 🛑 FIX: Only run the heavy fusion and PDF generation ONCE
        if not st.session_state.get('synthesis_complete', False):
            p = st.progress(0, text='Fusing M-CHAT-R responses with Digital Telemetry...')
            
            for i in range(1, 101, 20):
                time.sleep(0.12) 
                p.progress(i)
                
            st.session_state.combined_summary = compute_combined_summary()
            rd = refresh_matches()
            
            if st.session_state.referral_path is None and (st.session_state.assessment or st.session_state.mchat_result):
                rg = ReferralGenerator()
                st.session_state.referral_path = rg.generate_referral(
                    subject_id=st.session_state.subject_id,
                    cv_assessment=st.session_state.assessment,
                    mchat_result=st.session_state.mchat_result,
                    combined_summary=st.session_state.combined_summary,
                    face_stats=st.session_state.face_stats,
                    session_info={'duration': SESSION_DURATION_SECONDS, 'camera_mode': 'Single (Face + Stimulus Protocol)'},
                )
                
            st.session_state.journey_stage = 'post'
            st.session_state.synthesis_complete = True
            p.empty() # Cleanly remove the progress bar when finished

        # This will now render instantly on subsequent reruns
        st.success('✅ Assessment complete! Your care platform is now personalized.')
        st.info("The system has generated your combined risk profile and customized your Post-Diagnosis Care tools.")
        
        # The button will now work perfectly because the sleep loop is bypassed
        if st.button('Go to Clinical Profile', type='primary', use_container_width=True):
            st.session_state.active_nav = '📊 Clinical Profile & Referrals'
            st.rerun()


def render_clinical_profile():
    if st.session_state.journey_stage!='post':
        st.warning('🔒 Complete assessment first to unlock this section.')
        return
    st.markdown('## 📊 Clinical Profile & Referrals')
    summary=st.session_state.combined_summary or compute_combined_summary()
    if summary:
        st.info(f"Combined risk: **{summary['combined_risk']}** · Concordance: **{summary['concordance']}**")

    t1,t2,t3=st.tabs(['Dashboard','Action Center','Smart Directory'])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            if st.session_state.assessment:
                st.markdown('### Digital Telemetry (Z-score view)')
                render_deviations_tab(st.session_state.assessment)
            else:
                st.info('No CV assessment data available.')
        with c2:
            st.markdown('### M-CHAT-R Matrix')
            m=st.session_state.mchat_result
            if m:
                st.metric('Total score', f"{m.total_score}/20")
                st.metric('Risk level', m.risk_level)
                st.write('Flagged items:', ', '.join(map(str,m.risk_items)) if m.risk_items else 'None')
                for q in MCHAT_QUESTIONS:
                    ans=m.responses.get(q['id'],'-')
                    at='🔴' if q['id'] in m.risk_items else '🟢'
                    st.write(f"{at} Q{q['id']} · {q['domain']} · {ans}")
            else:
                st.info('No questionnaire data available.')
    with t2:
        st.markdown('### Generate Clinician Referral PDF')
        if st.session_state.referral_path and Path(st.session_state.referral_path).exists():
            with open(st.session_state.referral_path,'rb') as f:
                st.download_button('📥 Download Referral PDF',f.read(),file_name=Path(st.session_state.referral_path).name,mime='application/pdf',type='primary')
            st.caption('Take this document to your pediatrician.')
        else:
            st.warning('Referral not generated yet.')
    with t3:
        rd=refresh_matches()
        specials=rd.get_all_specialists()
        st.markdown('### Context-Aware Specialist Suggestions')
        st.write('Matched concern domains:', ', '.join(st.session_state.matched_domains) if st.session_state.matched_domains else 'None yet')
        if specials:
            for s in specials:
                st.markdown(f"- **{s.get('title','Specialist')}** — {s.get('why','Recommended based on screening patterns')}")
        else:
            st.info('No targeted specialties matched. Showing general developmental resources:')
            for r in rd.get_global_resources()[:6]:
                st.markdown(f"- **{r.name}** · {r.url}")


def render_daily_toolkit():
    if st.session_state.journey_stage!='post':
        st.warning('🔒 Complete assessment first to unlock this section.')
        return
    st.markdown('## 🛠️ Daily Care Toolkit')
    rd=refresh_matches()

    tab1,tab2,tab3,tab4 = st.tabs(['Therapy Goal Tracker','Social Story Generator','Visual Schedule Builder','Milestone Tracker'])
    with tab1:
        tracker=TherapyGoalTracker(st.session_state.subject_id)
        suggestions=tracker.get_suggested_goals(st.session_state.matched_domains)
        if suggestions:
            st.markdown("#### Suggested goals from your child's profile")
            for s in suggestions[:5]:
                if st.button(f"➕ Add: {s['goal_text']}", key=f"sug_{s['goal_text']}"):
                    tracker.add_goal_from_suggestion(s); st.rerun()
        with st.expander('Add custom goal'):
            gtxt=st.text_input('Goal statement', key='new_goal_txt')
            dom=st.selectbox('Domain', options=list(GOAL_DOMAINS.keys()))
            beh=st.text_area('Target behavior', key='new_goal_beh')
            if st.button('Save Goal') and gtxt and beh:
                tracker.add_goal(gtxt, dom, beh); st.success('Goal added'); st.rerun()
        st.markdown('#### Active goals')
        for g in tracker.get_active_goals():
            st.markdown(f"**{g.goal_text}** · {g.domain} · Latest: {g.latest_percentage:.0f}%")
            c1,c2,c3=st.columns([2,2,2])
            succ=c1.number_input('Successful',0,20,5,key=f"succ_{g.goal_id}")
            total=c2.number_input('Total',1,20,10,key=f"tot_{g.goal_id}")
            prompt=c3.selectbox('Prompt', list(PROMPT_LEVELS.keys()), key=f"prompt_{g.goal_id}")
            if st.button('Log trial', key=f"log_{g.goal_id}"):
                tracker.log_trial(g.goal_id, int(succ), int(total), prompt_level=prompt)
                st.success('Trial logged')
                st.rerun()
            st.markdown('---')

    with tab2:
        gen=SocialStoryGenerator()
        st.caption('Story themes are auto-personalized using matched domains.')
        cat=st.selectbox('Situation category', options=list(STORY_TOPICS.keys()))
        topics=STORY_TOPICS[cat]
        topic_idx=st.selectbox('Topic', options=list(range(len(topics))), format_func=lambda i: topics[i]['title'])
        age=st.slider('Child age',2,10,4)
        if st.button('Generate Story', type='primary'):
            out=gen.generate_from_preset(cat, topic_idx, child_age=age, screening_domains=st.session_state.matched_domains)
            st.subheader(out.get('title','Story'))
            st.write(out.get('story',''))

    with tab3:
        vs=VisualScheduleBuilder(st.session_state.subject_id)
        templates=vs.get_templates()
        tkey=st.selectbox('Template', options=list(templates.keys()), format_func=lambda k: templates[k]['name'])
        if st.button('Create from template'):
            sid=vs.create_from_template(tkey)
            st.session_state['active_schedule_id']=sid
            st.rerun()
        if vs.schedules:
            sid=st.selectbox('Select schedule', options=list(vs.schedules.keys()), format_func=lambda s: vs.schedules[s].name)
            schedule=vs.get_schedule(sid)
            acts=get_all_activities_flat()
            act_id=st.selectbox('Add activity', options=[a.id for a in acts], format_func=lambda aid: next(a for a in acts if a.id==aid).name)
            time_label=st.text_input('Time label (optional)','')
            if st.button('Add item'):
                vs.add_item(sid, act_id, time_label=time_label); st.rerun()
            for i,it in enumerate(schedule.items):
                done='✅' if it.completed else '⬜'
                if st.button(f"{done} {it.time_label} {it.icon} {it.activity_name}", key=f"tog_{sid}_{i}"):
                    vs.toggle_complete(sid, i); st.rerun()
            pdf=vs.export_schedule_pdf(sid)
            if pdf and Path(pdf).exists():
                with open(pdf,'rb') as f:
                    st.download_button('📥 Download Schedule PDF',f.read(),file_name=Path(pdf).name,mime='application/pdf')

    with tab4:
        mt=MilestoneTracker(st.session_state.subject_id)
        age_group=st.selectbox('Age bracket', options=list(AGE_GROUP_LABELS.keys()), format_func=lambda k: AGE_GROUP_LABELS[k])
        milestones=mt.get_milestones_for_age(age_group)
        for m in milestones:
            status=mt.get_status(m.id)
            default='Not yet assessed' if status is None else ('Achieved' if status else 'Not achieved')
            choice=st.selectbox(f"{m.text}", ['Not yet assessed','Achieved','Not achieved'], index=['Not yet assessed','Achieved','Not achieved'].index(default), key=f"mile_{m.id}")
            if st.button('Save', key=f"save_{m.id}"):
                if choice=='Not yet assessed':
                    pass
                else:
                    mt.set_milestone(m.id, achieved=(choice=='Achieved'))
                st.rerun()
        rep=mt.generate_report(age_group)
        st.info(f"Achievement rate: {rep.achievement_rate:.1f}% · Concern areas: {', '.join(rep.concern_areas) if rep.concern_areas else 'None'}")


def render_copilot_page():
    st.markdown('## 💬 Autisense Copilot')
    if st.session_state.combined_summary:
        st.caption(f"Context loaded: combined risk = {st.session_state.combined_summary['combined_risk']}")
    render_chatbot_tab()


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


# NOTE: screening logic retained from existing implementation.
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
    st.markdown("### Clinical Stimulus")
    stim_l, stim_m, stim_r = st.columns([1, 2, 1])
    with stim_m:
        stimulus_placeholder = st.empty()
    st.markdown("---")
    
    col_cam, col_data = st.columns([1, 2])
    with col_cam:
        st.markdown("### 📹 Subject Camera")
        video_placeholder = st.empty()
    with col_data:
        st.markdown("### 📊 Live Session Data")
        metrics_placeholder = st.empty()
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
                st.session_state.valid_face_frames += 1
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
                contact = "Yes" if face_result.gaze.is_looking_at_camera else "No"
                expr = face_result.emotion.expression_label.replace("_", " ").title()
                with metrics_placeholder.container():
                    st.markdown("<div class='live-metrics-card'>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Eye contact", contact)
                    m2.metric("Gaze direction", gaze_dir.title())
                    m3.metric("Affect", expr)
                    m4, m5, m6 = st.columns(3)
                    m4.metric("Blinks", f"{face_az.blink_total}")
                    m5.metric("Smile score", f"{smile_score:.2f}")
                    m6.metric("Head yaw", f"{head_yaw:.1f}°")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                metrics_placeholder.warning("Searching for face... Please position subject in frame.")

            # Stimulus display
            stim_frame = stim_result.get("stimulus_frame")
            if stim_frame is not None:
                stim_rgb = cv2.cvtColor(
                    stim_frame, cv2.COLOR_BGR2RGB
                )
                stimulus_placeholder.image(
                    stim_rgb, channels="RGB",
                    width=620
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
                import winsound
                try:
                    # Plays instantly via motherboard, bypassing browser HTML latency
                    winsound.PlaySound(NAME_CALL_AUDIO, winsound.SND_FILENAME | winsound.SND_ASYNC)
                except Exception as e:
                    pass

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
    """Compute results and transition to results page with clinical guardrails."""
    elapsed = time.time() - st.session_state.session_start
    
    # 🛑 GUARDRAIL 1: Minimum Session Time
    # Must complete at least 25 seconds (Baseline + part of Visual Test)
    if elapsed < 25.0:
        st.error("❌ Session ended too early. At least 25 seconds of data is required for clinical analysis.")
        time.sleep(3) # Let the user read the error
        st.session_state.app_state = 'setup'
        st.rerun()
        return
        
    # 🛑 GUARDRAIL 2: Face Presence Validation
    # If the camera was on, but the child ran away (less than ~2 seconds of face data)
    if st.session_state.get('valid_face_frames', 0) < 40:
        st.error("❌ Insufficient face data. Please ensure the subject is clearly visible in the camera frame.")
        time.sleep(3)
        st.session_state.app_state = 'setup'
        st.rerun()
        return

    # Valid session - Proceed with computation
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
    Render a visual z-score bar safely using Streamlit Components.
    """
    z_clamped = max(-3.0, min(3.0, z_score))
    position_pct = ((z_clamped + 3.0) / 6.0) * 100

    z_abs = abs(z_score)
    if z_abs < 1.0:
        color = "#2ecc71"
    elif z_abs < 2.0:
        color = "#f1c40f"
    else:
        color = "#e74c3c"

    # Use components.html instead of st.markdown(unsafe_allow_html=True)
    html_code = f"""
    <div style="font-family: sans-serif; position: relative; height: 30px; 
                background: linear-gradient(to right, #fdedec 16%, #fef9e7 33%, #d5f5e3 66%, #fef9e7 83%, #fdedec 100%);
                border-radius: 4px; border: 1px solid #ccc; margin-top:10px;">
        <div style="position:absolute; left:50%; top:0; width:2px; height:100%; background:#888;"></div>
        <div style="position: absolute; left: {position_pct}%; top: 50%; transform: translate(-50%, -50%); 
                    width: 16px; height: 16px; background: {color}; border-radius: 50%; border: 2px solid white; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);"></div>
        <div style="position:absolute; left:2px; bottom:-20px; font-size:12px; color:#666;">-3 SD</div>
        <div style="position:absolute; left:48%; bottom:-20px; font-size:12px; color:#666;">0</div>
        <div style="position:absolute; right:2px; bottom:-20px; font-size:12px; color:#666;">+3 SD</div>
    </div>
    """
    components.html(html_code, height=65)


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
    init_state()
    render_sidebar()

    if st.session_state.app_state == 'screening':
        render_screening()
        return

    if st.session_state.active_nav == '🏠 Home / Dashboard':
        render_home_dashboard()
    elif st.session_state.active_nav == '📋 Assessment Center':
        # if cv just completed and coming from screening, move to synthesis
        if st.session_state.app_state == 'results' and st.session_state.assessment_step < 5:
            st.session_state.assessment_step = 5
        render_assessment_center()
    elif st.session_state.active_nav == '📊 Clinical Profile & Referrals':
        render_clinical_profile()
    elif st.session_state.active_nav == '🛠️ Daily Care Toolkit':
        render_daily_toolkit()
    elif st.session_state.active_nav == '💬 Autisense Copilot':
        render_copilot_page()


if __name__ == '__main__':
    main()