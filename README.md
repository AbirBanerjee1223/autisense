# 🧠 Autisense: Agentic Cognitive Ecosystem for Early Intervention

**Autisense** is an AI-enabled, edge-computed platform designed to bridge the massive gap in early Autism Spectrum Disorder (ASD) care. It provides an end-to-end care continuum—moving seamlessly from multimodal early screening to diagnostic support and daily post-diagnosis intervention.

Instead of replacing clinicians, Autisense empowers them with quantified, objective biometric data, while simultaneously providing parents with an actionable, day-to-day care toolkit.

---

## 🌟 The Care Continuum (Core Features)

### Phase 1: Multimodal Clinical Screening

Autisense fuses traditional parent reporting with state-of-the-art computer vision to create a robust risk matrix.

* **90-Second Digital Telemetry Protocol:** A hardware-free active stimulus test using the device's standard webcam. Processes 100% on the edge via the MediaPipe Tasks API.
* **Visual Preference (15-45s):** Split-screen Social vs. Geometric tracking.
* **Auditory Latency (45-55s):** Zero-latency Name-Call tracking measuring precise head-yaw orientation response times.
* **Social Reciprocity (55-75s):** Smile-prompt mirroring using facial blendshapes.
* **Motor Stereotypy Tracking:** Digital Signal Processing (DSP) to detect body rocking (zero-crossing Hz) and hand-flapping.


* **M-CHAT-R/F Digitization:** The global gold-standard parent questionnaire built natively into the assessment flow.

### Phase 2: Diagnostic Support & Triage

Outputs are mapped directly to **DSM-5-TR** criteria and converted into Z-Score Standard Deviations against published neurotypical baselines.

* **Clinician Referral PDF:** Auto-generates structured evidence packets for pediatricians, detailing exact Z-score deviations and M-CHAT results to accelerate the diagnostic pipeline.
* **Context-Aware Resource Directory:** Dynamically filters local specialists (SLPs, BCBAs, OTs) based *only* on the child's specific flagged DSM-5 domains.
* **NeuroLens AI Copilot:** A RAG-powered Gemini assistant securely injected with the child's specific session context to answer parent questions empathetically and accurately.

### Phase 3: Post-Diagnosis Daily Care Toolkit

The platform transforms into a daily companion to support early intervention therapies (like ABA, ESDM, and TEACCH).

* **Agentic Therapy Goal Tracker:** ABA-standard data collection tracking prompt hierarchies, discrete trials, and mastery criteria. Auto-suggests goals based on screening deficits.
* **Social Story Generator:** On-demand, AI-generated behavioral narratives strictly adhering to Carol Gray’s 10.2 Criteria (maintaining the correct ratio of descriptive to directive sentences).
* **Visual Schedule Builder:** TEACCH-method visual routines using emoji-based cards. Fully exportable to PDF for physical printing.
* **Milestone Tracker:** Digitized tracking using the *CDC's 2022 "Learn the Signs. Act Early."* evidence-informed guidelines.

---

## 🔬 Clinical Methodology & Scientific Backing

Autisense does not invent new medicine; it scales proven clinical science using AI.

* **Gaze & Latency Algorithms:** Modeled on *Jones & Klin (2013)* for visual preference and *Ozonoff et al. (2010)* for early behavioral markers.
* **Scoring Architecture:** Z-Scores clamped to ±4.0 SD to ensure clinical plausibility.
* **Privacy by Design:** 100% Edge-computed computer vision. **No video data ever leaves the local device.** Requires explicit parental consent gate prior to screening.

---

## 🏗️ System Architecture & Tech Stack

* **Frontend:** Streamlit (Custom "Soft Clinical" Pediatric UI/CSS)
* **Computer Vision:** OpenCV, MediaPipe Tasks API (`FaceLandmarker`, `PoseLandmarker`)
* **Math & DSP:** NumPy, SciPy (Signal processing for stereotypy frequencies)
* **Generative AI:** Google Gemini API (`gemini-2.0-flash`) with Multi-Key Routing.
* **Reporting:** FPDF (Generates PDFs strictly in-memory to prevent disk clutter)
* **Audio:** `winsound` (Windows) for zero-latency auditory stimulus triggers.

---

## ⚙️ Installation & Setup

### 1. Prerequisites

* Python 3.9 to 3.11
* A working webcam

### 2. Clone and Install

```bash
git clone https://github.com/your-username/autisense.git
cd autisense
pip install -r requirements.txt

```

### 3. Generate Stimulus & Model Assets

Autisense requires specific audio/video files and ML models to run the screening protocol.

```bash
# Downloads the required MediaPipe .task files to models/
python model_downloader.py

# Generates the synthetic video and audio stimuli to stimuli/
python stimulus_creator.py 

```

### 4. Configure Environment Variables

Autisense utilizes a **Multi-Key LLM Routing Strategy** to prevent `429 Quota Exceeded` errors during heavy multi-turn usage. Create a `.env` file in the root directory and add three separate Google Gemini API keys:

```env
# .env
GEMINI_API_KEY_CHAT="your_first_gemini_key_here"      # Powers the Autisense Copilot
GEMINI_API_KEY_STORY="your_second_gemini_key_here"    # Powers the Social Story Generator
GEMINI_API_KEY_THERAPY="your_third_gemini_key_here"   # Powers the Therapy Goal suggestions

```

### 5. Run the Application

```bash
streamlit run main.py

```

---

## 📂 Repository Structure

```text
autisense/
├── main.py                     # Primary Streamlit router and clinical UI
├── config.py                   # Global constants, paths, and neurotypical baselines
├── requirements.txt            # Python dependencies
├── .env                        # API Keys (Git-ignored)
├── models/                     # MediaPipe tasks (FaceLandmarker, PoseLandmarker)
├── stimuli/                    # Audio/Video assets for the 90-second protocol
├── reports/                    # Generated PDFs (Referrals, Schedules)
└── modules/
    ├── face_analyzer.py        # CV: Head pose, Gaze, Blendshapes (Blinks/Smiles)
    ├── body_analyzer.py        # CV: DSP for motor stereotypy (Rocking/Flapping)
    ├── stimulus_engine.py      # Timeline director for the 90s protocol
    ├── risk_engine.py          # Math: Converts pixels to Z-Scores and DSM-5 mapping
    ├── mchat.py                # Logic & scoring for the M-CHAT-R/F questionnaire
    ├── chatbot.py              # Gemini RAG Copilot with session context injection
    ├── social_stories.py       # Carol Gray compliant AI story generator
    ├── visual_schedule.py      # TEACCH method printable schedule builder
    ├── therapy_goals.py        # ABA discrete trial data collection system
    ├── milestones.py           # CDC 2022 Milestone tracker
    ├── referral_generator.py   # PDF generation for Clinician packets
    └── data_store.py           # Local JSON persistence for patient data

```

---

## 🛡️ Disclaimer

**Autisense is a screening and diagnostic support prototype, not a definitive diagnostic tool.** All Z-scores and clinical deviations should be reviewed by a licensed developmental pediatrician or neurologist. Information generated by the AI copilot and post-diagnosis tools are for educational and management purposes and do not substitute professional medical advice.
