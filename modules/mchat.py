"""
M-CHAT-R/F: Modified Checklist for Autism in Toddlers, Revised with Follow-Up.
Validated screening instrument for children aged 16-30 months.

FREE for clinical, research, and educational purposes.
Source: https://mchatscreen.com
Reference: Robins, Fein, & Barton (2009). JADD, 39(6), 827-843.

Scoring:
  - 0-2:  LOW risk    → Rescreen at next well-child visit if <24mo
  - 3-7:  MEDIUM risk → Administer Follow-Up; if Follow-Up ≥2, refer
  - 8-20: HIGH risk   → Bypass Follow-Up, refer immediately
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class MCHATResult:
    total_score: int
    risk_level: str                     # "LOW", "MEDIUM", "HIGH"
    risk_items: List[int]               # Question IDs answered at-risk
    critical_items_flagged: List[int]   # Critical items answered at-risk
    recommended_action: str
    detailed_actions: List[str]
    responses: Dict[int, str]           # {question_id: "Yes"/"No"}
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def risk_percentage(self) -> float:
        return (self.total_score / 20) * 100

    @property
    def flagged_domains(self) -> List[str]:
        """Return unique domains of all at-risk items."""
        domains = set()
        for q in MCHAT_QUESTIONS:
            if q["id"] in self.risk_items:
                domains.add(q["domain"])
        return sorted(domains)


# ─────────────────────────────────────────────
# ALL 20 M-CHAT-R QUESTIONS
# ─────────────────────────────────────────────
MCHAT_QUESTIONS = [
    {
        "id": 1,
        "text": "If you point at something across the room, does your child look at it?",
        "example": "For example, if you point at a toy or an animal, does your child look at the toy or animal?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Joint Attention"
    },
    {
        "id": 2,
        "text": "Have you ever wondered if your child might be deaf?",
        "example": "",
        "risk_answer": "Yes",
        "critical": True,
        "domain": "Auditory Response"
    },
    {
        "id": 3,
        "text": "Does your child play pretend or make-believe?",
        "example": "For example, pretend to drink from an empty cup, pretend to talk on a phone, or pretend to feed a doll or stuffed animal?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Imaginative Play"
    },
    {
        "id": 4,
        "text": "Does your child like climbing on things?",
        "example": "For example, furniture, playground equipment, or stairs.",
        "risk_answer": "No",
        "critical": False,
        "domain": "Motor Development"
    },
    {
        "id": 5,
        "text": "Does your child make unusual finger movements near his or her eyes?",
        "example": "For example, does your child wiggle his or her fingers close to his or her eyes?",
        "risk_answer": "Yes",
        "critical": True,
        "domain": "Repetitive Behaviors"
    },
    {
        "id": 6,
        "text": "Does your child point with one finger to ask for something or to get help?",
        "example": "For example, pointing to a snack or toy that is out of reach.",
        "risk_answer": "No",
        "critical": False,
        "domain": "Requesting"
    },
    {
        "id": 7,
        "text": "Does your child point with one finger to show you something interesting?",
        "example": "For example, pointing to an airplane in the sky or a big truck in the road.",
        "risk_answer": "No",
        "critical": True,
        "domain": "Joint Attention"
    },
    {
        "id": 8,
        "text": "Is your child interested in other children?",
        "example": "For example, does your child watch other children, smile at them, or go to them?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Social Interest"
    },
    {
        "id": 9,
        "text": "Does your child show you things by bringing them to you or holding them up for you to see — not to get help, but just to share?",
        "example": "For example, showing you a flower, a stuffed animal, or a toy truck.",
        "risk_answer": "No",
        "critical": True,
        "domain": "Showing / Sharing"
    },
    {
        "id": 10,
        "text": "Does your child respond when you call his or her name?",
        "example": "For example, does he or she look up, talk or babble, or stop what he or she is doing when you call his or her name?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Name Response"
    },
    {
        "id": 11,
        "text": "When you smile at your child, does he or she smile back at you?",
        "example": "",
        "risk_answer": "No",
        "critical": False,
        "domain": "Social Reciprocity"
    },
    {
        "id": 12,
        "text": "Does your child get upset by everyday sounds?",
        "example": "For example, does your child scream or cry to noise such as a vacuum cleaner or loud music?",
        "risk_answer": "Yes",
        "critical": False,
        "domain": "Sensory Sensitivity"
    },
    {
        "id": 13,
        "text": "Does your child walk?",
        "example": "",
        "risk_answer": "No",
        "critical": False,
        "domain": "Motor Development"
    },
    {
        "id": 14,
        "text": "Does your child look you in the eye when you are talking to him or her, playing with him or her, or dressing him or her?",
        "example": "",
        "risk_answer": "No",
        "critical": True,
        "domain": "Eye Contact"
    },
    {
        "id": 15,
        "text": "Does your child try to copy what you do?",
        "example": "For example, wave bye-bye, clap, or make a funny noise when you make one.",
        "risk_answer": "No",
        "critical": True,
        "domain": "Imitation"
    },
    {
        "id": 16,
        "text": "If you turn your head to look at something, does your child look around to see what you are looking at?",
        "example": "",
        "risk_answer": "No",
        "critical": False,
        "domain": "Joint Attention"
    },
    {
        "id": 17,
        "text": "Does your child try to get you to look at him or her?",
        "example": "For example, does your child look at you for praise, or say 'look' or 'look at me'?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Attention Seeking"
    },
    {
        "id": 18,
        "text": "Does your child understand when you tell him or her to do something?",
        "example": "For example, if you don't point, can your child understand 'put the book on the chair' or 'bring me the blanket'?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Receptive Language"
    },
    {
        "id": 19,
        "text": "If something new happens, does your child look at your face to see how you feel about it?",
        "example": "For example, if he or she hears a strange or funny noise, or sees a new toy, will he or she look at your face?",
        "risk_answer": "No",
        "critical": False,
        "domain": "Social Referencing"
    },
    {
        "id": 20,
        "text": "Does your child like movement activities?",
        "example": "For example, being swung or bounced on your knee.",
        "risk_answer": "No",
        "critical": False,
        "domain": "Sensory Processing"
    },
]


# ─────────────────────────────────────────────
# FOLLOW-UP QUESTIONS (SELECTED ITEMS)
# ─────────────────────────────────────────────
MCHAT_FOLLOWUP = {
    2: [
        "Does your child respond to sounds in the environment? For example, does he/she turn toward a new sound?",
        "Does your child respond when called from another room?",
        "Does your child respond to voices on TV or radio by looking or turning?",
    ],
    5: [
        "Does your child move his/her fingers near his/her eyes more than twice a day?",
        "Does your child stare at his/her fingers, hands, or objects he/she is holding for extended periods?",
    ],
    7: [
        "Does your child ever point at something to get you to look (not to get you to give it)?",
        "Does your child look back and forth between an interesting sight and you?",
    ],
    9: [
        "Does your child bring things to show you, like a drawing or a bug found outside?",
        "Does your child seek your reaction when he/she sees something new or exciting?",
    ],
    14: [
        "Does your child make eye contact during everyday interactions (not just when wanting something)?",
        "Does your child maintain eye contact for a few seconds when you talk to him/her?",
    ],
    15: [
        "If you clap your hands, does your child try to clap?",
        "If you wave, does your child wave back?",
        "If you blow a kiss, does your child try to blow one back or make any imitation attempt?",
    ],
}


# ─────────────────────────────────────────────
# DOMAIN GROUPINGS FOR CLINICAL SUMMARY
# ─────────────────────────────────────────────
DOMAIN_DSM5_MAP = {
    "Joint Attention": {
        "dsm5": "A.3 — Deficits in developing, maintaining, and understanding relationships",
        "description": "Difficulty sharing attention or following another person's gaze/point",
        "related_questions": [1, 7, 16],
    },
    "Social Reciprocity": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Reduced social-emotional back-and-forth interaction",
        "related_questions": [11, 19],
    },
    "Eye Contact": {
        "dsm5": "A.2 — Deficits in nonverbal communicative behaviors",
        "description": "Reduced or atypical eye contact patterns",
        "related_questions": [14],
    },
    "Showing / Sharing": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Reduced sharing of interests, emotions, or affect",
        "related_questions": [9],
    },
    "Name Response": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Reduced response to social bids for attention",
        "related_questions": [10],
    },
    "Auditory Response": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Possible auditory processing concerns; rule out hearing impairment",
        "related_questions": [2],
    },
    "Imitation": {
        "dsm5": "A.2 — Deficits in nonverbal communicative behaviors",
        "description": "Reduced motor imitation of gestures and actions",
        "related_questions": [15],
    },
    "Imaginative Play": {
        "dsm5": "A.3 — Deficits in developing, maintaining, and understanding relationships",
        "description": "Reduced symbolic or pretend play",
        "related_questions": [3],
    },
    "Repetitive Behaviors": {
        "dsm5": "B.1 — Stereotyped or repetitive motor movements",
        "description": "Unusual repetitive motor mannerisms",
        "related_questions": [5],
    },
    "Social Interest": {
        "dsm5": "A.3 — Deficits in developing, maintaining, and understanding relationships",
        "description": "Reduced interest in peers",
        "related_questions": [8],
    },
    "Requesting": {
        "dsm5": "A.2 — Deficits in nonverbal communicative behaviors",
        "description": "Reduced use of pointing to request",
        "related_questions": [6],
    },
    "Attention Seeking": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Reduced initiation of social interaction",
        "related_questions": [17],
    },
    "Receptive Language": {
        "dsm5": "A.2 — Deficits in nonverbal communicative behaviors",
        "description": "Difficulty understanding spoken instructions",
        "related_questions": [18],
    },
    "Sensory Sensitivity": {
        "dsm5": "B.4 — Hyper- or hyporeactivity to sensory input",
        "description": "Unusual sensory reactivity",
        "related_questions": [12],
    },
    "Motor Development": {
        "dsm5": "N/A — Motor milestone tracking",
        "description": "Gross motor development milestones",
        "related_questions": [4, 13],
    },
    "Sensory Processing": {
        "dsm5": "B.4 — Hyper- or hyporeactivity to sensory input",
        "description": "Sensory-seeking or sensory-avoidant patterns",
        "related_questions": [20],
    },
    "Social Referencing": {
        "dsm5": "A.1 — Deficits in social-emotional reciprocity",
        "description": "Reduced social referencing during novel situations",
        "related_questions": [19],
    },
}


# ─────────────────────────────────────────────
# SCREENER CLASS
# ─────────────────────────────────────────────
class MCHATScreener:
    """Implements the M-CHAT-R scoring algorithm and risk classification."""

    def __init__(self):
        self.responses: Dict[int, str] = {}
        self.followup_responses: Dict[int, Dict[int, str]] = {}

    def set_response(self, question_id: int, answer: str) -> None:
        """Record a parent response. answer must be 'Yes' or 'No'."""
        if answer not in ("Yes", "No"):
            raise ValueError(f"Answer must be 'Yes' or 'No', got '{answer}'")
        if question_id < 1 or question_id > 20:
            raise ValueError(f"Question ID must be 1-20, got {question_id}")
        self.responses[question_id] = answer

    def set_followup_response(self, question_id: int, sub_index: int, answer: str) -> None:
        """Record a follow-up sub-question response."""
        if question_id not in self.followup_responses:
            self.followup_responses[question_id] = {}
        self.followup_responses[question_id][sub_index] = answer

    def is_at_risk(self, question_id: int) -> bool:
        """Check if a single question was answered at-risk."""
        q = next((q for q in MCHAT_QUESTIONS if q["id"] == question_id), None)
        if q is None or question_id not in self.responses:
            return False
        return self.responses[question_id] == q["risk_answer"]

    def get_risk_items(self) -> List[int]:
        """Return list of question IDs answered at-risk."""
        return [q["id"] for q in MCHAT_QUESTIONS if self.is_at_risk(q["id"])]

    def get_critical_risk_items(self) -> List[int]:
        """Return list of critical question IDs answered at-risk."""
        return [
            q["id"] for q in MCHAT_QUESTIONS
            if q["critical"] and self.is_at_risk(q["id"])
        ]

    def is_complete(self) -> bool:
        """Check if all 20 questions have been answered."""
        return len(self.responses) == 20

    def get_unanswered(self) -> List[int]:
        """Return list of unanswered question IDs."""
        return [q["id"] for q in MCHAT_QUESTIONS if q["id"] not in self.responses]

    def get_progress(self) -> float:
        """Return completion percentage 0.0 to 1.0."""
        return len(self.responses) / 20.0

    def score(self) -> MCHATResult:
        """
        Compute the M-CHAT-R total score and risk classification.
        
        Returns MCHATResult with full clinical breakdown.
        Raises ValueError if questionnaire is incomplete.
        """
        if not self.is_complete():
            unanswered = self.get_unanswered()
            raise ValueError(
                f"Cannot score incomplete questionnaire. "
                f"Missing questions: {unanswered}"
            )

        risk_items = self.get_risk_items()
        critical_flagged = self.get_critical_risk_items()
        total_score = len(risk_items)

        # ── Risk Classification (per published algorithm) ──
        if total_score >= 8:
            risk_level = "HIGH"
            recommended_action = (
                "Immediate referral for diagnostic evaluation and early intervention services."
            )
            detailed_actions = [
                "Bypass M-CHAT-R/F Follow-Up — risk is sufficiently elevated",
                "Refer to developmental pediatrician or child psychologist for ADOS-2 evaluation",
                "Refer for comprehensive audiological evaluation",
                "Initiate early intervention services (do not wait for formal diagnosis)",
                "Refer to speech-language pathologist for communication assessment",
                "Connect family with state Early Intervention (Part C) program",
                "Schedule follow-up within 30 days to confirm referral completion",
            ]

        elif total_score >= 3:
            risk_level = "MEDIUM"
            recommended_action = (
                "Administer M-CHAT-R/F Follow-Up interview. "
                "If Follow-Up score ≥ 2, refer for diagnostic evaluation."
            )
            detailed_actions = [
                "Conduct M-CHAT-R/F Follow-Up interview on flagged items",
                "If Follow-Up confirms ≥ 2 items: refer for developmental evaluation",
                "If Follow-Up reduces to < 2 items: monitor and rescreen in 3-6 months",
                "Discuss developmental concerns with family using empathetic, strengths-based language",
                "Document this screening result in the child's medical record",
                "Consider concurrent speech/language and occupational therapy referrals",
            ]

        else:
            risk_level = "LOW"
            recommended_action = (
                "No immediate action required. Rescreen at 24-month well-child visit "
                "if child is currently under 24 months."
            )
            detailed_actions = [
                "No immediate referral indicated based on M-CHAT-R score alone",
                "If child < 24 months, rescreen at the 24-month well-child visit",
                "If child ≥ 24 months, no further M-CHAT-R screening needed",
                "Continue routine developmental surveillance at all well-child visits",
                "Remain responsive to any future parental concerns about development",
                "Consider other screening if clinical judgment suggests concerns despite low score",
            ]

        return MCHATResult(
            total_score=total_score,
            risk_level=risk_level,
            risk_items=risk_items,
            critical_items_flagged=critical_flagged,
            recommended_action=recommended_action,
            detailed_actions=detailed_actions,
            responses=dict(self.responses),
        )

    def get_domain_summary(self) -> List[Dict]:
        """
        Group at-risk items by clinical domain with DSM-5 mapping.
        Returns a list of domain summaries only for flagged domains.
        """
        result = self.score()
        summary = []

        for domain_name, domain_info in DOMAIN_DSM5_MAP.items():
            flagged_in_domain = [
                qid for qid in domain_info["related_questions"]
                if qid in result.risk_items
            ]

            if flagged_in_domain:
                flagged_questions = [
                    next(q for q in MCHAT_QUESTIONS if q["id"] == qid)
                    for qid in flagged_in_domain
                ]

                has_critical = any(q["critical"] for q in flagged_questions)

                summary.append({
                    "domain": domain_name,
                    "dsm5_code": domain_info["dsm5"],
                    "description": domain_info["description"],
                    "flagged_questions": flagged_in_domain,
                    "flagged_count": len(flagged_in_domain),
                    "total_in_domain": len(domain_info["related_questions"]),
                    "contains_critical_item": has_critical,
                    "severity": "high" if has_critical else "moderate",
                })

        # Sort by severity (critical domains first)
        summary.sort(key=lambda x: (x["contains_critical_item"], x["flagged_count"]), reverse=True)
        return summary

    def needs_followup(self) -> bool:
        """Check if the M-CHAT-R/F Follow-Up interview should be administered."""
        if not self.is_complete():
            return False
        result = self.score()
        return result.risk_level == "MEDIUM"

    def get_followup_items(self) -> List[Dict]:
        """
        Return Follow-Up questions for items that were flagged at-risk.
        Only returns follow-up items that exist in the MCHAT_FOLLOWUP bank.
        """
        risk_items = self.get_risk_items()
        followup_items = []

        for qid in risk_items:
            if qid in MCHAT_FOLLOWUP:
                parent_q = next(q for q in MCHAT_QUESTIONS if q["id"] == qid)
                followup_items.append({
                    "parent_question_id": qid,
                    "parent_question_text": parent_q["text"],
                    "domain": parent_q["domain"],
                    "followup_probes": MCHAT_FOLLOWUP[qid],
                })

        return followup_items

    def score_followup(self) -> Optional[Dict]:
        """
        Score the Follow-Up interview.
        An item PASSES Follow-Up if ANY sub-question is answered 'Yes' (typical behavior).
        An item FAILS Follow-Up if ALL sub-questions are answered 'No' (at-risk).
        """
        if not self.followup_responses:
            return None

        items_still_at_risk = []
        items_passed = []

        for qid, sub_responses in self.followup_responses.items():
            # Item passes if ANY follow-up probe is answered indicating typical behavior
            parent_q = next((q for q in MCHAT_QUESTIONS if q["id"] == qid), None)
            if parent_q is None:
                continue

            any_typical = any(
                ans == "Yes" for ans in sub_responses.values()
            )

            if any_typical:
                items_passed.append(qid)
            else:
                items_still_at_risk.append(qid)

        followup_score = len(items_still_at_risk)

        if followup_score >= 2:
            followup_risk = "REFER"
            followup_action = "Refer for diagnostic evaluation — Follow-Up confirms risk."
        else:
            followup_risk = "MONITOR"
            followup_action = "Risk reduced by Follow-Up. Monitor and rescreen in 3-6 months."

        return {
            "followup_score": followup_score,
            "items_still_at_risk": items_still_at_risk,
            "items_resolved": items_passed,
            "followup_risk": followup_risk,
            "followup_action": followup_action,
        }

    def generate_combined_risk_summary(
        self,
        cv_assessment=None
    ) -> Dict:
        """
        Combine M-CHAT-R results with the CV screening assessment 
        for a unified multi-modal risk picture.
        
        Parameters:
            cv_assessment: The assessment object from risk_engine.compute_assessment()
        
        Returns a combined summary dictionary.
        """
        mchat_result = self.score()
        domain_summary = self.get_domain_summary()

        combined = {
            "mchat": {
                "total_score": mchat_result.total_score,
                "risk_level": mchat_result.risk_level,
                "risk_items": mchat_result.risk_items,
                "critical_items_flagged": mchat_result.critical_items_flagged,
                "flagged_domains": mchat_result.flagged_domains,
                "domain_details": domain_summary,
                "recommended_action": mchat_result.recommended_action,
            },
            "cv_screening": None,
            "combined_risk": mchat_result.risk_level,
            "combined_recommendation": mchat_result.recommended_action,
            "concordance": "N/A",
        }

        if cv_assessment is not None:
            cv_risk = getattr(cv_assessment, 'risk_level', 'Unknown')
            combined["cv_screening"] = {
                "risk_level": cv_risk,
                "summary": getattr(cv_assessment, 'summary', ''),
                "domain_scores": getattr(cv_assessment, 'domain_scores', {}),
            }

            # ── Combined risk matrix ──
            risk_hierarchy = {"Typical": 0, "Borderline": 1, "Elevated": 2, "High": 3}
            mchat_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

            cv_rank = risk_hierarchy.get(cv_risk, 0)
            mq_rank = mchat_rank.get(mchat_result.risk_level, 0)

            max_rank = max(cv_rank, mq_rank)

            if max_rank >= 2:
                combined["combined_risk"] = "HIGH"
                combined["combined_recommendation"] = (
                    "Both screening modalities indicate elevated concern. "
                    "Immediate referral for comprehensive developmental evaluation is recommended."
                )
            elif max_rank == 1:
                combined["combined_risk"] = "MEDIUM"
                combined["combined_recommendation"] = (
                    "One or both screening modalities indicate moderate concern. "
                    "Administer M-CHAT-R/F Follow-Up and consider referral for evaluation."
                )
            else:
                combined["combined_risk"] = "LOW"
                combined["combined_recommendation"] = (
                    "Both screening modalities are within typical range. "
                    "Continue routine developmental surveillance."
                )

            # ── Concordance check ──
            if cv_rank >= 1 and mq_rank >= 1:
                combined["concordance"] = "CONVERGENT — Both modalities flag concern"
            elif cv_rank == 0 and mq_rank == 0:
                combined["concordance"] = "CONVERGENT — Both modalities within typical range"
            else:
                combined["concordance"] = (
                    "DIVERGENT — Modalities disagree. Clinical judgment advised. "
                    f"CV Screening: {cv_risk} | M-CHAT-R: {mchat_result.risk_level}"
                )

        return combined

    def to_dict(self) -> Dict:
        """Serialize the screener state for storage."""
        return {
            "responses": dict(self.responses),
            "followup_responses": {
                str(k): dict(v) for k, v in self.followup_responses.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MCHATScreener":
        """Restore screener state from stored data."""
        screener = cls()
        for qid_str, answer in data.get("responses", {}).items():
            screener.responses[int(qid_str)] = answer
        for qid_str, subs in data.get("followup_responses", {}).items():
            screener.followup_responses[int(qid_str)] = {
                int(k): v for k, v in subs.items()
            }
        return screener