# modules/resource_directory.py
"""
Context-Aware Resource & Referral Directory.
Maps flagged screening domains to curated professional resources,
organizations, evidence-based interventions, and helplines.

All resources are free/public access. No paid partnerships.
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Resource:
    """A single resource entry."""
    name: str
    url: str
    description: str
    resource_type: str       # "organization", "tool", "directory", "hotline", "app", "guide"
    free: bool = True
    phone: str = ""
    note: str = ""


@dataclass
class Intervention:
    """An evidence-based intervention approach."""
    name: str
    abbreviation: str
    description: str
    evidence_level: str       # "strong", "moderate", "emerging"
    age_range: str
    typical_provider: str
    reference: str


@dataclass
class DomainResources:
    """Complete resource bundle for a single clinical domain."""
    domain: str
    dsm5_code: str
    description: str
    specialists: List[Dict]
    resources: List[Resource]
    interventions: List[Intervention]
    parent_tips: List[str]


# ─────────────────────────────────────────────
# SPECIALIST DATABASE
# ─────────────────────────────────────────────
SPECIALIST_TYPES = {
    "developmental_pediatrician": {
        "title": "Developmental-Behavioral Pediatrician",
        "abbreviation": "DBP",
        "description": (
            "Physician with specialized training in developmental and behavioral "
            "disorders of childhood. Can diagnose ASD and prescribe medications."
        ),
        "find_url": "https://www.sdbp.org/find-a-provider",
        "typical_wait": "3-12 months (varies by region)",
    },
    "child_psychologist": {
        "title": "Child Psychologist (ASD Specialty)",
        "abbreviation": "PhD/PsyD",
        "description": (
            "Licensed psychologist trained in administering ADOS-2, ADI-R, and "
            "comprehensive cognitive/behavioral assessments."
        ),
        "find_url": "https://locator.apa.org/",
        "typical_wait": "2-6 months",
    },
    "slp": {
        "title": "Speech-Language Pathologist",
        "abbreviation": "SLP / CCC-SLP",
        "description": (
            "Evaluates and treats speech, language, social communication, and "
            "feeding/swallowing disorders. Often the first interventionist a child sees."
        ),
        "find_url": "https://profind.asha.org/",
        "typical_wait": "1-3 months",
    },
    "ot": {
        "title": "Occupational Therapist",
        "abbreviation": "OT / OTR",
        "description": (
            "Addresses sensory processing, fine motor skills, self-care skills, "
            "and adaptive behavior. Key for sensory integration concerns."
        ),
        "find_url": "https://www.aota.org/practice/find-an-ot",
        "typical_wait": "1-3 months",
    },
    "bcba": {
        "title": "Board Certified Behavior Analyst",
        "abbreviation": "BCBA",
        "description": (
            "Designs and supervises Applied Behavior Analysis (ABA) programs. "
            "Conducts functional behavior assessments."
        ),
        "find_url": "https://www.bacb.com/find-a-certificant/",
        "typical_wait": "1-4 months",
    },
    "audiologist": {
        "title": "Audiologist",
        "abbreviation": "AuD",
        "description": (
            "Evaluates hearing and auditory processing. Critical to rule out "
            "hearing loss as a contributor to speech/language delays."
        ),
        "find_url": "https://www.asha.org/profind/",
        "typical_wait": "2-6 weeks",
    },
    "pediatric_neurologist": {
        "title": "Pediatric Neurologist",
        "abbreviation": "MD (Neuro)",
        "description": (
            "Evaluates neurological conditions that may co-occur with ASD, "
            "including seizure disorders, genetic conditions, and motor disorders."
        ),
        "find_url": "https://www.childneurologysociety.org/resources/find-a-child-neurologist/",
        "typical_wait": "2-6 months",
    },
}


# ─────────────────────────────────────────────
# GLOBAL RESOURCES (always shown)
# ─────────────────────────────────────────────
GLOBAL_RESOURCES = [
    Resource(
        name="Autism Speaks Resource Guide",
        url="https://www.autismspeaks.org/resource-guide",
        description="Searchable directory of autism service providers, support groups, and resources by ZIP code.",
        resource_type="directory",
    ),
    Resource(
        name="CDC - Learn the Signs. Act Early.",
        url="https://www.cdc.gov/actearly/",
        description="Free milestone checklists, tips, and parent resources from the Centers for Disease Control.",
        resource_type="guide",
    ),
    Resource(
        name="Autism Navigator",
        url="https://autismnavigator.com/",
        description="Free video library showing early signs of autism and typical development side-by-side.",
        resource_type="tool",
    ),
    Resource(
        name="First Signs",
        url="https://www.firstsigns.org/",
        description="Educational materials about early recognition of developmental delays and disorders.",
        resource_type="organization",
    ),
    Resource(
        name="SPARK for Autism (Simons Foundation)",
        url="https://sparkforautism.org/",
        description="Largest autism research study in the US. Free genetic testing for qualifying families.",
        resource_type="organization",
    ),
    Resource(
        name="Family Voices",
        url="https://familyvoices.org/",
        description="National network connecting families of children with special healthcare needs to resources.",
        resource_type="organization",
    ),
    Resource(
        name="Autism Society of America Helpline",
        url="https://www.autism-society.org/get-help/",
        description="National helpline providing information, referrals, and support.",
        resource_type="hotline",
        phone="1-800-328-8476",
    ),
    Resource(
        name="Early Intervention (Part C) - IDEA",
        url="https://www.cdc.gov/ncbddd/actearly/parents/states.html",
        description="Find your state's Early Intervention program. Free evaluation for children 0-3 years.",
        resource_type="directory",
        note="Federally mandated. Every state has one. Services are free or low-cost.",
    ),
    Resource(
        name="Understood.org",
        url="https://www.understood.org/",
        description="Free resources for families navigating learning and thinking differences.",
        resource_type="guide",
    ),
]


# ─────────────────────────────────────────────
# EVIDENCE-BASED INTERVENTIONS DATABASE
# ─────────────────────────────────────────────
INTERVENTIONS_DB = {
    "aba": Intervention(
        name="Applied Behavior Analysis",
        abbreviation="ABA",
        description=(
            "Systematic approach using behavioral principles to teach skills and "
            "reduce challenging behaviors. The most researched intervention for ASD."
        ),
        evidence_level="strong",
        age_range="18 months - adult",
        typical_provider="BCBA (Board Certified Behavior Analyst)",
        reference="National Autism Center, National Standards Project Phase 2 (2015)",
    ),
    "esdm": Intervention(
        name="Early Start Denver Model",
        abbreviation="ESDM",
        description=(
            "Play-based, relationship-focused intervention for toddlers 12-48 months. "
            "Combines ABA with developmental and relationship-based approaches."
        ),
        evidence_level="strong",
        age_range="12-48 months",
        typical_provider="Certified ESDM Therapist",
        reference="Dawson et al. (2010). Pediatrics, 125(1), e17-e23.",
    ),
    "pecs": Intervention(
        name="Picture Exchange Communication System",
        abbreviation="PECS",
        description=(
            "Augmentative communication system teaching children to exchange "
            "pictures for desired items/activities. Builds toward spontaneous communication."
        ),
        evidence_level="strong",
        age_range="18 months - adult",
        typical_provider="SLP or BCBA",
        reference="Bondy & Frost (1994). Focus on Autistic Behavior, 9(3), 1-19.",
    ),
    "dir_floortime": Intervention(
        name="DIR/Floortime",
        abbreviation="DIR",
        description=(
            "Developmental, Individual-difference, Relationship-based model. "
            "Follows child's lead in play to build social-emotional and communication skills."
        ),
        evidence_level="moderate",
        age_range="12 months - 12 years",
        typical_provider="DIR/Floortime certified therapist",
        reference="Greenspan & Wieder (2006). Engaging Autism. Da Capo Press.",
    ),
    "teacch": Intervention(
        name="TEACCH Structured Teaching",
        abbreviation="TEACCH",
        description=(
            "Uses visual supports, structured work systems, and environmental organization "
            "to support learning and independence."
        ),
        evidence_level="moderate",
        age_range="All ages",
        typical_provider="Special educator, OT, or BCBA",
        reference="Mesibov, Shea, & Schopler (2005). The TEACCH Approach. Springer.",
    ),
    "hanen": Intervention(
        name="Hanen - More Than Words",
        abbreviation="Hanen MTW",
        description=(
            "Parent-implemented program teaching responsive interaction strategies "
            "to promote communication in children with ASD."
        ),
        evidence_level="moderate",
        age_range="Birth - 5 years",
        typical_provider="Hanen-certified SLP",
        reference="Sussman (1999). More Than Words. Hanen Centre.",
    ),
    "social_thinking": Intervention(
        name="Social Thinking",
        abbreviation="ST",
        description=(
            "Curriculum teaching the cognitive process behind social skills - "
            "understanding others' perspectives, expected/unexpected behaviors."
        ),
        evidence_level="moderate",
        age_range="4 years - adult",
        typical_provider="SLP, Psychologist, or Special Educator",
        reference="Winner (2007). Thinking About You Thinking About Me. Think Social Publishing.",
    ),
    "sensory_integration": Intervention(
        name="Ayres Sensory Integration",
        abbreviation="ASI",
        description=(
            "Structured therapy addressing sensory processing challenges through "
            "active engagement in sensory-rich activities."
        ),
        evidence_level="moderate",
        age_range="2 years - 12 years",
        typical_provider="OT with sensory integration certification",
        reference="Ayres (1972). Sensory Integration and Learning Disorders. Western Psychological Services.",
    ),
}


# ─────────────────────────────────────────────
# DOMAIN → RESOURCE MAPPING
# ─────────────────────────────────────────────
DOMAIN_RESOURCE_MAP = {
    "social_communication": DomainResources(
        domain="Social Communication",
        dsm5_code="A.1 - Social-Emotional Reciprocity",
        description="Concerns with social back-and-forth, sharing interests, and initiating/responding to social interactions.",
        specialists=[
            SPECIALIST_TYPES["child_psychologist"],
            SPECIALIST_TYPES["slp"],
            SPECIALIST_TYPES["developmental_pediatrician"],
        ],
        resources=[
            Resource(
                name="Social Communication Milestones - ASHA",
                url="https://www.asha.org/public/speech/development/social-communication/",
                description="Comprehensive guide to social communication development milestones.",
                resource_type="guide",
            ),
            Resource(
                name="Autism Speaks - Social Skills Resources",
                url="https://www.autismspeaks.org/social-skills-and-autism",
                description="Free guides, tool kits, and video resources for building social skills.",
                resource_type="guide",
            ),
            Resource(
                name="Hanen Centre - Parent Tips",
                url="https://www.hanen.org/helpful-info/articles.aspx",
                description="Free articles on supporting communication development at home.",
                resource_type="guide",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["esdm"],
            INTERVENTIONS_DB["dir_floortime"],
            INTERVENTIONS_DB["hanen"],
            INTERVENTIONS_DB["social_thinking"],
        ],
        parent_tips=[
            "Follow your child's lead during play - join what they're interested in before redirecting",
            "Narrate your actions and your child's actions during daily routines (e.g., 'I'm putting on your shoes!')",
            "Wait at least 5 seconds after speaking to give your child time to process and respond",
            "Use exaggerated facial expressions and animated speech to draw attention to your face",
            "Create opportunities for your child to communicate (e.g., place desired items out of reach)",
            "Respond to ALL communication attempts - gestures, sounds, looks, and words all count",
        ],
    ),
    "nonverbal_communication": DomainResources(
        domain="Nonverbal Communication",
        dsm5_code="A.2 - Nonverbal Communicative Behaviors",
        description="Concerns with eye contact, gestures, facial expressions, and body language in communication.",
        specialists=[
            SPECIALIST_TYPES["slp"],
            SPECIALIST_TYPES["child_psychologist"],
            SPECIALIST_TYPES["ot"],
        ],
        resources=[
            Resource(
                name="ASHA - Eye Contact and Communication",
                url="https://www.asha.org/public/speech/development/",
                description="Understanding the role of eye contact and nonverbal cues in communication development.",
                resource_type="guide",
            ),
            Resource(
                name="Autism Speaks - Visual Supports Tool Kit",
                url="https://www.autismspeaks.org/tool-kit/visual-supports-and-autism-spectrum-disorder",
                description="Free toolkit for using visual supports to enhance communication.",
                resource_type="tool",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["pecs"],
            INTERVENTIONS_DB["hanen"],
            INTERVENTIONS_DB["dir_floortime"],
        ],
        parent_tips=[
            "Get on your child's eye level during interactions - sit on the floor or kneel down",
            "Hold interesting objects near your face to naturally draw eye contact",
            "Point to things and look back at your child - model joint attention",
            "Use gestures alongside words (wave while saying 'bye-bye', nod while saying 'yes')",
            "Celebrate any eye contact your child makes - smile warmly and respond immediately",
            "Don't force eye contact - build positive associations with looking at faces instead",
        ],
    ),
    "relationships": DomainResources(
        domain="Developing & Maintaining Relationships",
        dsm5_code="A.3 - Developing, Maintaining, and Understanding Relationships",
        description="Concerns with peer interest, imaginative play, social engagement, and adjusting behavior to social context.",
        specialists=[
            SPECIALIST_TYPES["child_psychologist"],
            SPECIALIST_TYPES["bcba"],
            SPECIALIST_TYPES["slp"],
        ],
        resources=[
            Resource(
                name="Autism Speaks - Making Friends",
                url="https://www.autismspeaks.org/social-skills-and-autism",
                description="Guides for supporting friendship development in children with autism.",
                resource_type="guide",
            ),
            Resource(
                name="IRIS Center - Peer-Mediated Instruction",
                url="https://iris.peabody.vanderbilt.edu/",
                description="Free modules on evidence-based peer interaction strategies for educators.",
                resource_type="guide",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["social_thinking"],
            INTERVENTIONS_DB["dir_floortime"],
            INTERVENTIONS_DB["aba"],
        ],
        parent_tips=[
            "Arrange short, structured playdates with one peer at a time",
            "Choose activities with clear rules (board games, building blocks) over unstructured free play initially",
            "Practice play scenarios at home before trying them in social settings",
            "Model pretend play - pick up a toy phone and 'call' your child, feed a stuffed animal together",
            "Praise specific social behaviors: 'I loved how you shared the blocks with your friend!'",
            "Don't compare your child's social development to siblings or peers - every child's path is different",
        ],
    ),
    "repetitive_behaviors": DomainResources(
        domain="Repetitive Behaviors & Restricted Interests",
        dsm5_code="B.1 - Stereotyped or Repetitive Motor Movements",
        description="Concerns with motor stereotypies (hand flapping, body rocking), repetitive object use, or echolalia.",
        specialists=[
            SPECIALIST_TYPES["bcba"],
            SPECIALIST_TYPES["ot"],
            SPECIALIST_TYPES["child_psychologist"],
        ],
        resources=[
            Resource(
                name="Autism Speaks - Repetitive Behaviors Tool Kit",
                url="https://www.autismspeaks.org/tool-kit/challenging-behaviors-tool-kit",
                description="Free guide for understanding and responding to repetitive and challenging behaviors.",
                resource_type="tool",
            ),
            Resource(
                name="Indiana Resource Center for Autism - Restricted Interests",
                url="https://www.iidc.indiana.edu/irca/",
                description="Research-based information on understanding and supporting restricted interests.",
                resource_type="guide",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["aba"],
            INTERVENTIONS_DB["sensory_integration"],
            INTERVENTIONS_DB["teacch"],
        ],
        parent_tips=[
            "Understand that stimming often serves a purpose - it may be self-regulating or enjoyable",
            "Only redirect repetitive behaviors that are harmful or significantly disruptive",
            "Offer alternative sensory outlets (fidget toys, chewy necklaces, movement breaks)",
            "Use restricted interests as motivation - incorporate them into learning and social activities",
            "Keep a log of when repetitive behaviors increase to identify triggers (tiredness, overstimulation)",
            "Work with an OT to create a sensory diet that provides regular sensory input throughout the day",
        ],
    ),
    "sensory_processing": DomainResources(
        domain="Sensory Processing",
        dsm5_code="B.4 - Hyper- or Hyporeactivity to Sensory Input",
        description="Concerns with over- or under-sensitivity to sounds, textures, lights, smells, or movement.",
        specialists=[
            SPECIALIST_TYPES["ot"],
            SPECIALIST_TYPES["audiologist"],
            SPECIALIST_TYPES["developmental_pediatrician"],
        ],
        resources=[
            Resource(
                name="STAR Institute for Sensory Processing",
                url="https://sensoryhealth.org/",
                description="Leading resource for sensory processing information, research, and provider directory.",
                resource_type="organization",
            ),
            Resource(
                name="Autism Speaks - Sensory Issues",
                url="https://www.autismspeaks.org/sensory-issues",
                description="Overview of sensory challenges in autism with practical strategies.",
                resource_type="guide",
            ),
            Resource(
                name="Sensory Processing Disorder Foundation",
                url="https://sensoryhealth.org/basic/about-spd",
                description="Educational resources about sensory processing differences.",
                resource_type="guide",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["sensory_integration"],
            INTERVENTIONS_DB["teacch"],
            INTERVENTIONS_DB["aba"],
        ],
        parent_tips=[
            "Create a sensory-friendly space at home - a quiet area with soft lighting and comfort items",
            "Give advance warnings before loud or overwhelming events (e.g., 'The blender will be loud')",
            "Carry a sensory kit in your bag (noise-canceling headphones, sunglasses, fidget, snack)",
            "Respect sensory boundaries - if your child covers their ears, the sound is genuinely painful to them",
            "Introduce new textures/foods gradually and without pressure",
            "Work with an OT to build a 'sensory diet' - scheduled sensory activities throughout the day",
        ],
    ),
    "auditory_response": DomainResources(
        domain="Auditory Response",
        dsm5_code="A.1 - Social-Emotional Reciprocity",
        description="Concerns with responding to name calls, verbal instructions, or environmental sounds.",
        specialists=[
            SPECIALIST_TYPES["audiologist"],
            SPECIALIST_TYPES["slp"],
            SPECIALIST_TYPES["developmental_pediatrician"],
        ],
        resources=[
            Resource(
                name="ASHA - Hearing Screening",
                url="https://www.asha.org/public/hearing/hearing-screening/",
                description="Information about hearing evaluations and when to seek testing.",
                resource_type="guide",
            ),
            Resource(
                name="Hearing First",
                url="https://www.hearingfirst.org/",
                description="Resources for families navigating hearing concerns and auditory development.",
                resource_type="organization",
            ),
        ],
        interventions=[
            INTERVENTIONS_DB["hanen"],
            INTERVENTIONS_DB["aba"],
            INTERVENTIONS_DB["esdm"],
        ],
        parent_tips=[
            "FIRST STEP: Always rule out hearing loss with a formal audiological evaluation",
            "Say your child's name clearly, at close range, before giving instructions",
            "Reduce background noise when communicating (turn off TV, move to quieter space)",
            "Use your child's name + pause + instruction (e.g., 'Alex... [wait for look] ...time for shoes')",
            "Pair verbal instructions with visual cues (point, gesture, show the object)",
            "Practice name-response as a game - call name, celebrate when they look, repeat throughout the day",
        ],
    ),
}


# ─────────────────────────────────────────────
# DOMAIN MATCHING ENGINE
# ─────────────────────────────────────────────
# Maps keywords from screening results to resource domains
DOMAIN_KEYWORD_MAP = {
    "social_communication": [
        "social_preference", "social_visual", "social reciprocity",
        "name response", "name_call", "joint attention",
        "showing", "sharing", "attention seeking",
        "social interest", "social referencing",
    ],
    "nonverbal_communication": [
        "eye_contact", "eye contact", "gaze", "flat_affect", "flat affect",
        "emotional_reciprocity", "smile", "imitation", "requesting",
    ],
    "relationships": [
        "imaginative play", "social interest", "peer",
    ],
    "repetitive_behaviors": [
        "motor_stereotypy", "motor stereotypy", "repetitive",
        "hand flapping", "body rocking", "stimming",
    ],
    "sensory_processing": [
        "sensory sensitivity", "sensory processing",
        "sensory", "hyper", "hypo",
    ],
    "auditory_response": [
        "auditory", "name_call", "name response",
        "hearing", "deaf",
    ],
}


class ResourceDirectory:
    """
    Context-aware resource directory that maps screening results
    to relevant resources, specialists, and interventions.
    """

    def __init__(self):
        self.matched_domains: Set[str] = set()

    def match_from_cv_assessment(self, cv_assessment) -> List[str]:
        """
        Analyze CV screening assessment and return matched resource domain keys.
        """
        matched = set()

        # Check domain scores
        domain_scores = getattr(cv_assessment, 'domain_scores', {})
        for score_name, val in domain_scores.items():
            z = val if isinstance(val, (int, float)) else val.get('z_score', 0)
            if abs(z) >= 1.0:
                score_lower = score_name.lower()
                for domain_key, keywords in DOMAIN_KEYWORD_MAP.items():
                    if any(kw in score_lower for kw in keywords):
                        matched.add(domain_key)

        # Check deviations
        deviations = getattr(cv_assessment, 'deviations', [])
        for dev in deviations:
            sig = getattr(dev, 'clinical_significance', 'typical')
            if sig in ('borderline', 'atypical'):
                name_lower = getattr(dev, 'domain_name', '').lower()
                for domain_key, keywords in DOMAIN_KEYWORD_MAP.items():
                    if any(kw in name_lower for kw in keywords):
                        matched.add(domain_key)

        self.matched_domains.update(matched)
        return sorted(matched)

    def match_from_mchat(self, mchat_result) -> List[str]:
        """
        Analyze M-CHAT-R results and return matched resource domain keys.
        """
        matched = set()

        flagged_domains = getattr(mchat_result, 'flagged_domains', [])
        for flagged in flagged_domains:
            flagged_lower = flagged.lower()
            for domain_key, keywords in DOMAIN_KEYWORD_MAP.items():
                if any(kw in flagged_lower for kw in keywords):
                    matched.add(domain_key)

        self.matched_domains.update(matched)
        return sorted(matched)

    def get_resources_for_domain(self, domain_key: str) -> Optional[DomainResources]:
        """Get the full resource bundle for a specific domain."""
        return DOMAIN_RESOURCE_MAP.get(domain_key)

    def get_all_matched_resources(self) -> List[DomainResources]:
        """Get resource bundles for all matched domains."""
        results = []
        for key in sorted(self.matched_domains):
            domain_res = DOMAIN_RESOURCE_MAP.get(key)
            if domain_res:
                results.append(domain_res)
        return results

    def get_global_resources(self) -> List[Resource]:
        """Return resources that are always relevant regardless of screening results."""
        return GLOBAL_RESOURCES

    def get_all_interventions(self) -> List[Intervention]:
        """Return all unique interventions across matched domains, sorted by evidence level."""
        seen = set()
        interventions = []
        evidence_order = {"strong": 0, "moderate": 1, "emerging": 2}

        for domain_res in self.get_all_matched_resources():
            for intervention in domain_res.interventions:
                if intervention.abbreviation not in seen:
                    seen.add(intervention.abbreviation)
                    interventions.append(intervention)

        interventions.sort(key=lambda x: evidence_order.get(x.evidence_level, 3))
        return interventions

    def get_all_specialists(self) -> List[Dict]:
        """Return all unique specialists across matched domains."""
        seen = set()
        specialists = []

        for domain_res in self.get_all_matched_resources():
            for spec in domain_res.specialists:
                title = spec.get("title", "")
                if title not in seen:
                    seen.add(title)
                    specialists.append(spec)

        return specialists

    def get_all_parent_tips(self) -> Dict[str, List[str]]:
        """Return parent tips grouped by domain."""
        tips = {}
        for domain_res in self.get_all_matched_resources():
            tips[domain_res.domain] = domain_res.parent_tips
        return tips

    def get_priority_actions(self, combined_risk: str = "LOW") -> List[str]:
        """
        Return prioritized action items based on combined risk level.
        These are concrete, numbered steps for a parent to take.
        """
        actions = []

        if combined_risk in ("HIGH", "High", "Elevated"):
            actions = [
                "📞 Call your pediatrician TODAY to discuss screening results and request a developmental referral",
                "📋 Contact your state's Early Intervention program (free evaluation for children 0-3)",
                "🏥 Request a comprehensive audiological evaluation to rule out hearing concerns",
                "📄 Download and bring the Referral PDF from this app to your pediatrician appointment",
                "📅 While waiting for evaluation, contact a Speech-Language Pathologist for initial assessment",
                "📚 Visit AutismNavigator.com to view free video examples of early autism signs",
                "💪 Remember: Early intervention leads to significantly better outcomes - you are doing the right thing",
            ]
        elif combined_risk in ("MEDIUM", "Medium", "Borderline"):
            actions = [
                "📞 Schedule an appointment with your pediatrician to discuss screening results",
                "📋 If your child is under 3, contact your state's Early Intervention program for a free evaluation",
                "📄 Download the Referral PDF to share with your child's doctor",
                "👀 Monitor developmental milestones using the CDC milestone tracker",
                "📅 Plan to rescreen in 3-6 months to track progress",
                "📚 Review the parent tips in this resource guide for activities you can start at home today",
            ]
        else:
            actions = [
                "✅ Current screening results are within typical range - no immediate action required",
                "👀 Continue monitoring developmental milestones at routine well-child visits",
                "📅 If your child is under 24 months, rescreen at the 24-month well-child visit",
                "📞 Contact your pediatrician anytime you have concerns about your child's development",
                "📚 Explore the parent tips and developmental resources provided for enrichment activities",
            ]

        return actions

    def generate_summary_dict(self) -> Dict:
        """Generate a complete summary for storage or chatbot context injection."""
        return {
            "matched_domains": sorted(self.matched_domains),
            "specialist_count": len(self.get_all_specialists()),
            "intervention_count": len(self.get_all_interventions()),
            "resource_domains": [
                {
                    "domain": dr.domain,
                    "dsm5": dr.dsm5_code,
                    "specialist_types": [s["title"] for s in dr.specialists],
                    "intervention_names": [i.name for i in dr.interventions],
                    "tip_count": len(dr.parent_tips),
                }
                for dr in self.get_all_matched_resources()
            ],
        }