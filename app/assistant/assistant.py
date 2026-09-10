"""
AI Operations Assistant for Godrej Warehouse AI.
Deterministic, grounded operational query engine for warehouse supervisors.
Provides factual, explainable answers regarding material handling violations,
risk classifications, repeat infractions, and preventive safety recommendations.
Strictly adheres to Godrej Responsible AI damage assessment principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.analytics.engine import AnalyticsEngine, BEHAVIOUR_PREVENTION_MAP
from app.risk.models import (
    DamageAssessmentStatus,
    IncidentRecord,
    RiskConfig,
    RiskLevel,
)


class QueryIntent(str, Enum):
    """Recognized operator query intents."""
    SUMMARY = "SUMMARY"
    HIGHEST_RISK = "HIGHEST_RISK"
    DIAGNOSIS = "DIAGNOSIS"
    REPEAT_HISTORY = "REPEAT_HISTORY"
    RECOMMENDATIONS = "RECOMMENDATIONS"
    FILTER_RULES = "FILTER_RULES"
    FILTER_RISK = "FILTER_RISK"
    BEHAVIOUR_FREQUENCY = "BEHAVIOUR_FREQUENCY"
    FREQUENCY = "BEHAVIOUR_FREQUENCY"
    ANALYTICS_RISK = "ANALYTICS_RISK"
    ANALYTICS_LOCATION = "ANALYTICS_LOCATION"
    PREVENTION_PRIORITIES = "PREVENTION_PRIORITIES"
    FALLBACK = "FALLBACK"


@dataclass
class AssistantResponse:
    """Standardized response from the AI Operations Assistant."""
    answer: str
    intent: QueryIntent
    matched_incidents: List[IncidentRecord] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "intent": self.intent.value,
            "matched_count": len(self.matched_incidents),
            "matched_incident_ids": [i.incident_id for i in self.matched_incidents],
            "data": self.data,
        }


def _dict_to_incident(data: Dict[str, Any]) -> IncidentRecord:
    """Converts a raw JSON dictionary to an IncidentRecord instance."""
    level_str = str(data.get("risk_level", "MEDIUM")).upper()
    try:
        risk_level = RiskLevel(level_str)
    except ValueError:
        risk_level = RiskLevel.MEDIUM

    status_str = str(data.get("damage_assessment", "POTENTIAL_DAMAGE_RISK"))
    try:
        damage_status = DamageAssessmentStatus(status_str)
    except ValueError:
        damage_status = DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK

    raw_bbox = data.get("bbox", [0, 0, 0, 0])
    bbox = tuple(raw_bbox) if len(raw_bbox) == 4 else (0, 0, 0, 0)

    return IncidentRecord(
        incident_id=str(data.get("incident_id", "")),
        event_id=str(data.get("event_id", "")),
        rule_name=str(data.get("rule_name", "")),
        human_readable_name=str(data.get("human_readable_name", "")),
        risk_level=risk_level,
        risk_score=float(data.get("risk_score", 0.0)),
        timestamp_seconds=float(data.get("timestamp_seconds", 0.0)),
        frame_idx=int(data.get("frame_idx", 0)),
        track_id=int(data.get("track_id", 0)),
        display_label=str(data.get("display_label", f"Object #{data.get('track_id', 0)}")),
        secondary_track_id=data.get("secondary_track_id"),
        secondary_label=data.get("secondary_label"),
        class_name=str(data.get("class_name", "carton")),
        confidence=float(data.get("confidence", 0.8)),
        bbox=bbox,
        metrics=data.get("metrics", {}) or {},
        damage_assessment=damage_status,
        explanation=str(data.get("explanation", "")),
        recommendation=str(data.get("recommendation", "")),
        evidence_image_path=data.get("evidence_image_path"),
        video_clip_path=data.get("video_clip_path"),
    )


class OperationsAssistant:
    """
    Deterministic, grounded AI Operations Assistant for Godrej Warehouse AI.
    Interprets incident manifest data, evaluates risk drivers, tracks repeat
    infractions, and provides actionable safety mitigation recommendations.
    """

    POLICY_DISCLAIMER = (
        "ℹ️ **Godrej Operational Policy**: Observed Behaviour → Potential Damage Risk → Confirmed Damage. "
        "All incidents represent Potential Damage Risk (Unconfirmed); physical damage is not confirmed without physical QA inspection."
    )

    RULE_KEYWORDS = {
        "product_dropped": ["drop", "dropped", "falling", "fall", "freefall"],
        "product_thrown": ["throw", "thrown", "throwing", "toss", "tossed", "ballistic"],
        "product_dragged": ["drag", "dragged", "dragging", "slide", "sliding"],
        "product_pushed": ["push", "pushed", "pushing", "kick", "kicked", "kicking", "shove"],
        "product_rolled": ["roll", "rolled", "rolling", "tumble", "tumbling"],
        "unstable_stacking": ["stack", "stacking", "stacked", "column", "lean", "leaning", "unstable"],
        "pallet_overhang": ["overhang", "pallet", "protrusion", "protrude", "perimeter"],
        "product_outside_designated_area": ["outside", "designated area", "unauthorized zone", "placement", "storage area", "permitted area", "boundary", "demarcation"],
        "product_handled_without_required_equipment": ["equipment", "required equipment", "forklift", "pallet jack", "trolley", "manual handling", "unassisted", "machinery", "mechanical aid"],
        "unsafe_loading_unloading_sequence": ["sequence", "loading sequence", "unloading sequence", "unsafe sequence", "out of order", "out of sequence", "loading", "unloading", "positioning step"],
    }

    def __init__(
        self,
        incidents: Optional[List[IncidentRecord]] = None,
        config: Optional[RiskConfig] = None,
    ):
        self.config = config or RiskConfig()
        self.incidents: List[IncidentRecord] = []
        self._by_id: Dict[str, IncidentRecord] = {}
        self._by_track: Dict[int, List[IncidentRecord]] = {}
        self._by_level: Dict[RiskLevel, List[IncidentRecord]] = {
            RiskLevel.LOW: [],
            RiskLevel.MEDIUM: [],
            RiskLevel.HIGH: [],
            RiskLevel.CRITICAL: [],
        }
        self._by_rule: Dict[str, List[IncidentRecord]] = {}
        self.analytics_engine: AnalyticsEngine = AnalyticsEngine([])

        if incidents:
            self.set_incidents(incidents)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        config: Optional[RiskConfig] = None,
    ) -> OperationsAssistant:
        """Constructs an OperationsAssistant by loading a JSON incident manifest."""
        assistant = cls(config=config)
        assistant.load_manifest(manifest_path)
        return assistant

    @classmethod
    def from_incidents(
        cls,
        incidents: List[IncidentRecord],
        config: Optional[RiskConfig] = None,
    ) -> OperationsAssistant:
        """Constructs an OperationsAssistant from a list of IncidentRecord instances."""
        return cls(incidents=incidents, config=config)

    def load_manifest(self, manifest_path: Union[str, Path]) -> int:
        """
        Loads incidents from a JSON manifest file on disk.
        Returns the number of incidents loaded.
        """
        path = Path(manifest_path)
        if not path.exists():
            self.set_incidents([])
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            self.set_incidents([])
            return 0

        loaded_incidents = [_dict_to_incident(item) for item in data if isinstance(item, dict)]
        self.set_incidents(loaded_incidents)
        return len(loaded_incidents)

    def set_incidents(self, incidents: List[IncidentRecord]) -> None:
        """Updates the assistant's incident records and rebuilds fast lookup indexes."""
        self.incidents = list(incidents)
        self.analytics_engine = AnalyticsEngine(self.incidents)
        self._by_id.clear()
        self._by_track.clear()
        self._by_level = {
            RiskLevel.LOW: [],
            RiskLevel.MEDIUM: [],
            RiskLevel.HIGH: [],
            RiskLevel.CRITICAL: [],
        }
        self._by_rule.clear()

        for inc in self.incidents:
            # Index by ID (case-insensitive key)
            self._by_id[inc.incident_id.lower()] = inc

            # Index by track_id
            if inc.track_id not in self._by_track:
                self._by_track[inc.track_id] = []
            self._by_track[inc.track_id].append(inc)

            # Index by risk level
            if inc.risk_level in self._by_level:
                self._by_level[inc.risk_level].append(inc)

            # Index by rule name
            if inc.rule_name not in self._by_rule:
                self._by_rule[inc.rule_name] = []
            self._by_rule[inc.rule_name].append(inc)

    # -------------------------------------------------------------------------
    # Core Intent Routing
    # -------------------------------------------------------------------------

    def ask(self, query: str) -> AssistantResponse:
        """
        Primary interface for operator queries.
        Parses intent, extracts entities, and returns a grounded response.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # Check for Behaviour Frequency / Trend query patterns
        freq_keywords = [
            "most frequent", "most frequently", "frequently",
            "most common", "common violation", "common behaviour", "common behavior",
            "happens the most", "happens most", "occurs the most", "occurs most",
            "occur most", "occurs most often", "occur most often", "happens most often", "happening most often",
            "behaviour frequency", "behavior frequency", "violation frequency", "incident frequency",
            "frequency of", "frequency list", "show frequency", "show behaviour frequency", "show behavior frequency",
            "recurring most often", "recurs most often", "recurrent most often", "recurring most",
            "which behaviour occurs", "which behavior occurs", "which violation occurs",
            "which handling behaviour happens", "which handling behavior happens",
        ]
        freq_patterns = [
            r"\bmost\s+frequent(?:ly)?\b",
            r"\bmost\s+common\b",
            r"\b(?:happens?|occurred?|occurs?)\s+(?:the\s+)?most\b",
            r"\b(?:occur|occurs|happens?|recurring|recurs?)\s+most\s+often\b",
            r"\b(?:behaviour|behavior|violation|incident|handling)\s+frequency\b",
            r"\bfrequency\s+(?:of|analysis|breakdown|list)\b",
            r"\bwhich\s+(?:behaviour|behavior|violation|handling)\b.*\b(?:often|common|frequent|most)\b",
        ]
        is_frequency_query = (
            any(k in q_lower for k in freq_keywords) or
            any(re.search(pat, q_lower) for pat in freq_patterns)
        )

        # Check for Location / Zone Analytics query patterns
        loc_keywords = [
            "which location", "which zone", "where do most", "location has the most",
            "zone has the most", "highest risk location", "highest risk zone", "top risk location",
            "location analytics", "zone analytics", "location breakdown", "zone breakdown",
            "incidents by location", "incidents by zone", "location trends", "zone trends",
            "incident location", "where did incidents occur", "location risk",
        ]
        loc_patterns = [
            r"\b(?:which|what)\s+(?:location|zone|bay|area)\b",
            r"\b(?:location|zone|bay)\s+(?:analytics|breakdown|distribution|trends?)\b",
            r"\bwhere\s+do\s+most\b",
        ]
        is_location_query = (
            any(k in q_lower for k in loc_keywords) or
            any(re.search(pat, q_lower) for pat in loc_patterns)
        )

        # Check for Prevention Priorities / Supervisor Focus query patterns
        priority_keywords = [
            "supervisor focus", "focus on", "focus areas", "prevention priorities",
            "prevention priority", "main prevention priorities", "top prevention priorities",
            "what should the supervisor focus on", "what should supervisor focus on",
            "what should we prioritize", "what should managers prioritize",
            "what should warehouse managers prioritize", "supervisor priority",
            "supervisor priorities", "key priorities", "mitigation priorities",
        ]
        priority_patterns = [
            r"\bwhat\s+should\s+(?:the\s+)?(?:supervisor|managers?|we|team)\s+focus\s+on\b",
            r"\b(?:prevention|supervisor|mitigation)\s+priorit(?:y|ies)\b",
            r"\bwhat\s+(?:are|is)\s+(?:the\s+)?(?:main|key|top)\s+prevention\s+priorit(?:y|ies)\b",
        ]
        is_priority_query = (
            any(k in q_lower for k in priority_keywords) or
            any(re.search(pat, q_lower) for pat in priority_patterns)
        )

        # Check for Risk Analytics & Distribution query patterns
        risk_analytics_keywords = [
            "how many critical", "how many high risk", "how many medium risk", "how many low risk",
            "how many high-risk", "how many medium-risk", "how many low-risk",
            "risk distribution", "distribution of risk", "risk breakdown", "average risk score",
            "average risk", "max risk score", "highest risk score", "breakdown of risk",
            "risk score average", "show me the risk distribution", "show the risk distribution",
        ]
        risk_analytics_patterns = [
            r"\bhow\s+many\s+(?:critical|high|medium|low)(?:-|\s+)?(?:risk)?\s+incidents?\b",
            r"\b(?:show(?:\s+me)?\s+)?(?:the\s+)?risk\s+(?:distribution|breakdown)\b",
            r"\bdistribution\s+of\s+risk\b",
            r"\baverage\s+risk\s+score\b",
        ]
        is_risk_analytics_query = (
            any(k in q_lower for k in risk_analytics_keywords) or
            any(re.search(pat, q_lower) for pat in risk_analytics_patterns)
        )

        if not self.incidents:
            if is_frequency_query:
                return self.get_behaviour_frequency()
            if is_location_query:
                return self.get_location_analytics()
            if is_priority_query:
                return self.get_prevention_priorities()
            if is_risk_analytics_query:
                return self.get_risk_distribution(query=q_clean)
            return AssistantResponse(
                answer=(
                    "⚠️ **No incident records are currently loaded in the system.**\n\n"
                    "- Please run the **Video Tracking & Risk Runner** tab, or\n"
                    "- Run `python scripts/generate_evidence.py` to generate synthetic warehouse incident data.\n\n"
                    f"{self.POLICY_DISCLAIMER}"
                ),
                intent=QueryIntent.SUMMARY,
                matched_incidents=[],
                data={"total_incidents": 0},
            )

        # 1. Check for specific Incident ID query (e.g. INC-DROP-1-F32)
        match_inc = re.search(r"\b(inc-[a-z0-9\-]+)\b", q_lower)
        if match_inc:
            target_id = match_inc.group(1)
            return self.get_incident_diagnosis(target_id)

        # 2. Check for Location / Zone Analytics query
        if is_location_query:
            return self.get_location_analytics()

        # 3. Check for Prevention Priorities / Supervisor Focus query
        if is_priority_query:
            return self.get_prevention_priorities()

        # 4. Check for Risk Distribution & Analytics query
        if is_risk_analytics_query:
            return self.get_risk_distribution(query=q_clean)

        # 5. Check for Behaviour Frequency / Trend query
        if is_frequency_query:
            return self.get_behaviour_frequency()

        # 6. Check for Repeat Violation / Tracked Object History query
        repeat_keywords = ["repeat", "chronic", "multiple times", "again", "recurring", "multiple infractions"]
        is_repeat_query = any(k in q_lower for k in repeat_keywords)

        track_match = re.search(r"(?:carton|track|object|pallet|entity|box|t)\s*#?\s*(\d+)", q_lower)
        if not track_match:
            track_match = re.search(r"#(\d+)", q_lower)

        if track_match:
            track_id = int(track_match.group(1))
            return self.get_repeat_history(track_id=track_id)
        elif is_repeat_query:
            return self.get_repeat_history(track_id=None)

        # 7. Check for Highest Risk / Critical Incidents query
        highest_risk_keywords = [
            "highest risk", "highest-risk", "top risk", "top risks", "most severe",
            "worst", "critical incident", "critical incidents", "high risk incidents",
            "highest score", "most dangerous", "severe incidents", "top 5",
        ]
        if any(k in q_lower for k in highest_risk_keywords):
            return self.get_highest_risk()

        # 8. Check for Prevention / Mitigation Recommendation query
        rec_keywords = ["prevent", "prevention", "recommend", "recommendation", "mitigate", "mitigation", "how to fix", "how do we prevent", "guideline"]
        if any(k in q_lower for k in rec_keywords):
            # Extract target rule if present
            matched_rule = self._extract_rule(q_lower)
            return self.get_recommendations(target=matched_rule)

        # 9. Check for Executive Summary / Overview query
        summary_keywords = [
            "summarize", "summary", "overview", "what happened", "status report",
            "incident count", "overall", "total incidents", "briefing", "report",
        ]
        if any(k in q_lower for k in summary_keywords):
            return self.get_summary()

        # 10. Check for Filter by Rule (e.g. "show all drops", "dragging events")
        matched_rule = self._extract_rule(q_lower)
        if matched_rule:
            return self.get_filtered(rule_name=matched_rule)

        # 11. Check for Filter by Risk Level (e.g. "show critical", "show medium risks")
        matched_level = self._extract_risk_level(q_lower)
        if matched_level:
            return self.get_filtered(risk_level=matched_level)

        # 12. Fallback for unrecognized questions
        return self.get_help_fallback(query)

    def _extract_rule(self, text: str) -> Optional[str]:
        """Matches behaviour rule names from query text."""
        for rule_name, kws in self.RULE_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in kws):
                return rule_name
        return None

    def _extract_risk_level(self, text: str) -> Optional[RiskLevel]:
        """Matches RiskLevel from query text."""
        if "critical" in text:
            return RiskLevel.CRITICAL
        if "high" in text:
            return RiskLevel.HIGH
        if "medium" in text or "moderate" in text:
            return RiskLevel.MEDIUM
        if "low" in text or "minor" in text:
            return RiskLevel.LOW
        return None

    # -------------------------------------------------------------------------
    # Grounded Query Implementations
    # -------------------------------------------------------------------------

    def get_summary(self) -> AssistantResponse:
        """Generates an executive operational summary of all recorded incidents."""
        total = len(self.incidents)
        crit = len(self._by_level[RiskLevel.CRITICAL])
        high = len(self._by_level[RiskLevel.HIGH])
        med = len(self._by_level[RiskLevel.MEDIUM])
        low = len(self._by_level[RiskLevel.LOW])

        # Rule breakdown
        rule_counts = {r: len(incs) for r, incs in self._by_rule.items()}
        rule_lines = []
        for r, c in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
            rule_disp = self._by_rule[r][0].human_readable_name if self._by_rule[r] else r
            rule_lines.append(f"- **{rule_disp}**: {c} incident(s)")

        # Chronic entities
        multi_infraction_tracks = [
            (tid, len(incs)) for tid, incs in self._by_track.items() if len(incs) > 1
        ]
        chronic_note = ""
        if multi_infraction_tracks:
            multi_str = ", ".join(f"Track #{tid} ({c} infractions)" for tid, c in multi_infraction_tracks)
            chronic_note = f"\n⚠️ **Repeat Infraction Alert**: {multi_str} require priority physical packaging QA before dispatch.\n"

        answer = (
            f"### 📋 Warehouse Handling Operations Summary\n\n"
            f"A total of **{total} handling incident(s)** have been detected and evaluated by the Risk Engine:\n\n"
            f"- 🔴 **CRITICAL Risk**: {crit}\n"
            f"- 🟠 **HIGH Risk**: {high}\n"
            f"- 🟡 **MEDIUM Risk**: {med}\n"
            f"- 🔵 **LOW Risk**: {low}\n\n"
            f"#### Breakdown by Violation Type:\n"
            + "\n".join(rule_lines) + "\n"
            + chronic_note + "\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.SUMMARY,
            matched_incidents=self.incidents,
            data={
                "total": total,
                "critical": crit,
                "high": high,
                "medium": med,
                "low": low,
                "rule_counts": rule_counts,
            },
        )

    @staticmethod
    def _short_behaviour_keyword(disp_name: str) -> str:
        """Converts human readable behaviour title into conversational gerund/noun phrase."""
        mapping = {
            "Product Dragged": "dragging",
            "Product Dropped": "dropping",
            "Product Thrown": "throwing",
            "Product Pushed": "pushing/kicking",
            "Product Rolled": "rolling",
            "Unstable Stacking": "unstable stacking",
            "Pallet Overhang": "pallet overhang",
            "Product Outside Designated Area": "placement outside designated areas",
            "Product Handled Without Required Equipment": "handling without required equipment",
            "Unsafe Loading Unloading Sequence": "unsafe loading/unloading sequences",
        }
        return mapping.get(disp_name, disp_name.lower())

    def get_behaviour_frequency(self) -> AssistantResponse:
        """
        Analyzes and ranks observed warehouse behaviours by frequency of occurrence.
        Identifies the most common handling violation and provides grounded trend breakdown.
        """
        if not self.incidents:
            return AssistantResponse(
                answer="No recorded behaviour incidents are available for frequency analysis.",
                intent=QueryIntent.BEHAVIOUR_FREQUENCY,
                matched_incidents=[],
                data={"total_incidents": 0, "frequency_ranking": []},
            )

        total_incidents = len(self.incidents)

        # Count incidents by behaviour
        behaviour_counts: Dict[str, int] = {}
        for inc in self.incidents:
            name = inc.human_readable_name or inc.rule_name.replace("_", " ").title()
            behaviour_counts[name] = behaviour_counts.get(name, 0) + 1

        # Sort behaviours by count descending, tie-breaking deterministically by name ascending
        sorted_behaviours = sorted(
            behaviour_counts.items(),
            key=lambda x: (-x[1], x[0]),
        )

        top_name, top_count = sorted_behaviours[0]
        tied_top = [name for name, count in sorted_behaviours if count == top_count]

        if len(tied_top) == 1:
            headline = f"{top_name} is the most frequently recorded behaviour with {top_count} incident{'s' if top_count != 1 else ''}."
            action_str = self._short_behaviour_keyword(top_name)
            interpretation = f"This indicates that {action_str} is the most recurring observed handling issue in the current video/session."
        else:
            tied_str = " and ".join(tied_top) if len(tied_top) == 2 else (", ".join(tied_top[:-1]) + f", and {tied_top[-1]}")
            headline = f"{tied_str} are the most frequently recorded behaviours with {top_count} incident{'s' if top_count != 1 else ''} each."
            actions = [self._short_behaviour_keyword(name) for name in tied_top]
            actions_str = " and ".join(actions) if len(actions) == 2 else (", ".join(actions[:-1]) + f", and {actions[-1]}")
            interpretation = f"This indicates that {actions_str} are the most recurring observed handling issues in the current video/session."

        freq_lines = []
        ranking_data = []
        for rank_idx, (name, count) in enumerate(sorted_behaviours, start=1):
            freq_lines.append(f"{rank_idx}. {name} — {count}")
            ranking_data.append({
                "rank": rank_idx,
                "behaviour": name,
                "count": count,
                "percentage": round((count / total_incidents) * 100.0, 1),
            })

        answer = (
            f"{headline}\n\n"
            f"Behaviour frequency:\n"
            + "\n".join(freq_lines) + "\n\n"
            f"Total incidents analyzed: {total_incidents}.\n\n"
            f"{interpretation}\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.BEHAVIOUR_FREQUENCY,
            matched_incidents=self.incidents,
            data={
                "total_incidents": total_incidents,
                "most_frequent_behaviour": top_name if len(tied_top) == 1 else tied_top,
                "highest_count": top_count,
                "frequency_ranking": ranking_data,
            },
        )

    def get_location_analytics(self) -> AssistantResponse:
        """Analyzes spatial and location data across recorded incidents."""
        analytics = self.analytics_engine.analyze()
        loc_data = analytics.location_analysis

        if loc_data == "location_data_unavailable" or not isinstance(loc_data, dict):
            answer = (
                "ℹ️ **Location/Zone Analytics Unavailable**\n\n"
                "The current incident telemetry does not contain designated location, loading bay, "
                "or zone identifiers (`location_data_unavailable`).\n\n"
                "To enable spatial heatmaps and location analytics, configure camera bay and floor zone "
                "metadata in the warehouse camera topology.\n\n"
                f"{self.POLICY_DISCLAIMER}"
            )
            return AssistantResponse(
                answer=answer,
                intent=QueryIntent.ANALYTICS_LOCATION,
                matched_incidents=[],
                data={
                    "location_data_unavailable": True,
                    "location_counts": "location_data_unavailable",
                    "highest_risk_location": "location_data_unavailable",
                },
            )

        lines = []
        for loc_name, item in loc_data.items():
            lines.append(
                f"- **{loc_name}**: **{item.incident_count} incident(s)** ({item.percentage}%) | "
                f"Avg Risk: **{item.average_risk_score:.1f}/100** | "
                f"Critical: {item.risk_distribution.get('CRITICAL', 0)}, High: {item.risk_distribution.get('HIGH', 0)}"
            )

        answer = (
            f"### 📍 Warehouse Location & Zone Risk Analytics\n\n"
            f"Highest-risk location: **{analytics.highest_risk_location}**\n\n"
            f"#### Location Incident Breakdown:\n"
            + "\n".join(lines) + "\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )
        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.ANALYTICS_LOCATION,
            matched_incidents=self.incidents,
            data={
                "location_counts": analytics.location_counts,
                "highest_risk_location": analytics.highest_risk_location,
                "location_analysis": {k: v.to_dict() for k, v in loc_data.items()},
            },
        )

    def get_prevention_priorities(self) -> AssistantResponse:
        """Synthesizes prioritized operational recommendations for warehouse supervisors."""
        analytics = self.analytics_engine.analyze()
        insights = analytics.prevention_insights

        lines = []
        for idx, ins in enumerate(insights, 1):
            severity_badge = f"[{ins.severity}]" if ins.severity else ""
            lines.append(
                f"{idx}. **{ins.title}** {severity_badge}\n"
                f"   - **Observation**: {ins.observation}\n"
                f"   - **Priority Action**: {ins.recommendation}"
            )

        answer = (
            "### 🎯 Warehouse Supervisor Focus & Prevention Priorities\n\n"
            "Based on deterministic analysis of current incident frequencies, risk severity tiers, "
            "and repeat infraction patterns, the following priority actions are recommended:\n\n"
            + "\n\n".join(lines) + "\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.PREVENTION_PRIORITIES,
            matched_incidents=self.incidents,
            data={
                "priorities": [ins.to_dict() for ins in insights],
                "total_priorities": len(insights),
            },
        )

    def get_risk_distribution(self, query: str = "") -> AssistantResponse:
        """Provides risk tier counts, percentages, and score statistics from analytics."""
        analytics = self.analytics_engine.analyze()
        r_counts = analytics.risk_counts
        r_pcts = analytics.risk_percentages
        total = analytics.total_incidents

        # Specific query highlighting (e.g. "how many critical incidents")
        specific_note = ""
        q_lower = query.lower()
        for lvl in ["critical", "high", "medium", "low"]:
            if f"how many {lvl}" in q_lower or f"{lvl} incident" in q_lower:
                lvl_upper = lvl.upper()
                cnt = r_counts.get(lvl_upper, 0)
                pct = r_pcts.get(lvl_upper, 0.0)
                specific_note = f"There {'was' if cnt == 1 else 'were'} **{cnt} {lvl_upper} risk incident{'s' if cnt != 1 else ''}** ({pct}% of total incidents).\n\n"
                break

        highest_note = ""
        if analytics.highest_risk_incident:
            h_inc = analytics.highest_risk_incident
            highest_note = f"The highest-risk incident is **`{h_inc.get('incident_id')}`** with a risk score of **{analytics.max_risk_score:.0f}/100** ({h_inc.get('human_readable_name')}).\n\n"

        answer = (
            f"### 📊 Warehouse Handling Risk Distribution\n\n"
            f"{specific_note}"
            f"Across **{total} total incident(s)** evaluated by the Risk Engine:\n\n"
            f"- 🔴 **CRITICAL Risk**: {r_counts.get('CRITICAL', 0)} ({r_pcts.get('CRITICAL', 0.0)}%)\n"
            f"- 🟠 **HIGH Risk**: {r_counts.get('HIGH', 0)} ({r_pcts.get('HIGH', 0.0)}%)\n"
            f"- 🟡 **MEDIUM Risk**: {r_counts.get('MEDIUM', 0)} ({r_pcts.get('MEDIUM', 0.0)}%)\n"
            f"- 🔵 **LOW Risk**: {r_counts.get('LOW', 0)} ({r_pcts.get('LOW', 0.0)}%)\n\n"
            f"**Average Risk Score**: {analytics.average_risk_score:.1f}/100\n"
            f"**Maximum Risk Score**: {analytics.max_risk_score:.0f}/100\n\n"
            f"{highest_note}"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.ANALYTICS_RISK,
            matched_incidents=self.incidents,
            data={
                "total_incidents": total,
                "risk_counts": r_counts,
                "risk_percentages": r_pcts,
                "average_risk_score": analytics.average_risk_score,
                "max_risk_score": analytics.max_risk_score,
                "highest_risk_incident": analytics.highest_risk_incident,
            },
        )

    def get_highest_risk(self, top_n: int = 5) -> AssistantResponse:
        """Returns the highest-risk incidents sorted by calibrated risk score descending."""
        sorted_incidents = sorted(self.incidents, key=lambda i: i.risk_score, reverse=True)
        top_incidents = sorted_incidents[:top_n]

        lines = []
        for idx, inc in enumerate(top_incidents, 1):
            badge = f"[{inc.risk_level.value}]"
            lines.append(
                f"{idx}. **{badge} `{inc.incident_id}`** — **Score {inc.risk_score:.0f}/100**\n"
                f"   - **Violation**: {inc.human_readable_name} on `{inc.display_label}`\n"
                f"   - **Timestamp**: {inc.timestamp_seconds:.2f}s (Frame #{inc.frame_idx})\n"
                f"   - **Hazard Driver**: {inc.explanation}\n"
                f"   - **Mitigation**: {inc.recommendation}"
            )

        answer = (
            f"### 🚨 Top {len(top_incidents)} Highest-Risk Handling Incidents\n\n"
            + "\n\n".join(lines) + "\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.HIGHEST_RISK,
            matched_incidents=top_incidents,
            data={"top_incidents": [i.to_dict() for i in top_incidents]},
        )

    def get_incident_diagnosis(self, incident_id: str) -> AssistantResponse:
        """Provides an in-depth diagnosis of a specific incident, its kinematics, and risk drivers."""
        key = incident_id.lower().strip()
        incident = self._by_id.get(key)

        if not incident:
            # Try prefix / partial match
            for inc_key, inc in self._by_id.items():
                if key in inc_key:
                    incident = inc
                    break

        if not incident:
            available_ids = ", ".join(f"`{i.incident_id}`" for i in self.incidents[:6])
            return AssistantResponse(
                answer=(
                    f"❌ **Incident `{incident_id}` was not found in the loaded records.**\n\n"
                    f"Available incidents include: {available_ids}"
                    + (f" and {len(self.incidents) - 6} more." if len(self.incidents) > 6 else ".")
                ),
                intent=QueryIntent.DIAGNOSIS,
                matched_incidents=[],
                data={"searched_id": incident_id},
            )

        # Extract metric lines
        metric_lines = []
        if incident.metrics:
            for m_key, m_val in incident.metrics.items():
                fmt_key = m_key.replace("_", " ").title()
                if isinstance(m_val, float):
                    metric_lines.append(f"- **{fmt_key}**: {m_val:.1f}")
                else:
                    metric_lines.append(f"- **{fmt_key}**: {m_val}")
        else:
            metric_lines.append("- No secondary kinematics recorded.")

        # Check repeat status for this track
        track_incidents = self._by_track.get(incident.track_id, [])
        infraction_order = 1
        for i_idx, t_inc in enumerate(track_incidents, 1):
            if t_inc.incident_id == incident.incident_id:
                infraction_order = i_idx
                break

        repeat_context = f"Infraction #{infraction_order} of {len(track_incidents)} recorded on {incident.display_label}."
        if len(track_incidents) > 1:
            repeat_context += " Repeat infraction penalty was applied to score."

        answer = (
            f"### 🔍 Diagnosis for Incident `{incident.incident_id}`\n\n"
            f"- **Classification**: **{incident.risk_level.value} Risk** (Calibrated Score: **{incident.risk_score:.0f}/100**)\n"
            f"- **Observed Behaviour**: {incident.human_readable_name} (`{incident.rule_name}`)\n"
            f"- **Target Entity**: `{incident.display_label}` (Class: {incident.class_name}, Detection Confidence: {incident.confidence*100:.1f}%)\n"
            f"- **Timestamp**: {incident.timestamp_seconds:.2f}s | Frame #{incident.frame_idx}\n"
            f"- **Repeat Context**: {repeat_context}\n"
            f"- **Damage Assessment**: `{incident.damage_assessment.value}` ({incident.damage_assessment.human_readable})\n\n"
            f"#### 📊 Telemetry & Kinematic Evidence:\n"
            + "\n".join(metric_lines) + "\n\n"
            f"#### 💡 Why This Score Was Assigned:\n"
            f"> {incident.explanation}\n\n"
            f"#### 🛡️ Prevention Recommendation:\n"
            f"> {incident.recommendation}\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.DIAGNOSIS,
            matched_incidents=[incident],
            data={"incident": incident.to_dict()},
        )

    def get_repeat_history(self, track_id: Optional[int] = None) -> AssistantResponse:
        """Summarizes repeat violations and chronic infractions on tracked entities."""
        if track_id is not None:
            track_incidents = self._by_track.get(track_id, [])
            if not track_incidents:
                available_tracks = ", ".join(f"Track #{tid}" for tid in sorted(self._by_track.keys()))
                return AssistantResponse(
                    answer=(
                        f"ℹ️ **No incidents found for Track #{track_id}.**\n\n"
                        f"Active tracked entities with recorded incidents: {available_tracks}."
                    ),
                    intent=QueryIntent.REPEAT_HISTORY,
                    matched_incidents=[],
                    data={"track_id": track_id},
                )

            disp_label = track_incidents[0].display_label
            lines = []
            for i_idx, inc in enumerate(track_incidents, 1):
                penalty_note = ""
                if i_idx == 2:
                    penalty_note = " (+10 pts 2nd infraction penalty)"
                elif i_idx >= 3:
                    penalty_note = f" (+20 pts chronic #{i_idx} penalty)"

                lines.append(
                    f"{i_idx}. **`{inc.incident_id}`** — **[{inc.risk_level.value}] {inc.human_readable_name}** "
                    f"(Score: {inc.risk_score:.0f}/100{penalty_note}) @ {inc.timestamp_seconds:.2f}s (Frame #{inc.frame_idx})\n"
                    f"   - *Explanation*: {inc.explanation}"
                )

            chronic_warning = ""
            if len(track_incidents) >= 3:
                chronic_warning = (
                    "\n🚨 **CHRONIC OFFENDER ALERT**: This object has accumulated 3 or more handling violations. "
                    "Mandatory quarantine and pre-dispatch physical QA packaging inspection is required.\n"
                )

            answer = (
                f"### 🔁 Repeat Violation History for {disp_label} (Track #{track_id})\n\n"
                f"Recorded **{len(track_incidents)} handling incident(s)** involving this entity:\n\n"
                + "\n".join(lines) + "\n"
                + chronic_warning + "\n"
                f"#### Applicable Prevention Guidelines:\n"
                f"> {track_incidents[-1].recommendation}\n\n"
                f"{self.POLICY_DISCLAIMER}"
            )

            return AssistantResponse(
                answer=answer,
                intent=QueryIntent.REPEAT_HISTORY,
                matched_incidents=track_incidents,
                data={"track_id": track_id, "incidents": [i.to_dict() for i in track_incidents]},
            )

        # General repeat violation overview across all tracks
        multi_tracks = {tid: incs for tid, incs in self._by_track.items() if len(incs) > 1}
        if not multi_tracks:
            return AssistantResponse(
                answer=(
                    "✅ **No repeat handling violations detected across the recorded entities.**\n"
                    "Each detected violation occurred on a separate tracked object with no recurring infractions."
                ),
                intent=QueryIntent.REPEAT_HISTORY,
                matched_incidents=[],
                data={"multi_infraction_tracks": []},
            )

        sections = []
        all_multi_incidents = []
        for tid, incs in sorted(multi_tracks.items(), key=lambda x: len(x[1]), reverse=True):
            all_multi_incidents.extend(incs)
            disp = incs[0].display_label
            max_score = max(i.risk_score for i in incs)
            highest_level = max(incs, key=lambda i: i.risk_score).risk_level.value
            rule_list = ", ".join(i.human_readable_name for i in incs)
            sections.append(
                f"- **{disp} (Track #{tid})**: **{len(incs)} incidents** | Highest Risk: **{highest_level} ({max_score:.0f}/100)**\n"
                f"  - Violations: {rule_list}\n"
                f"  - Recommendation: {incs[-1].recommendation}"
            )

        answer = (
            f"### 🔁 Chronic Repeat Violation Summary\n\n"
            f"Found **{len(multi_tracks)} tracked entity/entities** with multiple recorded violations:\n\n"
            + "\n".join(sections) + "\n\n"
            f"Entities with >= 3 violations are automatically escalated and flagged for mandatory quarantine inspection.\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.REPEAT_HISTORY,
            matched_incidents=all_multi_incidents,
            data={"multi_tracks_count": len(multi_tracks)},
        )

    def get_recommendations(self, target: Optional[str] = None) -> AssistantResponse:
        """Returns actionable damage prevention and handling guidelines."""
        # If target rule matches
        if target and target in self.config.recommendations:
            rec = self.config.recommendations[target]
            rule_disp = target.replace("_", " ").title()
            matched = self._by_rule.get(target, [])

            answer = (
                f"### 🛡️ Prevention Recommendation: {rule_disp}\n\n"
                f"> **Operational Guideline**: {rec}\n\n"
                f"Currently **{len(matched)} incident(s)** of this type are recorded in the session.\n\n"
                f"{self.POLICY_DISCLAIMER}"
            )
            return AssistantResponse(
                answer=answer,
                intent=QueryIntent.RECOMMENDATIONS,
                matched_incidents=matched,
                data={"rule": target, "recommendation": rec},
            )

        # General prevention guidelines across all observed rules
        lines = []
        for rule_name, rec in self.config.recommendations.items():
            disp = rule_name.replace("_", " ").title()
            occ = len(self._by_rule.get(rule_name, []))
            lines.append(f"- **{disp}** ({occ} recorded):\n  > {rec}")

        answer = (
            "### 🛡️ Warehouse Handling Prevention Guidelines\n\n"
            + "\n\n".join(lines) + "\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )

        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.RECOMMENDATIONS,
            matched_incidents=self.incidents,
            data={"recommendations": self.config.recommendations},
        )

    def get_filtered(
        self,
        rule_name: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
    ) -> AssistantResponse:
        """Filters incidents by behaviour rule or risk tier."""
        if rule_name:
            matches = self._by_rule.get(rule_name, [])
            disp_name = rule_name.replace("_", " ").title()
            if not matches:
                return AssistantResponse(
                    answer=f"ℹ️ No incidents found for violation rule: **{disp_name}**.",
                    intent=QueryIntent.FILTER_RULES,
                    matched_incidents=[],
                    data={"rule_name": rule_name},
                )

            lines = [
                f"- **`{i.incident_id}`** ({i.risk_level.value} | {i.risk_score:.0f} pts): {i.display_label} @ {i.timestamp_seconds:.2f}s"
                for i in matches
            ]
            answer = (
                f"### 🔍 Incidents matching violation: **{disp_name}** ({len(matches)} found)\n\n"
                + "\n".join(lines) + "\n\n"
                f"**Recommendation**: {matches[0].recommendation}\n\n"
                f"{self.POLICY_DISCLAIMER}"
            )
            return AssistantResponse(
                answer=answer,
                intent=QueryIntent.FILTER_RULES,
                matched_incidents=matches,
                data={"rule_name": rule_name, "count": len(matches)},
            )

        if risk_level:
            matches = self._by_level.get(risk_level, [])
            if not matches:
                return AssistantResponse(
                    answer=f"ℹ️ No incidents found with risk level: **{risk_level.value}**.",
                    intent=QueryIntent.FILTER_RISK,
                    matched_incidents=[],
                    data={"risk_level": risk_level.value},
                )

            lines = [
                f"- **`{i.incident_id}`** ({i.human_readable_name} | {i.risk_score:.0f} pts): {i.display_label} @ {i.timestamp_seconds:.2f}s"
                for i in matches
            ]
            answer = (
                f"### 🔍 Incidents with **{risk_level.value} Risk** ({len(matches)} found)\n\n"
                + "\n".join(lines) + "\n\n"
                f"{self.POLICY_DISCLAIMER}"
            )
            return AssistantResponse(
                answer=answer,
                intent=QueryIntent.FILTER_RISK,
                matched_incidents=matches,
                data={"risk_level": risk_level.value, "count": len(matches)},
            )

        return self.get_summary()

    def get_help_fallback(self, query: str = "") -> AssistantResponse:
        """Returns helpful guidance when the query intent cannot be resolved."""
        sample_id = self.incidents[0].incident_id if self.incidents else "INC-DROP-1-F32"
        answer = (
            "🤖 **Godrej Operations Assistant — Grounded Query Guide**\n\n"
            "I provide factual answers strictly grounded in verified incident manifest telemetry. "
            "Here are sample questions you can ask:\n\n"
            "- 📊 **Summary**: *\"Summarize current incidents\"* or *\"What happened today?\"*\n"
            "- 🚨 **Highest Risk**: *\"What are the highest-risk incidents?\"* or *\"Show critical incidents\"*\n"
            f"- 🔍 **Diagnosis**: *\"Why was {sample_id} classified as critical?\"* or *\"Explain {sample_id}\"*\n"
            "- 🔁 **Repeat Infractions**: *\"What happened to Carton #7?\"* or *\"Which product had repeated violations?\"*\n"
            "- 📈 **Frequency & Trends**: *\"Which behaviour occurs most frequently?\"* or *\"What is the most common violation?\"*\n"
            "- 📊 **Risk Distribution**: *\"Show me the risk distribution\"* or *\"How many critical incidents occurred?\"*\n"
            "- 📍 **Location Analytics**: *\"Which location has the most incidents?\"* or *\"Location analytics\"*\n"
            "- 🎯 **Supervisor Priorities**: *\"What should the supervisor focus on?\"* or *\"What are the main prevention priorities?\"*\n"
            "- 🛡️ **Prevention**: *\"What prevention recommendations apply?\"* or *\"How do we prevent drops?\"*\n"
            "- 🔍 **Filter**: *\"Show all dragged cartons\"* or *\"List high risk incidents\"*\n\n"
            f"{self.POLICY_DISCLAIMER}"
        )
        return AssistantResponse(
            answer=answer,
            intent=QueryIntent.FALLBACK,
            matched_incidents=[],
            data={"unmatched_query": query},
        )
