"""
Warehouse Risk & Behaviour Analytics Engine.
Processes verified IncidentRecord streams or manifests into warehouse-level
operational intelligence: behaviour frequency distributions, risk concentration,
repeat entity tracking, zone analytics, temporal clustering, and targeted prevention insights.
Strictly adheres to Godrej Responsible AI damage assessment principles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.analytics.models import (
    BehaviourFrequencyItem,
    LocationAnalyticsItem,
    PreventionInsight,
    RepeatTrackItem,
    SUPPORTED_BEHAVIOURS,
    TemporalCluster,
    WarehouseAnalytics,
)
from app.risk.models import DamageAssessmentStatus, IncidentRecord, RiskLevel


# Mapping of behaviour rule identifiers to conversational display titles and specific prevention guidelines
BEHAVIOUR_PREVENTION_MAP: Dict[str, Dict[str, str]] = {
    "product_dragged": {
        "display": "Product Dragged",
        "action": "dragging products along the floor",
        "guideline": (
            "Prioritize warehouse worker training on proper two-person lifting techniques and "
            "enforce the mandatory use of hand trolleys or pallet jacks for ground transit."
        ),
    },
    "product_dropped": {
        "display": "Product Dropped",
        "action": "product drops and unassisted freefalls",
        "guideline": (
            "Reinforce two-handed lifting compliance, inspect carton slip-resistance, and verify "
            "adequate grip gear to prevent handling drops during manual loading."
        ),
    },
    "product_thrown": {
        "display": "Product Thrown",
        "action": "throwing or tossing cartons",
        "guideline": (
            "Mandate immediate physical QA audit of tossed cartons and review transfer zone layout "
            "to eliminate worker tossing between stations."
        ),
    },
    "product_pushed": {
        "display": "Product Pushed",
        "action": "pushing or kicking cartons across the floor",
        "guideline": (
            "Verify packaging lower corner integrity, eliminate floor pushing habits, and ensure "
            "designated walkways remain unobstructed."
        ),
    },
    "product_rolled": {
        "display": "Product Rolled",
        "action": "rolling cartons end-over-end",
        "guideline": (
            "Inspect internal product orientation and seals for crushing; handle cylindrical or bulky "
            "items with specialized cradles or clamp trucks."
        ),
    },
    "unstable_stacking": {
        "display": "Unstable Stacking",
        "action": "unstable column stacking",
        "guideline": (
            "Enforce column center-of-mass alignment within 15% base margin, verify maximum tier limits, "
            "and apply stretch wrap reinforcement immediately after palletizing."
        ),
    },
    "pallet_overhang": {
        "display": "Pallet Overhang",
        "action": "pallet perimeter overhang",
        "guideline": (
            "Reposition cartons flush within pallet edges to eliminate perimeter overhang exceeding 5%, "
            "preventing forklift shearing and storage rack snagging."
        ),
    },
    "product_outside_designated_area": {
        "display": "Product Outside Designated Area",
        "action": "placing products outside designated zones",
        "guideline": (
            "Inspect floor demarcations, reposition misplaced merchandise into designated pallet bays, "
            "and ensure main transit aisles remain fully clear."
        ),
    },
    "product_handled_without_required_equipment": {
        "display": "Product Handled Without Required Equipment",
        "action": "manual handling without required lifting equipment",
        "guideline": (
            "Ensure designated mechanical handling aids (forklifts, pallet jacks, lift tables) are "
            "accessible and enforced for heavy or oversized merchandise."
        ),
    },
    "unsafe_loading_unloading_sequence": {
        "display": "Unsafe Loading Unloading Sequence",
        "action": "unsafe loading/unloading sequence execution",
        "guideline": (
            "Enforce prescribed loading sequence SOPs and verify that products are safely positioned "
            "and stabilized before initiating subsequent transfer actions."
        ),
    },
}


class AnalyticsEngine:
    """
    Computes operational warehouse intelligence from material handling incidents.
    Provides deterministic, explainable aggregation across behaviours, risk levels,
    repeat entities, locations, and temporal patterns.
    """

    def __init__(self, incidents: Optional[List[IncidentRecord]] = None):
        self.incidents: List[IncidentRecord] = list(incidents) if incidents else []

    @classmethod
    def from_manifest(cls, manifest_path: Union[str, Path]) -> AnalyticsEngine:
        """Loads incidents from a serialized JSON manifest file on disk."""
        path = Path(manifest_path)
        if not path.exists():
            return cls([])

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return cls([])

        incidents: List[IncidentRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            level_str = str(item.get("risk_level", "MEDIUM")).upper()
            try:
                risk_level = RiskLevel(level_str)
            except ValueError:
                risk_level = RiskLevel.MEDIUM

            status_str = str(item.get("damage_assessment", "POTENTIAL_DAMAGE_RISK"))
            try:
                damage_status = DamageAssessmentStatus(status_str)
            except ValueError:
                damage_status = DamageAssessmentStatus.POTENTIAL_DAMAGE_RISK

            raw_bbox = item.get("bbox", [0, 0, 0, 0])
            bbox = tuple(raw_bbox) if len(raw_bbox) == 4 else (0, 0, 0, 0)

            incidents.append(
                IncidentRecord(
                    incident_id=str(item.get("incident_id", "")),
                    event_id=str(item.get("event_id", "")),
                    rule_name=str(item.get("rule_name", "")),
                    human_readable_name=str(item.get("human_readable_name", "")),
                    risk_level=risk_level,
                    risk_score=float(item.get("risk_score", 0.0)),
                    timestamp_seconds=float(item.get("timestamp_seconds", 0.0)),
                    frame_idx=int(item.get("frame_idx", 0)),
                    track_id=int(item.get("track_id", 0)),
                    display_label=str(item.get("display_label", f"Object #{item.get('track_id', 0)}")),
                    secondary_track_id=item.get("secondary_track_id"),
                    secondary_label=item.get("secondary_label"),
                    class_name=str(item.get("class_name", "carton")),
                    confidence=float(item.get("confidence", 0.8)),
                    bbox=bbox,
                    metrics=item.get("metrics", {}) or {},
                    damage_assessment=damage_status,
                    explanation=str(item.get("explanation", "")),
                    recommendation=str(item.get("recommendation", "")),
                    evidence_image_path=item.get("evidence_image_path"),
                    video_clip_path=item.get("video_clip_path"),
                )
            )

        return cls(incidents)

    def analyze(self) -> WarehouseAnalytics:
        """
        Executes complete warehouse risk and behaviour intelligence analysis.
        Returns a standardized WarehouseAnalytics result.
        """
        total = len(self.incidents)

        # 1. Behaviour Frequency Analysis
        b_counts, b_pcts, most_freq, least_freq = self._analyze_behaviours(self.incidents, total)

        # 2. Risk Distribution Analysis
        r_counts, r_pcts, avg_score, max_score, highest_inc = self._analyze_risk(self.incidents, total)

        # 3. Repeat Violations Analysis
        repeat_tracks, total_repeats, highest_repeat = self._analyze_repeats(self.incidents)

        # 4. Location / Zone Analytics
        loc_counts, highest_loc, loc_analysis = self._analyze_locations(self.incidents, total)

        # 5. Temporal Analytics & Clustering
        temporal_analysis = self._analyze_temporal(self.incidents)

        # 6. Actionable Prevention Insights
        insights = self._generate_prevention_insights(
            total=total,
            behaviour_counts=b_counts,
            behaviour_percentages=b_pcts,
            most_frequent_behaviours=most_freq,
            risk_counts=r_counts,
            repeat_tracks=repeat_tracks,
            highest_repeat_track=highest_repeat,
            location_analysis=loc_analysis,
            highest_risk_location=highest_loc,
            temporal_analysis=temporal_analysis,
        )

        return WarehouseAnalytics(
            total_incidents=total,
            behaviour_counts=b_counts,
            behaviour_percentages=b_pcts,
            most_frequent_behaviours=most_freq,
            least_frequent_behaviours=least_freq,
            risk_counts=r_counts,
            risk_percentages=r_pcts,
            average_risk_score=avg_score,
            max_risk_score=max_score,
            highest_risk_incident=highest_inc.to_dict() if highest_inc else None,
            repeat_tracks=repeat_tracks,
            total_repeat_incidents=total_repeats,
            highest_repeat_track=highest_repeat,
            location_counts=loc_counts,
            highest_risk_location=highest_loc,
            location_analysis=loc_analysis,
            temporal_analysis=temporal_analysis,
            prevention_insights=insights,
        )

    # -------------------------------------------------------------------------
    # Internal Analysis Routines
    # -------------------------------------------------------------------------

    @staticmethod
    def _analyze_behaviours(
        incidents: List[IncidentRecord], total: int
    ) -> Tuple[Dict[str, int], Dict[str, float], List[str], List[str]]:
        """Groups incidents by behaviour, calculates percentages, and identifies frequency extremes."""
        if total == 0:
            return {}, {}, [], []

        # Count occurrences by rule_name
        counts: Dict[str, int] = {}
        for inc in incidents:
            rule = inc.rule_name
            counts[rule] = counts.get(rule, 0) + 1

        percentages: Dict[str, float] = {
            r: round((c / total) * 100.0, 2) for r, c in counts.items()
        }

        # Identify frequency extremes with deterministic tie-breaking (sorted alphabetically)
        max_count = max(counts.values())
        min_count = min(counts.values())

        most_frequent = sorted([r for r, c in counts.items() if c == max_count])
        least_frequent = sorted([r for r, c in counts.items() if c == min_count])

        return counts, percentages, most_frequent, least_frequent

    @staticmethod
    def _analyze_risk(
        incidents: List[IncidentRecord], total: int
    ) -> Tuple[Dict[str, int], Dict[str, float], float, float, Optional[IncidentRecord]]:
        """Calculates risk level distribution, calibrated score statistics, and highest risk event."""
        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if total == 0:
            return (
                {lvl: 0 for lvl in levels},
                {lvl: 0.0 for lvl in levels},
                0.0,
                0.0,
                None,
            )

        counts: Dict[str, int] = {lvl: 0 for lvl in levels}
        total_score = 0.0
        max_score = 0.0
        highest_incident: Optional[IncidentRecord] = None

        for inc in incidents:
            lvl_val = inc.risk_level.value if isinstance(inc.risk_level, RiskLevel) else str(inc.risk_level)
            if lvl_val in counts:
                counts[lvl_val] += 1
            else:
                counts[lvl_val] = 1

            score = inc.risk_score
            total_score += score
            if score > max_score or highest_incident is None:
                max_score = score
                highest_incident = inc
            elif score == max_score and highest_incident is not None:
                # Deterministic tie-break by frame index, then incident ID
                if (inc.frame_idx, inc.incident_id) < (highest_incident.frame_idx, highest_incident.incident_id):
                    highest_incident = inc

        percentages: Dict[str, float] = {
            lvl: round((counts.get(lvl, 0) / total) * 100.0, 2) for lvl in levels
        }
        avg_score = round(total_score / total, 2)

        return counts, percentages, avg_score, max_score, highest_incident

    @staticmethod
    def _analyze_repeats(
        incidents: List[IncidentRecord]
    ) -> Tuple[List[RepeatTrackItem], int, Optional[RepeatTrackItem]]:
        """Identifies tracked entities with multiple recorded violations without assuming human identity."""
        if not incidents:
            return [], 0, None

        by_track: Dict[int, List[IncidentRecord]] = {}
        for inc in incidents:
            by_track.setdefault(inc.track_id, []).append(inc)

        repeat_items: List[RepeatTrackItem] = []
        total_repeat_incidents = 0

        for tid, t_incs in by_track.items():
            if len(t_incs) > 1:
                # Chronological sorting
                sorted_t = sorted(t_incs, key=lambda x: (x.timestamp_seconds, x.frame_idx))
                disp = sorted_t[0].display_label
                rule_list = list(dict.fromkeys(i.rule_name for i in sorted_t))
                inc_ids = [i.incident_id for i in sorted_t]
                max_risk_lvl = max(sorted_t, key=lambda i: i.risk_score).risk_level.value
                avg_score = round(sum(i.risk_score for i in sorted_t) / len(sorted_t), 2)
                cnt = len(sorted_t)
                total_repeat_incidents += cnt

                repeat_items.append(
                    RepeatTrackItem(
                        track_id=tid,
                        display_label=disp,
                        incident_count=cnt,
                        behaviours=rule_list,
                        incident_ids=inc_ids,
                        max_risk_level=max_risk_lvl,
                        average_risk_score=avg_score,
                    )
                )

        # Sort repeat tracks primarily by incident count descending, secondarily by track_id ascending
        repeat_items.sort(key=lambda item: (-item.incident_count, item.track_id))

        highest_repeat = repeat_items[0] if repeat_items else None
        return repeat_items, total_repeat_incidents, highest_repeat

    @staticmethod
    def _extract_location(inc: IncidentRecord) -> Optional[str]:
        """Extracts spatial or zone metadata from an incident without inventing locations."""
        metrics = inc.metrics or {}
        candidate_keys = [
            "location",
            "zone",
            "zone_id",
            "location_name",
            "bay",
            "loading_bay",
            "area",
            "workstation",
        ]
        for k in candidate_keys:
            val = metrics.get(k)
            if isinstance(val, str) and val.strip():
                val_clean = val.strip()
                # Ignore coordinate box representations like "[100, 150, 800, 650]"
                if not (val_clean.startswith("[") or val_clean.startswith("(")):
                    return val_clean

        for attr in ["location", "zone", "loading_bay"]:
            val = getattr(inc, attr, None)
            if isinstance(val, str) and val.strip():
                val_clean = val.strip()
                if not (val_clean.startswith("[") or val_clean.startswith("(")):
                    return val_clean

        return None

    @classmethod
    def _analyze_locations(
        cls, incidents: List[IncidentRecord], total: int
    ) -> Tuple[
        Union[Dict[str, int], str],
        str,
        Union[Dict[str, LocationAnalyticsItem], str],
    ]:
        """
        Groups incidents by location/zone when present in telemetry.
        If spatial information is unavailable, explicitly returns 'location_data_unavailable'.
        Never fabricates locations.
        """
        if total == 0:
            return "location_data_unavailable", "location_data_unavailable", "location_data_unavailable"

        by_loc: Dict[str, List[IncidentRecord]] = {}
        for inc in incidents:
            loc = cls._extract_location(inc)
            if loc:
                by_loc.setdefault(loc, []).append(inc)

        # If zero incidents provide location data, return the explicit sentinel
        if not by_loc:
            return (
                "location_data_unavailable",
                "location_data_unavailable",
                "location_data_unavailable",
            )

        loc_counts: Dict[str, int] = {}
        loc_analysis: Dict[str, LocationAnalyticsItem] = {}
        highest_score = -1.0
        highest_risk_loc: Optional[str] = None

        for loc_name, loc_incs in by_loc.items():
            cnt = len(loc_incs)
            loc_counts[loc_name] = cnt
            pct = round((cnt / total) * 100.0, 2)
            r_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            total_loc_score = 0.0

            for inc in loc_incs:
                lvl = inc.risk_level.value
                r_dist[lvl] = r_dist.get(lvl, 0) + 1
                total_loc_score += inc.risk_score

            avg_loc_score = round(total_loc_score / cnt, 2)
            loc_analysis[loc_name] = LocationAnalyticsItem(
                location=loc_name,
                incident_count=cnt,
                percentage=pct,
                risk_distribution=r_dist,
                average_risk_score=avg_loc_score,
            )

            # Determine highest risk location by average risk score, tie-break alphabetically
            if avg_loc_score > highest_score:
                highest_score = avg_loc_score
                highest_risk_loc = loc_name
            elif avg_loc_score == highest_score and highest_risk_loc is not None:
                if loc_name < highest_risk_loc:
                    highest_risk_loc = loc_name

        return loc_counts, highest_risk_loc or "location_data_unavailable", loc_analysis

    @staticmethod
    def _analyze_temporal(incidents: List[IncidentRecord]) -> Dict[str, Any]:
        """Performs temporal ordering, timeline mapping, and high-density cluster detection."""
        if not incidents:
            return {
                "timeline": [],
                "clusters": [],
                "by_behaviour": {},
                "by_risk_level": {},
                "duration_seconds": 0.0,
            }

        sorted_incs = sorted(incidents, key=lambda x: (x.timestamp_seconds, x.frame_idx))
        duration = round(
            sorted_incs[-1].timestamp_seconds - sorted_incs[0].timestamp_seconds, 2
        )

        timeline = [
            {
                "incident_id": i.incident_id,
                "timestamp_seconds": i.timestamp_seconds,
                "frame_idx": i.frame_idx,
                "rule_name": i.rule_name,
                "human_readable_name": i.human_readable_name,
                "risk_level": i.risk_level.value,
                "risk_score": i.risk_score,
                "display_label": i.display_label,
            }
            for i in sorted_incs
        ]

        by_behaviour: Dict[str, List[float]] = {}
        by_risk: Dict[str, List[float]] = {}
        for i in sorted_incs:
            by_behaviour.setdefault(i.rule_name, []).append(i.timestamp_seconds)
            by_risk.setdefault(i.risk_level.value, []).append(i.timestamp_seconds)

        # Detect temporal clusters within a 5-second window
        clusters: List[Dict[str, Any]] = []
        cluster_window = 5.0
        current_cluster: List[IncidentRecord] = []

        for inc in sorted_incs:
            if not current_cluster:
                current_cluster.append(inc)
            else:
                if inc.timestamp_seconds - current_cluster[0].timestamp_seconds <= cluster_window:
                    current_cluster.append(inc)
                else:
                    if len(current_cluster) >= 2:
                        b_counts = {}
                        r_counts = {}
                        for c_inc in current_cluster:
                            b_counts[c_inc.rule_name] = b_counts.get(c_inc.rule_name, 0) + 1
                            r_counts[c_inc.risk_level.value] = r_counts.get(c_inc.risk_level.value, 0) + 1
                        dom_b = max(b_counts.items(), key=lambda x: x[1])[0]
                        dom_r = max(r_counts.items(), key=lambda x: x[1])[0]
                        clusters.append(
                            TemporalCluster(
                                cluster_id=len(clusters) + 1,
                                start_time=round(current_cluster[0].timestamp_seconds, 2),
                                end_time=round(current_cluster[-1].timestamp_seconds, 2),
                                start_frame=current_cluster[0].frame_idx,
                                end_frame=current_cluster[-1].frame_idx,
                                incident_count=len(current_cluster),
                                incident_ids=[c.incident_id for c in current_cluster],
                                dominant_behaviour=dom_b,
                                dominant_risk_level=dom_r,
                            ).to_dict()
                        )
                    current_cluster = [inc]

        if len(current_cluster) >= 2:
            b_counts = {}
            r_counts = {}
            for c_inc in current_cluster:
                b_counts[c_inc.rule_name] = b_counts.get(c_inc.rule_name, 0) + 1
                r_counts[c_inc.risk_level.value] = r_counts.get(c_inc.risk_level.value, 0) + 1
            dom_b = max(b_counts.items(), key=lambda x: x[1])[0]
            dom_r = max(r_counts.items(), key=lambda x: x[1])[0]
            clusters.append(
                TemporalCluster(
                    cluster_id=len(clusters) + 1,
                    start_time=round(current_cluster[0].timestamp_seconds, 2),
                    end_time=round(current_cluster[-1].timestamp_seconds, 2),
                    start_frame=current_cluster[0].frame_idx,
                    end_frame=current_cluster[-1].frame_idx,
                    incident_count=len(current_cluster),
                    incident_ids=[c.incident_id for c in current_cluster],
                    dominant_behaviour=dom_b,
                    dominant_risk_level=dom_r,
                ).to_dict()
            )

        return {
            "timeline": timeline,
            "clusters": clusters,
            "by_behaviour": by_behaviour,
            "by_risk_level": by_risk,
            "duration_seconds": duration,
        }

    @staticmethod
    def _generate_prevention_insights(
        total: int,
        behaviour_counts: Dict[str, int],
        behaviour_percentages: Dict[str, float],
        most_frequent_behaviours: List[str],
        risk_counts: Dict[str, int],
        repeat_tracks: List[RepeatTrackItem],
        highest_repeat_track: Optional[RepeatTrackItem],
        location_analysis: Union[Dict[str, LocationAnalyticsItem], str],
        highest_risk_location: str,
        temporal_analysis: Dict[str, Any],
    ) -> List[PreventionInsight]:
        """
        Synthesizes deterministic, explainable prevention recommendations from aggregate patterns.
        Strictly frames observations as Potential Damage Risk rather than confirmed damage.
        """
        if total == 0:
            return [
                PreventionInsight(
                    category="GENERAL",
                    title="No Incidents Recorded",
                    observation="Zero material handling violations were recorded in the session.",
                    recommendation="Continue maintaining standard operational safety and warehouse handling procedures.",
                    severity="INFO",
                )
            ]

        insights: List[PreventionInsight] = []

        # 1. Primary Behaviour Frequency Recommendation
        if most_frequent_behaviours:
            if len(most_frequent_behaviours) == 1:
                top_rule = most_frequent_behaviours[0]
                top_meta = BEHAVIOUR_PREVENTION_MAP.get(top_rule, {})
                disp_title = top_meta.get("display", top_rule.replace("_", " ").title())
                guideline = top_meta.get(
                    "guideline",
                    f"Prioritize supervisor review and operator training to mitigate {top_rule}.",
                )
                cnt = behaviour_counts.get(top_rule, 0)
                pct = behaviour_percentages.get(top_rule, 0.0)

                insights.append(
                    PreventionInsight(
                        category="BEHAVIOUR_TREND",
                        title=f"Prioritize Mitigation for {disp_title}",
                        observation=(
                            f"{disp_title} is the most frequent handling violation, accounting for "
                            f"{cnt} incident(s) ({pct}% of all recorded events)."
                        ),
                        recommendation=guideline,
                        severity="HIGH" if pct >= 30.0 else "MEDIUM",
                    )
                )
            else:
                # Handle ties deterministically
                disp_names = [
                    BEHAVIOUR_PREVENTION_MAP.get(r, {}).get("display", r.replace("_", " ").title())
                    for r in most_frequent_behaviours
                ]
                top_cnt = behaviour_counts.get(most_frequent_behaviours[0], 0)
                names_str = " and ".join(disp_names) if len(disp_names) == 2 else (", ".join(disp_names[:-1]) + f", and {disp_names[-1]}")
                tied_guidelines = " ".join(
                    BEHAVIOUR_PREVENTION_MAP.get(r, {}).get("guideline", "")
                    for r in most_frequent_behaviours
                ).strip()

                insights.append(
                    PreventionInsight(
                        category="BEHAVIOUR_TREND",
                        title=f"Multiple Co-Dominant Violations: {names_str}",
                        observation=(
                            f"{names_str} are tied as the most frequent handling violations with "
                            f"{top_cnt} incidents each."
                        ),
                        recommendation=tied_guidelines or "Conduct joint training on these recurring handling issues.",
                        severity="HIGH",
                    )
                )

        # 2. Risk Severity & Escalation Insight
        crit_count = risk_counts.get("CRITICAL", 0)
        high_count = risk_counts.get("HIGH", 0)
        severe_total = crit_count + high_count
        severe_pct = round((severe_total / total) * 100.0, 1)

        if severe_pct >= 50.0:
            insights.append(
                PreventionInsight(
                    category="RISK_SEVERITY",
                    title="Urgent: High/Critical Risk Incidents Dominate Session",
                    observation=(
                        f"Elevated risk tiers (CRITICAL: {crit_count}, HIGH: {high_count}) constitute "
                        f"{severe_pct}% of all evaluated handling infractions."
                    ),
                    recommendation=(
                        "Initiate immediate supervisor floor review and equipment inspection. "
                        "Enforce pre-dispatch packaging audits to prevent transit failures."
                    ),
                    severity="CRITICAL",
                )
            )

        # 3. Repeat Infraction Entity Insight
        if repeat_tracks:
            highest_disp = highest_repeat_track.display_label if highest_repeat_track else f"Track #{repeat_tracks[0].track_id}"
            highest_cnt = highest_repeat_track.incident_count if highest_repeat_track else repeat_tracks[0].incident_count
            insights.append(
                PreventionInsight(
                    category="REPEAT_VIOLATION",
                    title=f"Repeat Infractions Flagged Across {len(repeat_tracks)} Entities",
                    observation=(
                        f"{len(repeat_tracks)} tracked object(s) accumulated multiple infractions. "
                        f"Most recurring: {highest_disp} with {highest_cnt} violations."
                    ),
                    recommendation=(
                        f"Quarantine {highest_disp} and all multi-infraction items for mandatory physical QA inspection. "
                        "Investigate whether workstation layout or conveyor transit caused repeat stress."
                    ),
                    severity="HIGH" if highest_cnt >= 3 else "MEDIUM",
                )
            )

        # 4. Location / Zone Concentration Insight (only when location data exists)
        if isinstance(location_analysis, dict) and location_analysis:
            if highest_risk_location and highest_risk_location != "location_data_unavailable":
                top_loc_item = location_analysis.get(highest_risk_location)
                if top_loc_item:
                    insights.append(
                        PreventionInsight(
                            category="LOCATION_CONCENTRATION",
                            title=f"Risk Concentration in Zone '{highest_risk_location}'",
                            observation=(
                                f"Zone '{highest_risk_location}' recorded {top_loc_item.incident_count} incident(s) "
                                f"({top_loc_item.percentage}%) with an average risk score of {top_loc_item.average_risk_score:.1f}/100."
                            ),
                            recommendation=(
                                f"Audit handling equipment and floor demarcation specifically in '{highest_risk_location}'. "
                                "Ensure clear transit paths and proper mechanical lifting aid availability."
                            ),
                            severity="HIGH" if top_loc_item.average_risk_score >= 70.0 else "MEDIUM",
                        )
                    )

        # 5. Temporal Cluster Insight
        clusters = temporal_analysis.get("clusters", [])
        if clusters:
            insights.append(
                PreventionInsight(
                    category="TEMPORAL_SPIKE",
                    title=f"Detected {len(clusters)} High-Density Incident Cluster(s)",
                    observation=(
                        f"Identified {len(clusters)} temporal clusters where multiple violations occurred in close succession "
                        f"(within {5.0} seconds)."
                    ),
                    recommendation=(
                        "Review operational throughput pacing and staffing during peak handling windows to prevent "
                        "hurried manual transfers."
                    ),
                    severity="MEDIUM",
                )
            )

        return insights


def analyze_incidents(incidents: List[IncidentRecord]) -> WarehouseAnalytics:
    """Convenience functional API to execute analytics over an incident list."""
    return AnalyticsEngine(incidents).analyze()
