from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from server.mock_data import CANDIDATE_DB, TASKS


SAFE_ACTIONS = {
    "list_candidates",
    "fetch_profile",
    "run_interview",
    "shortlist_candidate",
    "reject_candidate",
    "finish",
}


class RecruitmentAction(BaseModel):
    action_type: str = Field(description="Action verb to apply in the environment")
    candidate_id: Optional[str] = Field(default=None, description="Candidate id such as C-2002")
    rationale: Optional[str] = Field(default=None, description="Optional recruiter note for shortlist/reject actions")


class RecruitmentReward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""


class CandidatePreview(BaseModel):
    candidate_id: str
    name: str
    years_experience: int
    expected_salary_lpa: float
    headline: str


class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    years_experience: int
    expected_salary_lpa: float
    current_ctc_lpa: float
    skills: List[str]
    education: str
    location: str
    risk_flags: List[str]
    summary: str


class RecruitmentObservation(BaseModel):
    task_id: str
    difficulty: str
    objective: str
    step_count: int
    max_steps: int
    done: bool
    api_calls_used: int
    api_budget: int
    candidate_pool: List[CandidatePreview]
    fetched_profiles: List[CandidateProfile]
    interview_results: Dict[str, str]
    shortlisted_ids: List[str]
    rejected_ids: List[str]
    progress: float = Field(ge=0.0, le=1.0)
    last_feedback: str
    warnings: List[str] = Field(default_factory=list)


class RecruitmentState(BaseModel):
    task_id: str
    difficulty: str
    objective: str
    step_count: int
    max_steps: int
    done: bool
    api_calls_used: int
    api_budget: int
    fetched_ids: List[str]
    interview_results: Dict[str, str]
    shortlisted_ids: List[str]
    rejected_ids: List[str]
    touched_ids: List[str]
    final_score: Optional[float] = None


class HireFlowRecruitmentEnv:
    def __init__(self) -> None:
        self.current_task_id = "easy_task"
        self.task: Dict[str, Any] = {}
        self.step_count = 0
        self.done = False
        self.api_calls_used = 0
        self.fetched_ids: Set[str] = set()
        self.interview_results: Dict[str, str] = {}
        self.shortlisted_ids: List[str] = []
        self.rejected_ids: List[str] = []
        self.touched_ids: Set[str] = set()
        self.final_score: Optional[float] = None
        self.reset("easy_task")

    def reset(self, task_id: str = "easy_task") -> RecruitmentObservation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.current_task_id = task_id
        self.task = TASKS[task_id]
        self.step_count = 0
        self.done = False
        self.api_calls_used = 0
        self.fetched_ids = set()
        self.interview_results = {}
        self.shortlisted_ids = []
        self.rejected_ids = []
        self.touched_ids = set()
        self.final_score = None

        return self._build_observation(
            "Environment reset. Review candidate pool and decide whom to fetch, shortlist, or reject."
        )

    def state(self) -> RecruitmentState:
        return RecruitmentState(
            task_id=self.current_task_id,
            difficulty=self.task["difficulty"],
            objective=self.task["objective"],
            step_count=self.step_count,
            max_steps=self.task["max_steps"],
            done=self.done,
            api_calls_used=self.api_calls_used,
            api_budget=self.task["api_budget"],
            fetched_ids=sorted(self.fetched_ids),
            interview_results=self.interview_results,
            shortlisted_ids=self.shortlisted_ids,
            rejected_ids=self.rejected_ids,
            touched_ids=sorted(self.touched_ids),
            final_score=self.final_score,
        )

    def step(self, action: RecruitmentAction) -> Tuple[RecruitmentObservation, RecruitmentReward, bool, Dict[str, Any]]:
        if self.done:
            reward = RecruitmentReward(
                score=0.0,
                components={"already_done": 0.0},
                reason="Episode already finished",
            )
            return self._build_observation("Episode already done."), reward, True, {"error": "episode_done"}

        self.step_count += 1
        reward_components: Dict[str, float] = {"step_penalty": -0.01}
        feedback = ""
        warnings: List[str] = []

        if action.action_type not in SAFE_ACTIONS:
            reward_components["invalid_action"] = -0.12
            feedback = f"Unsupported action_type: {action.action_type}"
            return self._respond(reward_components, feedback, warnings, info={"invalid_action": True})

        if action.action_type == "finish":
            self.done = True
            final = self._grade_episode()
            self.final_score = final
            reward_components["final_grade"] = final
            feedback = f"Episode finished. Final grader score={final:.3f}"
            return self._respond(reward_components, feedback, warnings, info={"grader_score": final})

        candidate_id = action.candidate_id
        valid_ids = set(self.task["candidate_ids"])
        if not candidate_id or candidate_id not in valid_ids:
            reward_components["invalid_candidate"] = -0.12
            feedback = "Invalid or missing candidate_id for this task"
            return self._respond(reward_components, feedback, warnings, info={"invalid_candidate": True})

        self.touched_ids.add(candidate_id)
        previous_progress = self._progress()

        if action.action_type == "list_candidates":
            reward_components["discovery"] = 0.02
            feedback = "Candidate pool refreshed"

        elif action.action_type == "fetch_profile":
            self.api_calls_used += 1
            if candidate_id in self.fetched_ids:
                reward_components["duplicate_fetch"] = -0.06
                feedback = f"Profile for {candidate_id} was already fetched"
            else:
                self.fetched_ids.add(candidate_id)
                reward_components["profile_retrieved"] = 0.08
                feedback = f"Fetched full profile for {candidate_id}"

            if self.api_calls_used > self.task["api_budget"]:
                reward_components["budget_overrun"] = -0.25
                warnings.append("API budget exceeded. Additional calls are heavily penalized.")

        elif action.action_type == "run_interview":
            if candidate_id not in self.fetched_ids:
                reward_components["interview_without_profile"] = -0.08
                feedback = f"Fetch profile before running interview for {candidate_id}"
            elif candidate_id in self.interview_results:
                reward_components["duplicate_interview"] = -0.05
                status = self.interview_results[candidate_id]
                feedback = f"Tech interview for {candidate_id} already completed ({status})"
            else:
                risk_flags = set(CANDIDATE_DB[candidate_id]["risk_flags"])
                if "weaker_cs_fundamentals" in risk_flags:
                    self.interview_results[candidate_id] = "fail"
                    reward_components["interview_fail"] = -0.22
                    feedback = f"Tech interview failed for {candidate_id}"
                    warnings.append(
                        "Interview failure reason: weaker CS fundamentals risk flag detected."
                    )
                else:
                    self.interview_results[candidate_id] = "pass"
                    reward_components["interview_pass"] = 0.12
                    feedback = f"Tech interview passed for {candidate_id}"

        elif action.action_type == "shortlist_candidate":
            if candidate_id in self.shortlisted_ids:
                reward_components["duplicate_shortlist"] = -0.1
                feedback = f"Candidate {candidate_id} already shortlisted"
            else:
                self.shortlisted_ids.append(candidate_id)
                fit = self._candidate_fit(candidate_id)
                reward_components["shortlist_quality"] = round((fit * 0.6) - 0.15, 4)
                feedback = f"Candidate {candidate_id} shortlisted"
                salary_cap = self.task["salary_cap_lpa"]
                if CANDIDATE_DB[candidate_id]["expected_salary_lpa"] > salary_cap:
                    reward_components["salary_violation"] = -0.2
                    warnings.append(
                        f"Shortlisted {candidate_id} above salary cap ({salary_cap:.1f} LPA)."
                    )
                interview_status = self.interview_results.get(candidate_id)
                if interview_status == "fail":
                    reward_components["interview_gate_violation"] = -0.3
                    warnings.append(f"{candidate_id} failed the tech interview but was shortlisted.")
                elif interview_status == "pass":
                    reward_components["interview_validated"] = 0.05

        elif action.action_type == "reject_candidate":
            if candidate_id in self.rejected_ids:
                reward_components["duplicate_reject"] = -0.06
                feedback = f"Candidate {candidate_id} already rejected"
            else:
                self.rejected_ids.append(candidate_id)
                fit = self._candidate_fit(candidate_id)
                # Rejecting weak fits is mildly positive, rejecting strong fits is negative.
                reward_components["reject_quality"] = round((0.25 - (fit * 0.4)), 4)
                feedback = f"Candidate {candidate_id} rejected"

        progress_delta = max(-0.25, min(0.25, self._progress() - previous_progress))
        reward_components["progress_delta"] = progress_delta

        if self.step_count >= self.task["max_steps"]:
            self.done = True
            self.final_score = self._grade_episode()
            reward_components["final_grade"] = self.final_score
            warnings.append("Maximum steps reached before explicit finish action.")
            feedback = f"Episode auto-finished at max steps. Final grader score={self.final_score:.3f}"

        return self._respond(reward_components, feedback, warnings, info={"grader_score": self.final_score})

    def _build_observation(
        self,
        feedback: str,
        warnings: Optional[List[str]] = None,
    ) -> RecruitmentObservation:
        warnings = warnings or []
        pool = [
            CandidatePreview(
                candidate_id=candidate_id,
                name=CANDIDATE_DB[candidate_id]["name"],
                years_experience=CANDIDATE_DB[candidate_id]["years_experience"],
                expected_salary_lpa=CANDIDATE_DB[candidate_id]["expected_salary_lpa"],
                headline=CANDIDATE_DB[candidate_id]["summary"],
            )
            for candidate_id in self.task["candidate_ids"]
        ]

        fetched_profiles = []
        for candidate_id in sorted(self.fetched_ids):
            if candidate_id not in self.task["candidate_ids"]:
                continue
            candidate = CANDIDATE_DB[candidate_id]
            fetched_profiles.append(
                CandidateProfile(
                    candidate_id=candidate_id,
                    name=candidate["name"],
                    years_experience=candidate["years_experience"],
                    expected_salary_lpa=candidate["expected_salary_lpa"],
                    current_ctc_lpa=candidate["current_ctc_lpa"],
                    skills=candidate["skills"],
                    education=candidate["education"],
                    location=candidate["location"],
                    risk_flags=candidate["risk_flags"],
                    summary=candidate["summary"],
                )
            )

        return RecruitmentObservation(
            task_id=self.current_task_id,
            difficulty=self.task["difficulty"],
            objective=self.task["objective"],
            step_count=self.step_count,
            max_steps=self.task["max_steps"],
            done=self.done,
            api_calls_used=self.api_calls_used,
            api_budget=self.task["api_budget"],
            candidate_pool=pool,
            fetched_profiles=fetched_profiles,
            interview_results={
                candidate_id: result
                for candidate_id, result in self.interview_results.items()
                if candidate_id in self.task["candidate_ids"]
            },
            shortlisted_ids=self.shortlisted_ids,
            rejected_ids=self.rejected_ids,
            progress=self._progress(),
            last_feedback=feedback,
            warnings=warnings,
        )

    def _respond(
        self,
        components: Dict[str, float],
        feedback: str,
        warnings: List[str],
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[RecruitmentObservation, RecruitmentReward, bool, Dict[str, Any]]:
        raw = sum(components.values())
        score = max(-1.0, min(1.0, raw))
        reward = RecruitmentReward(score=score, components=components, reason=feedback)
        info = info or {}
        if self.final_score is not None:
            info["final_score"] = self.final_score
        observation = self._build_observation(feedback, warnings)
        return observation, reward, self.done, info

    def _candidate_fit(self, candidate_id: str) -> float:
        candidate = CANDIDATE_DB[candidate_id]
        required_skills = set(self.task["required_skills"])
        candidate_skills = set(candidate["skills"])

        skill_match = len(required_skills.intersection(candidate_skills)) / max(1, len(required_skills))
        salary_cap = self.task["salary_cap_lpa"]
        salary_ok = 1.0 if candidate["expected_salary_lpa"] <= salary_cap else 0.0
        exp_factor = min(1.0, candidate["years_experience"] / 5.0)
        risk_penalty = 0.2 if candidate["risk_flags"] else 0.0

        score = (0.55 * skill_match) + (0.25 * salary_ok) + (0.2 * exp_factor) - risk_penalty
        return round(max(0.0, min(1.0, score)), 4)

    def _progress(self) -> float:
        targets = set(self.task["target_shortlist"])
        shortlisted = set(self.shortlisted_ids)

        if not targets:
            return 0.0

        correct_shortlists = len(shortlisted.intersection(targets))
        incorrect_shortlists = len(shortlisted.difference(targets))

        shortlist_component = (correct_shortlists / len(targets)) - (0.35 * incorrect_shortlists)
        shortlist_component = max(0.0, min(1.0, shortlist_component))

        fetch_component = len(self.fetched_ids.intersection(set(self.task["candidate_ids"]))) / max(
            1, len(targets)
        )
        fetch_component = min(1.0, fetch_component)

        budget_efficiency = max(
            0.0,
            1.0 - (max(0, self.api_calls_used - self.task["api_budget"]) / max(1, self.task["api_budget"])),
        )

        progress = (0.65 * shortlist_component) + (0.2 * fetch_component) + (0.15 * budget_efficiency)
        return round(max(0.0, min(1.0, progress)), 4)

    def _grade_episode(self) -> float:
        targets = set(self.task["target_shortlist"])
        shortlisted = set(self.shortlisted_ids)
        candidate_scope = set(self.task["candidate_ids"])

        true_pos = len(shortlisted.intersection(targets))
        false_pos = len(shortlisted.difference(targets))
        false_neg = len(targets.difference(shortlisted))

        precision = true_pos / max(1, true_pos + false_pos)
        recall = true_pos / max(1, true_pos + false_neg)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        salary_cap = self.task["salary_cap_lpa"]
        salary_violations = sum(
            1
            for cid in shortlisted
            if cid in candidate_scope and CANDIDATE_DB[cid]["expected_salary_lpa"] > salary_cap
        )
        interview_fail_violations = sum(
            1
            for cid in shortlisted
            if cid in candidate_scope and self.interview_results.get(cid) == "fail"
        )

        budget_over = max(0, self.api_calls_used - self.task["api_budget"])
        budget_penalty = min(0.45, budget_over * 0.18)
        salary_penalty = min(0.4, salary_violations * 0.2)
        interview_penalty = min(0.45, interview_fail_violations * 0.3)

        if self.step_count <= self.task["max_steps"]:
            efficiency_bonus = 0.08 * (1.0 - (self.step_count / self.task["max_steps"]))
            efficiency_bonus = max(0.0, efficiency_bonus)
        else:
            efficiency_bonus = 0.0

        final = (
            (0.78 * f1)
            + (0.14 * self._progress())
            - budget_penalty
            - salary_penalty
            - interview_penalty
            + efficiency_bonus
        )
        return round(max(0.0, min(1.0, final)), 4)