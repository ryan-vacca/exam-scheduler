import io
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Final Exam Scheduler", page_icon="📅", layout="wide")

COURSE_COLUMNS = [
    "course name",
    "professor name",
    "first-year required course",
    "upper-level required course",
    "elective",
    "enrollment",
    "duration",
]
ROOM_COLUMNS = ["room number", "capacity"]


@dataclass(frozen=True)
class Slot:
    dt: datetime

    @property
    def date(self):
        return self.dt.date()

    @property
    def time(self):
        return self.dt.time()

    @property
    def label(self):
        return self.dt.strftime("%a %b %d, %Y at %I:%M %p")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"yes", "y", "true", "1", "x", "required"}


def parse_duration(value) -> float:
    """Return duration in hours. Accepts numeric hours or Excel/time-like strings."""
    if pd.isna(value):
        return 3.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60
        except Exception:
            pass
    if "hour" in s:
        try:
            return float(s.split()[0])
        except Exception:
            pass
    raise ValueError(f"Could not parse duration value: {value}")


def validate_courses(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_columns(df)
    missing = [c for c in COURSE_COLUMNS if c not in df.columns]
    if missing:
        return df, [f"Missing required column(s): {', '.join(missing)}"]

    errors = []
    df = df[COURSE_COLUMNS].copy()
    df["course name"] = df["course name"].astype(str).str.strip()
    df["professor name"] = df["professor name"].astype(str).str.strip()

    for c in ["first-year required course", "upper-level required course", "elective"]:
        df[c] = df[c].apply(as_bool)

    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    if df["enrollment"].isna().any():
        errors.append("Every course must have a numeric enrollment.")
    if (df["enrollment"].fillna(0) <= 0).any():
        errors.append("Enrollment must be greater than zero for every course.")
    df["enrollment"] = df["enrollment"].fillna(0).astype(int)

    try:
        df["duration"] = df["duration"].apply(parse_duration)
    except ValueError as e:
        errors.append(str(e))

    if df["course name"].duplicated().any():
        errors.append("Course names must be unique so that exam-specific constraints can be applied reliably.")

    return df, errors


def validate_rooms(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_columns(df)
    missing = [c for c in ROOM_COLUMNS if c not in df.columns]
    if missing:
        return df, [f"Missing required room column(s): {', '.join(missing)}"]
    df = df[ROOM_COLUMNS].copy()
    errors = []
    df["room number"] = df["room number"].astype(str).str.strip()
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    if df["capacity"].isna().any() or (df["capacity"].fillna(0) <= 0).any():
        errors.append("Every room must have a positive numeric capacity.")
    df["capacity"] = df["capacity"].fillna(0).astype(int)
    return df, errors


def allocate_rooms(enrollment: int, rooms: pd.DataFrame) -> Optional[List[str]]:
    """Greedy room allocation minimizing number of rooms, then unused seats."""
    room_records = rooms.sort_values("capacity", ascending=False).to_dict("records")

    # Single room is preferable when possible; pick smallest room that fits.
    singles = [r for r in room_records if r["capacity"] >= enrollment]
    if singles:
        best = min(singles, key=lambda r: r["capacity"])
        return [best["room number"]]

    # Otherwise, use large rooms until capacity is met.
    chosen = []
    capacity = 0
    for r in room_records:
        chosen.append(r["room number"])
        capacity += r["capacity"]
        if capacity >= enrollment:
            return chosen
    return None


def build_slots(exam_dates: List[date], exam_times: List[time]) -> List[Slot]:
    return [Slot(datetime.combine(d, t)) for d in sorted(exam_dates) for t in sorted(exam_times)]


def occupied_room_sets(schedule_rows: List[dict], slot: Slot, duration: float) -> set:
    """Return rooms occupied by any exam overlapping the candidate interval."""
    occupied = set()
    for row in schedule_rows:
        if interval_conflicts(slot.dt, duration, row["_slot"].dt, row["Duration (Hours)"]):
            occupied.update(row["_room_list"])
    return occupied


def room_allocation_for_slot(enrollment: int, rooms: pd.DataFrame, occupied: set) -> Optional[List[str]]:
    available = rooms[~rooms["room number"].isin(occupied)].copy()
    return allocate_rooms(enrollment, available)


def interval_conflicts(start_a: datetime, duration_a: float, start_b: datetime, duration_b: float) -> bool:
    end_a = start_a + timedelta(hours=duration_a)
    end_b = start_b + timedelta(hours=duration_b)
    return start_a < end_b and start_b < end_a


def professor_conflict(schedule_rows: List[dict], professor: str, slot: Slot, duration: float) -> bool:
    professor = professor.strip().lower()
    for row in schedule_rows:
        if row["Professor Name"].strip().lower() == professor:
            if interval_conflicts(slot.dt, duration, row["_slot"].dt, row["Duration (Hours)"]):
                return True
    return False


def date_counts(schedule_rows: List[dict]) -> Dict[date, int]:
    counts = defaultdict(int)
    for row in schedule_rows:
        counts[row["Exam Date"]] += 1
    return counts


def category_dates(schedule_rows: List[dict], category_key: str) -> List[date]:
    return [row["Exam Date"] for row in schedule_rows if row[category_key]]


def min_date_gap(candidate: date, existing_dates: List[date]) -> int:
    if not existing_dates:
        return 999
    return min(abs((candidate - d).days) for d in existing_dates)


def slot_score(course: pd.Series, slot: Slot, schedule_rows: List[dict], slots: List[Slot]) -> float:
    """Lower is better. Balances daily load and spacing of required exams."""
    counts = date_counts(schedule_rows)
    avg_target = (len(schedule_rows) + 1) / max(1, len(set(s.date for s in slots)))
    load_penalty = abs((counts[slot.date] + 1) - avg_target) * 8
    same_day_penalty = counts[slot.date] * 5

    # Slightly prefer chronological filling when scores otherwise tie.
    chronology = slots.index(slot) * 0.001

    spacing_bonus = 0
    if course["first-year required course"]:
        dates = category_dates(schedule_rows, "First-Year Required")
        gap = min_date_gap(slot.date, dates)
        spacing_bonus -= min(gap, 7) * 14
        if gap == 0:
            spacing_bonus += 1000
        elif gap == 1:
            spacing_bonus += 220
    elif course["upper-level required course"]:
        dates = category_dates(schedule_rows, "Upper-Level Required")
        gap = min_date_gap(slot.date, dates)
        spacing_bonus -= min(gap, 5) * 8
        if gap == 0:
            spacing_bonus += 180
        elif gap == 1:
            spacing_bonus += 60

    return load_penalty + same_day_penalty + chronology + spacing_bonus


def schedule_exams(
    courses: pd.DataFrame,
    rooms: pd.DataFrame,
    slots: List[Slot],
    fixed_dates: Dict[str, date],
    blocked_dates: Dict[str, set],
) -> Tuple[pd.DataFrame, List[str]]:
    schedule_rows = []
    warnings = []

    # Hardest-to-place first: fixed, first-year required, upper-level required, large enrollment, electives.
    def priority(row):
        name = row["course name"]
        return (
            0 if name in fixed_dates else 1,
            0 if row["first-year required course"] else 1,
            0 if row["upper-level required course"] else 1,
            -row["enrollment"],
            name.lower(),
        )

    ordered = courses.sort_values(by="course name").to_dict("records")
    ordered.sort(key=lambda r: priority(pd.Series(r)))

    for rec in ordered:
        course = pd.Series(rec)
        name = course["course name"]
        fixed = fixed_dates.get(name)
        blocked = blocked_dates.get(name, set())

        candidate_slots = [s for s in slots if s.date not in blocked]
        if fixed:
            candidate_slots = [s for s in candidate_slots if s.date == fixed]

        if not candidate_slots:
            warnings.append(f"{name}: no permitted exam slots remain after applying date constraints.")
            continue

        candidate_slots.sort(key=lambda s: slot_score(course, s, schedule_rows, slots))
        placed = False

        for slot in candidate_slots:
            if professor_conflict(schedule_rows, course["professor name"], slot, course["duration"]):
                continue

            occupied = occupied_room_sets(schedule_rows, slot, course["duration"])
            room_list = room_allocation_for_slot(course["enrollment"], rooms, occupied)
            if not room_list:
                continue

            room_capacity = int(rooms.loc[rooms["room number"].isin(room_list), "capacity"].sum())
            schedule_rows.append({
                "Course Name": name,
                "Professor Name": course["professor name"],
                "Exam Date": slot.date,
                "Start Time": slot.time.strftime("%I:%M %p"),
                "Duration (Hours)": float(course["duration"]),
                "End Time": (slot.dt + timedelta(hours=float(course["duration"]))).time().strftime("%I:%M %p"),
                "Enrollment": int(course["enrollment"]),
                "Room(s)": ", ".join(room_list),
                "Room Capacity": room_capacity,
                "First-Year Required": bool(course["first-year required course"]),
                "Upper-Level Required": bool(course["upper-level required course"]),
                "Elective": bool(course["elective"]),
                "_slot": slot,
                "_start_dt": slot.dt,
                "_room_list": room_list,
            })
            placed = True
            break

        if not placed:
            reason = "no room/professor-compatible slot was available"
            if fixed:
                reason += f" on the required date {fixed.strftime('%b %d, %Y')}"
            warnings.append(f"{name}: {reason}.")

    if not schedule_rows:
        return pd.DataFrame(), warnings

    out = pd.DataFrame(schedule_rows)
    out = out.sort_values(["_start_dt", "Course Name"]).reset_index(drop=True)
    return out.drop(columns=["_slot", "_start_dt", "_room_list"]), warnings


def schedule_to_excel(schedule: pd.DataFrame, warnings: List[str]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        schedule.to_excel(writer, index=False, sheet_name="Exam Schedule")
        if warnings:
            pd.DataFrame({"Scheduling Warnings": warnings}).to_excel(writer, index=False, sheet_name="Warnings")
    return output.getvalue()


def template_excel() -> bytes:
    courses = pd.DataFrame([
        {
            "course name": "Contracts I",
            "professor name": "Professor Smith",
            "first-year required course": "Yes",
            "upper-level required course": "No",
            "elective": "No",
            "enrollment": 80,
            "duration": 3,
        },
        {
            "course name": "Evidence",
            "professor name": "Professor Jones",
            "first-year required course": "No",
            "upper-level required course": "Yes",
            "elective": "No",
            "enrollment": 55,
            "duration": 3,
        },
        {
            "course name": "Internet Law",
            "professor name": "Professor Lee",
            "first-year required course": "No",
            "upper-level required course": "No",
            "elective": "Yes",
            "enrollment": 30,
            "duration": 2,
        },
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        courses.to_excel(writer, index=False, sheet_name="Courses")
    return output.getvalue()


def room_template_excel() -> bytes:
    rooms = pd.DataFrame([
        {"room number": "Room 1", "capacity": 100},
        {"room number": "Room 2", "capacity": 60},
        {"room number": "Room 3", "capacity": 40},
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        rooms.to_excel(writer, index=False, sheet_name="Rooms")
    return output.getvalue()


st.title("📅 Final Exam Scheduler")
st.write(
    "Upload course and room spreadsheets, define the exam period, add course-specific date constraints, "
    "and generate a room-aware schedule that spreads exams across the available period."
)

with st.expander("Required spreadsheet formats", expanded=False):
    st.markdown(
        "**Course file columns:** course name, professor name, first-year required course, "
        "upper-level required course, elective, enrollment, duration. Boolean fields may use Yes/No, True/False, 1/0, or X. "
        "Duration is entered in hours (for example, 3 or 2.5).\n\n"
        "**Room file columns:** room number, capacity."
    )
    c1, c2 = st.columns(2)
    c1.download_button("Download course template", template_excel(), "course_template.xlsx")
    c2.download_button("Download room template", room_template_excel(), "room_template.xlsx")

st.header("1. Upload course and room data")
col1, col2 = st.columns(2)
course_file = col1.file_uploader("Course Excel file", type=["xlsx", "xls"], key="courses")
room_file = col2.file_uploader("Room Excel file", type=["xlsx", "xls"], key="rooms")

courses = None
rooms = None
course_errors = []
room_errors = []

if course_file:
    try:
        courses, course_errors = validate_courses(pd.read_excel(course_file))
        if course_errors:
            for e in course_errors:
                st.error(e)
        else:
            st.success(f"Loaded {len(courses)} courses.")
            st.dataframe(courses, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not read course file: {e}")

if room_file:
    try:
        rooms, room_errors = validate_rooms(pd.read_excel(room_file))
        if room_errors:
            for e in room_errors:
                st.error(e)
        else:
            st.success(f"Loaded {len(rooms)} rooms with total capacity {rooms['capacity'].sum()} seats per time slot.")
            st.dataframe(rooms, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not read room file: {e}")

st.header("2. Choose the exam period and available times")
col1, col2 = st.columns(2)
start_date = col1.date_input("First exam date", value=date.today())
end_date = col2.date_input("Last exam date", value=date.today() + timedelta(days=10))

include_weekends = st.checkbox("Allow Saturday and Sunday exams", value=False)

default_times = "09:00\n13:00"
time_text = st.text_area(
    "Available exam start times (one per line, 24-hour format)",
    value=default_times,
    help="Example: 09:00 on the first line and 13:00 on the second line.",
)

exam_times = []
time_errors = []
for line in [x.strip() for x in time_text.splitlines() if x.strip()]:
    try:
        exam_times.append(datetime.strptime(line, "%H:%M").time())
    except ValueError:
        time_errors.append(line)
if time_errors:
    st.error("Invalid time(s): " + ", ".join(time_errors) + ". Use HH:MM in 24-hour format.")

if end_date < start_date:
    st.error("The last exam date must be on or after the first exam date.")
    exam_dates = []
else:
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    exam_dates = [d for d in all_dates if include_weekends or d.weekday() < 5]

excluded_dates = st.multiselect(
    "Dates when no exams may be scheduled",
    options=exam_dates,
    format_func=lambda d: d.strftime("%A, %B %d, %Y"),
)
exam_dates = [d for d in exam_dates if d not in excluded_dates]

if exam_dates and exam_times:
    st.caption(f"Available schedule: {len(exam_dates)} exam days × {len(exam_times)} start times = {len(exam_dates) * len(exam_times)} time slots.")

fixed_dates = {}
blocked_dates = defaultdict(set)

st.header("3. Add course-specific date constraints")
if courses is not None and not course_errors and exam_dates:
    course_names = courses["course name"].tolist()
    st.write("For each course, you may require a particular exam date and/or prohibit one or more dates.")
    selected_courses = st.multiselect("Courses with special date constraints", course_names)

    for name in selected_courses:
        with st.expander(name, expanded=True):
            mode = st.radio(
                f"Constraint type for {name}",
                ["No fixed date", "Require a specific date"],
                horizontal=True,
                key=f"mode_{name}",
            )
            if mode == "Require a specific date":
                fixed_dates[name] = st.selectbox(
                    f"Required exam date for {name}",
                    options=exam_dates,
                    format_func=lambda d: d.strftime("%A, %B %d, %Y"),
                    key=f"fixed_{name}",
                )
            blocked = st.multiselect(
                f"Dates when {name} may NOT be scheduled",
                options=exam_dates,
                format_func=lambda d: d.strftime("%A, %B %d, %Y"),
                key=f"blocked_{name}",
            )
            blocked_dates[name].update(blocked)
            if name in fixed_dates and fixed_dates[name] in blocked_dates[name]:
                st.error(f"{name}: the required date is also prohibited. Remove one of these constraints.")

st.header("4. Generate the schedule")
st.caption(
    "Scheduling priorities: (1) honor fixed/prohibited dates; (2) avoid professor and room conflicts; "
    "(3) strongly space out first-year required exams; (4) space out upper-level required exams; "
    "and (5) balance the overall number of exams across available dates."
)

ready = (
    courses is not None and rooms is not None and not course_errors and not room_errors and
    bool(exam_dates) and bool(exam_times) and end_date >= start_date and not time_errors
)

if st.button("Generate Exam Schedule", type="primary", disabled=not ready):
    contradictory = [n for n, d in fixed_dates.items() if d in blocked_dates.get(n, set())]
    if contradictory:
        st.error("Resolve contradictory date constraints before scheduling: " + ", ".join(contradictory))
    else:
        slots = build_slots(exam_dates, exam_times)
        schedule, warnings = schedule_exams(courses, rooms, slots, fixed_dates, blocked_dates)
        st.session_state["schedule"] = schedule
        st.session_state["warnings"] = warnings

if "schedule" in st.session_state:
    schedule = st.session_state["schedule"]
    warnings = st.session_state.get("warnings", [])
    if schedule.empty:
        st.error("No exams could be scheduled with the current constraints and room inventory.")
    else:
        st.subheader("Generated schedule")
        st.dataframe(schedule, use_container_width=True, hide_index=True)

        scheduled = len(schedule)
        total = len(courses) if courses is not None else scheduled
        c1, c2, c3 = st.columns(3)
        c1.metric("Courses scheduled", f"{scheduled}/{total}")
        c2.metric("Exam days used", schedule["Exam Date"].nunique())
        c3.metric("Rooms available", len(rooms) if rooms is not None else 0)

        daily = schedule.groupby("Exam Date").size().rename("Exams").reset_index()
        st.subheader("Exams per day")
        st.bar_chart(daily.set_index("Exam Date"))

        if warnings:
            st.warning("Some courses could not be placed. See the warnings below and in the downloaded workbook.")
            for w in warnings:
                st.write("• " + w)

        st.download_button(
            "Download schedule as Excel",
            data=schedule_to_excel(schedule, warnings),
            file_name="final_exam_schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.divider()
st.caption("Tip: If a course cannot be scheduled, add room capacity, expand the exam period, add another start time, or relax a course-specific date constraint.")
