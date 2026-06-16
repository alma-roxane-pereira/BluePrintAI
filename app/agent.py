
from typing import TypedDict, List,Optional
from dotenv import load_dotenv
from tools import search_web
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import json

load_dotenv()


class AgentState(TypedDict):
    idea: str
    idea_cleaned: dict
    competitors: List[str]
    pain_points: List[str]
    scorecard: dict
    execution_risk: dict
    pivot_suggestion: dict
    assumption_checklist: dict
    evidence_confidence: dict
    geographic_bias: dict
    
    

def intent_extractor(state: AgentState) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant")
    prompt = f"""
You are an intent extraction assistant.
The user has entered a startup idea that may contain brand names, buzzwords, or vague language.

Your job:
1. Remove any brand names or made-up product names
2. Remove buzzwords like "AI-Driven", "Revolutionary", "Disrupting"
3. Extract the core concept in plain, searchable language
4. Identify the target user and core problem being solved

Input: {state["idea"]}

Return only a JSON object with these keys:
- cleaned_idea: the simplified searchable concept
- target_user: who this is for
- core_problem: what problem it solves

Return only valid JSON, nothing else.
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    extracted = json.loads(raw)
    return {"idea_cleaned": extracted}

def compute_confidence(results: list[str]) -> str:
    
    """Takes a list of result strings (already split by newline).
    Filters out empty lines, then maps count → confidence label."""
    
    non_empty = [r for r in results if r.strip()]
    count = len(non_empty)
    
    if count>=6:
        return "High"
    elif count>=3:
        return "Medium"
    elif count>=1:
        return "Low"
    else:
        return "Insufficient"
    
US_SIGNALS = [
     "us", "usa", "united states", "american", "america",
    "silicon valley", "y combinator", "techcrunch", "san francisco"
]

def detect_geographic_bias(results: list[str])-> str:
    """Checks what fraction of results contain US-specific signals.If 50%+ are US-centric,raises a warning"""
    
    non_empty = [r for r in results if r.strip()]
    if not non_empty:
        return "Insufficient data to detect bias"
    
    us_hit_count = sum(
        1 for r in non_empty
        if any(signal in r.lower() for signal in US_SIGNALS)
        
    )
    
    ratio = us_hit_count/len(non_empty)
    
    if ratio>=0.5:
        return "Warning:Results may  be US-centric"
    else:
        return "No significant geographic bias detected"
    
def competitor_finder(state: AgentState) -> dict:
    idea = state["idea_cleaned"].get("cleaned_idea", state["idea"])
    query = f"existing startups or products that do: {idea}"
    results = search_web(query)
    competitors = results.split("\n")
    
    confidence = compute_confidence(competitors)
    bias = detect_geographic_bias(competitors)
    return {"competitors": competitors,
            "evidence_confidence":{"competitors":confidence},
            "geographic_bias":{"competitors":bias}
            }


def pain_point_miner(state: AgentState) -> dict:
    idea = state["idea_cleaned"].get("cleaned_idea", state["idea"])
    query = f"reddit OR forum people complaining about problems with: {idea}"
    results = search_web(query)
    pain_points = results.split("\n")
    
    confidence = compute_confidence(pain_points)
    bias = detect_geographic_bias(pain_points)
    existing_confidence = state.get("evidence_confidence", {}) 
    existing_bias=state.get("geographic_bias",{})     
    return {
        "pain_points": pain_points,
             "evidence_confidence": {**existing_confidence, "pain_points": confidence},
        "geographic_bias": {**existing_bias, "pain_points": bias}  # ← NEW
    }

def execution_risk_agent(state: AgentState) -> dict:
    idea = state["idea_cleaned"].get("cleaned_idea", state["idea"])
    target_user = state["idea_cleaned"].get("target_user", "unknown")
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    prompt = f"""
You are a ruthless technical feasibility analyst.
Evaluate whether a first-time solo founder can realistically build this idea as an MVP.

Idea: {idea}
Target User: {target_user}

Score each dimension 1-10 where 10 = extremely high risk:

1. technical_complexity: How hard is the tech to build?
2. team_size_required: Can one person build this or does it need a team?
3. time_to_mvp: How long to build a basic working version? (1=weeks, 10=years)
4. dependency_risk: Does it need hardware, licenses, partnerships, or third party approvals?
5. capital_required: How much money is needed before first revenue?

Then provide:
- execution_risk_score: average of all 5 scores
- biggest_execution_challenge: one sentence on the hardest part to build
- mvp_suggestion: what the simplest possible first version looks like

Return only valid JSON with these exact keys:
technical_complexity, team_size_required, time_to_mvp, 
dependency_risk, capital_required, execution_risk_score,
biggest_execution_challenge, mvp_suggestion

Nothing else. Just JSON.
"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    execution_risk = json.loads(raw)
    return {"execution_risk": execution_risk}                   

def aggregator(state: AgentState) -> dict:
    idea = state["idea"]
    competitors = "\n".join(state["competitors"])
    pain_points = "\n".join(state["pain_points"])

    llm = ChatGroq(model="llama-3.1-8b-instant")


    prompt = f"""
You are a cynical venture capital analyst who has seen 10,000 startup pitches fail.
You do NOT give compliments. You identify risk.

Analyze this startup idea ruthlessly across 4 dimensions:

Idea: {idea}
Target User: {state["idea_cleaned"].get("target_user", "unknown")}
Core Problem: {state["idea_cleaned"].get("core_problem", "unknown")}

Competitors found:
{competitors}

Evidence of pain/demand:
{pain_points}

Score each dimension 1-10 where 10 = extremely risky/bad:

1. competition_score: How saturated is this market? Are there well-funded incumbents?
2. market_score: Is there real demand evidence or just assumptions?
3. retention_score: Why would users come back tomorrow? Is there a habit loop?
4. legal_score: Are there regulatory, privacy, or compliance risks?
5. willingness_to_pay_score: Will people actually pay for this? Who pays and how much?
6. defensibility_score: Can this be copied by a developer in a weekend?

Then provide:
- overall_score: average of all 6 scores
- verdict: one of "Strong idea", "Worth testing", "Needs more research", "Avoid"
- one_line_summary: one brutally honest sentence
- top_risk: the single biggest reason this could fail

Return only valid JSON with these exact keys:
competition_score, market_score, retention_score, legal_score, 
willingness_to_pay_score, defensibility_score, overall_score, 
verdict, one_line_summary, top_risk

Nothing else. No explanation. Just JSON.
"""    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    scorecard = json.loads(raw)
    return {"scorecard": scorecard}

def pivot_suggester(state: AgentState) -> dict:
    idea = state["idea"]
    scorecard = state["scorecard"]
    execution_risk = state["execution_risk"]
    
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    prompt = f"""
You are a startup mentor helping a founder who has a risky idea.
Your job is to suggest a narrower, more viable version of their idea.

Original idea: {idea}
Target user: {state["idea_cleaned"].get("target_user", "unknown")}
Core problem: {state["idea_cleaned"].get("core_problem", "unknown")}

Risk scores (10 = very risky):
- Competition: {scorecard.get("competition_score")}
- Retention: {scorecard.get("retention_score")}
- Legal: {scorecard.get("legal_score")}
- Willingness to pay: {scorecard.get("willingness_to_pay_score")}
- Defensibility: {scorecard.get("defensibility_score")}
- Execution complexity: {execution_risk.get("execution_risk_score")}

Top risk: {scorecard.get("top_risk")}
Biggest execution challenge: {execution_risk.get("biggest_execution_challenge")}

Suggest a pivot that:
1. Solves the same core problem
2. Targets a narrower audience
3. Is simpler to build
4. Has less competition
5. Is easier to monetize

Return only valid JSON with these exact keys:
- pivot_idea: one sentence describing the narrower idea
- why_better: one sentence explaining why this is less risky
- new_target_user: who this pivot"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    pivot = json.loads(raw)
    return {"pivot_suggestion": pivot}

def assumption_checklist(state: AgentState) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant")
    
    scorecard = state["scorecard"]
    execution_risk = state["execution_risk"]
    idea_cleaned = state["idea_cleaned"]
    
    prompt = f"""
You are a startup coach helping a first-time founder validate their idea before writing any code.

Idea: {state["idea"]}
Target user: {idea_cleaned.get("target_user", "unknown")}
Core problem: {idea_cleaned.get("core_problem", "unknown")}

Risk signals found:
- Top risk: {scorecard.get("top_risk")}
- Retention score: {scorecard.get("retention_score")} / 10
- Willingness to pay score: {scorecard.get("willingness_to_pay_score")} / 10
- Competition score: {scorecard.get("competition_score")} / 10
- Biggest execution challenge: {execution_risk.get("biggest_execution_challenge")}

Your job:
Generate exactly 4 assumption validation experiments the founder can run within 48 hours — before writing a single line of code.

Each experiment must:
1. Target one specific risky assumption
2. Be completable in under 48 hours
3. Have a clear pass/fail signal

Return only valid JSON with this exact structure:
{{
  "assumptions": [
    {{
      "assumption": "what we are assuming is true",
      "experiment": "exactly what to do in 48 hours",
      "pass_signal": "what result means the assumption is valid",
      "fail_signal": "what result means you should pivot or stop"
    }}
  ]
}}

Nothing else. Just JSON.
"""
    
    response = llm.invoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    checklist = json.loads(raw)
    return {"assumption_checklist": checklist}

def should_pivot(state: AgentState) -> str:
    overall = state["scorecard"].get("overall_score", 0)
    if overall >= 7:
        return "pivot"
    return "end"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_extractor", intent_extractor)
    graph.add_node("competitor_finder", competitor_finder)
    graph.add_node("pain_point_miner", pain_point_miner)
    graph.add_node("execution_risk_agent", execution_risk_agent)
    graph.add_node("aggregator", aggregator)
    graph.add_node("pivot_suggester", pivot_suggester)
    graph.add_node("assumption_checklist",assumption_checklist)

    graph.set_entry_point("intent_extractor")
    graph.add_edge("intent_extractor", "competitor_finder")
    graph.add_edge("competitor_finder", "pain_point_miner")
    graph.add_edge("pain_point_miner", "execution_risk_agent")
    graph.add_edge("execution_risk_agent", "aggregator")
    graph.add_conditional_edges(
        "aggregator",
        should_pivot,
        {
            "pivot": "pivot_suggester",
            "end": "assumption_checklist"
        }
    )
    graph.add_edge("pivot_suggester", "assumption_checklist")

    return graph.compile()

if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "idea": "AI tutor for competitive exams",
        "idea_cleaned": {},
        "competitors": [],
        "pain_points": [],
        "scorecard": {},
        "execution_risk": {},
        "pivot_suggestion":{},
        "assumption_checklist": {},
        "evidence_confidence": {},
        "geographic_bias":{}
    })
    print("\n--- SCORECARD ---")
    for key, value in result["scorecard"].items():
        print(f"{key}: {value}")
    
    print("\n--- EXECUTION RISK ---")
    for key, value in result["execution_risk"].items():
        print(f"{key}: {value}")
        
    print("\n---PIVOT SUGGESTION---")
    for key,value in result["pivot_suggestion"].items():
        print(f"{key}: {value}")
        
    print("\n--- ASSUMPTION CHECKLIST ---")
    for item in result["assumption_checklist"].get("assumptions", []):
        print(f"\nAssumption: {item['assumption']}")
        print(f"Experiment: {item['experiment']}")
        print(f"Pass: {item['pass_signal']}")
        print(f"Fail: {item['fail_signal']}")