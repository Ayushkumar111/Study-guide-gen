import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.graph import StateGraph
from typing import TypedDict, Literal
from fpdf import FPDF
from langsmith import traceable

load_dotenv()

#loading env variales from .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key


#shared state

class StudyState(TypedDict):
    topic:str
    search_results:list
    include_plan: bool
    summary:list
    study_plan:list
    resources:str 

#node 1 search topic    

def search_articles(state:StudyState) -> StudyState:
    search_tool= TavilySearchResults()
    results=search_tool.run(f"{state['topic']} beginner tutorial")
    print(" Search Results:")
    for result in results:
        print(result)
    return {**state, "search_results":results}


#node 2 summarize the articles 

def summarize_articles(state:StudyState)-> StudyState:
    llm=ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.0-flash" )
    snippets ="\n".join([r['title'] for r in state['search_results'][:3]])
    prompt = f""" you are an ai tutor , Here's what i found about "{state['topic']}":
    {snippets}
    please summarize the articles in a list of key points.
    """
    max_retries = 2 
    for attempt in range( max_retries):
        response=llm([HumanMessage(content=prompt)])
        summary= response.content.strip()
        if len(summary.split())>50:
            return {**state,"summary":summary}
        
    return{**state,"summary": "Could not generate a good summary"}
    

#node 3 to create study plan 

def generate_study_plan(state: StudyState) -> StudyState:
    
    if state.get('include_plan', True):
        llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.0-flash")
        prompt = f"""Based on this summary: {state['summary']}, create a 7-DAY learning plan for the topic "{state['topic']}" with daily goals and resources."""
        response = llm([HumanMessage(content=prompt)])
        return {**state, "study_plan": response.content}
    else:
        
        return {**state, "study_plan": ""}


#recommend videos and courses 

def recommend_resource(state:StudyState)-> StudyState:
    search_tool= TavilySearchResults()
    query=f"{state['topic']} tutorial site:youtube.com OR site:udemy.com OR site:coursera.org"
    results = search_tool.run(query)

    recommendations = "\n".join([f" {r['title']}\n {r['url']}"for r in results[:5]])

    return{**state,"resources":recommendations}


#node 5 to generate pdf 
def generate_pdf_output(state: StudyState) -> StudyState:
    print("\n Summary, Plan & Resources ready!")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Study Guide: {state['topic']}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, " Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, state['summary'])
    pdf.ln(5)

    if state.get('include_plan', True) and state['study_plan']:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, " 7-Day Study Plan", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, state['study_plan'])
        pdf.ln(5)
    

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, " Resources", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, state['resources'])

    pdf_bytes = pdf.output(dest='S')

    return {**state, "pdf_bytes": pdf_bytes}




def router(state: StudyState) -> Literal["generate_plan", "skip_plan"]:
    if state.get('include_plan', True):
        return "generate_plan"
    else:
        return "skip_plan"



##langgraph to build 

builder = StateGraph(StudyState)
builder.add_node("search", search_articles)
builder.add_node("summarize", summarize_articles)
builder.add_node("study_plann", generate_study_plan)
builder.add_node("resourcess", recommend_resource)
builder.add_node("output", generate_pdf_output)

builder.add_conditional_edges(
    "summarize",
    router,
    {
        "generate_plan": "study_plann",
        "skip_plan": "resourcess"
    }
)


builder.set_entry_point("search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", "study_plann")
builder.add_edge("study_plann", "resourcess")
builder.add_edge("resourcess", "output")
builder.set_finish_point("output")

graph = builder.compile()

if __name__ == "__main__":
    topic = input("Enter the topic you want to study:")
    include_plan = input("Include 7-day study plan? (y/n): ").lower() == 'y'
    graph.invoke({"topic":topic})
