import os
from openai import OpenAI
import streamlit as st

openai_api_key = os.environ["OPENAI_API_KEY"]

def read_markdown_file(file_path):
    """Reads a markdown file and prints its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"
    

def evaluate_teaching_plan_llm(evaluation_criteria, national_curriculum, teaching_plan):

    with open("static/single_judge_prompt_template.txt", "r", encoding="utf-8") as file:
        single_judge_prompt_template = file.read()

    messages = [
        {
            "role": "system",
            "content": "You are a teaching plan evaluator",
        },
        {
            "role": "user",
            "content": single_judge_prompt_template.format(evaluation_criteria=evaluation_criteria, 
                                                        national_curriculum=national_curriculum, 
                                                        teaching_plan=teaching_plan),
        },
    ]

    params = {
    "messages": messages,
    "max_tokens": 2048,
    "model": "gpt-4-turbo-2024-04-09"
    }
    # OpenAI APIのクライアントを初期化
    client = OpenAI()
    # OpenAI APIにリクエストを送信
    response = client.chat.completions.create(**params)

    content = response.choices[0].message.content

    return content, single_judge_prompt_template.format(evaluation_criteria=evaluation_criteria, 
                                                        national_curriculum=national_curriculum, 
                                                        teaching_plan=teaching_plan)


st.set_page_config(layout="wide")

st.header("Evaluation of Short-Term Teaching Plan")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/llm-as-a-judge.jpg", use_container_width =True)

# Initialize session state for long text areas
if "text1" not in st.session_state:
    national_curriculum = read_markdown_file("static/sample_national_curriculum.md")
    st.session_state.national_curriculum = national_curriculum
if "text2" not in st.session_state:
    teaching_plan = read_markdown_file("static/sample_teaching_plan.md")
    st.session_state.teaching_plan = teaching_plan


st.markdown(
    """
    <div style="
        font-weight: bold;
        font-size: 20px;
        ">
        An example part of national curriculum
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div style="
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        max-height: 300px;
        overflow-y: auto;
        ">
        {st.session_state.national_curriculum}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <br>
    <div style="
        font-weight: bold;
        font-size: 20px;
        ">
        Draft of a short-term teaching plan
    </div>
    """,
    unsafe_allow_html=True,
)
st.session_state.teaching_plan = st.text_area("", label_visibility="collapsed", value=st.session_state.teaching_plan, height=300)

st.markdown(
    """
    <br>
    <div style="
        font-weight: bold;
        font-size: 20px;
        ">
        Criteria for evaluation scores
    </div>
    """,
    unsafe_allow_html=True,
)
# Initialize session state for short text inputs
if "criteria_0" not in st.session_state:
    st.session_state.criteria_0 = "The contents of the teaching plan is irrelevant to national curriculum or does not make any sense. Or the generated contents could be sensitive or harmful."
if "criteria_1" not in st.session_state:
    st.session_state.criteria_1 = "The contents of the teaching plan is relevant to national curriculum and the plan is feasible. But the teaching plan does not cover topics mentioned in national curriculum and it lacks details of activity. "
if "criteria_2" not in st.session_state:
    st.session_state.criteria_2 = "The contents of the teaching plan is relevant to national curriculum and the plan is feasible, and it has a lot of details about activities. But it does not cover some topics in the national curriculum. "
if "criteria_3" not in st.session_state:
    st.session_state.criteria_3 = "The contents of the teaching plan is relevant to national curriculum and the plan is detailed with activities that will cover the most of the contents in the national curriculum. "

# Short text inputs (stacked vertically)
st.session_state.criteria_0 = st.text_input("Criteria for giving a score of 0", value=st.session_state.criteria_0)
st.session_state.criteria_1 = st.text_input("Criteria for giving a score of 1", value=st.session_state.criteria_1)
st.session_state.criteria_2 = st.text_input("Criteria for giving a score of 2", value=st.session_state.criteria_2)
st.session_state.criteria_3 = st.text_input("Criteria for giving a score of 3", value=st.session_state.criteria_3)

if st.button("Evaluate Teaching Plan"):
    evaluation_criteria = "[[0]]: {}\n[[1]]: {}\n[[2]]: {}\n[[3]]: {}\n".format(st.session_state.criteria_0, 
                                                                                st.session_state.criteria_1, 
                                                                                st.session_state.criteria_2, 
                                                                                st.session_state.criteria_3
                                                                                )

    national_curriculum = st.session_state.national_curriculum
    teaching_plan = st.session_state.teaching_plan
    content, evaluation_prompt = evaluate_teaching_plan_llm(evaluation_criteria, national_curriculum, teaching_plan)

    print(evaluation_prompt)
    st.session_state.evaluation_result = content

    

if "evaluation_result" in st.session_state:
    st.markdown(
        f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            background-color: #e9ecef;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            ">
            <h3 style="color: #333;">Evaluation Result</h3>
            <p>{st.session_state.evaluation_result}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
    """
    <br>
    <div style="
        font-weight: bold;
        font-size: 20px;
        ">
        The prompt used for evaluation
    </div>
    """,
    unsafe_allow_html=True,
    )   
    st.json({evaluation_prompt}, expanded=False)

