import streamlit as st

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

st.set_page_config(layout="wide")

# Initialize session state for long text areas
if "text1" not in st.session_state:
    national_curriculum = read_markdown_file("static/sample_national_curriculum.md")
    st.session_state.national_curriculum = national_curriculum
if "text2" not in st.session_state:
    teaching_plan = read_markdown_file("static/sample_teaching_plan.md")
    st.session_state.teaching_plan = teaching_plan



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

st.session_state.teaching_plan = st.text_area("Text Area 2", value=st.session_state.teaching_plan, height=300)

# Initialize session state for short text inputs
if "short_text1" not in st.session_state:
    st.session_state.short_text1 = "[[1]]: The contents of the teaching plan is relevant to national curriculum and the plan is feasible. But the teaching plan does not cover topics mentioned in national curriculum and it lacks details of activity. "
if "short_text2" not in st.session_state:
    st.session_state.short_text2 = "[[2]]: The contents of the teaching plan is relevant to national curriculum and the plan is feasible, and it has a lot of details about activities. But it does not cover some topics in the national curriculum. "
if "short_text3" not in st.session_state:
    st.session_state.short_text3 = "[[3]]: The contents of the teaching plan is relevant to national curriculum and the plan is detailed with activities that will cover the most of the contents in the national curriculum. "

# Short text inputs (stacked vertically)
st.session_state.short_text1 = st.text_input("", label_visibility="collapsed", value=st.session_state.short_text1)
st.session_state.short_text2 = st.text_input("", label_visibility="collapsed", value=st.session_state.short_text2)
st.session_state.short_text3 = st.text_input("", label_visibility="collapsed", value=st.session_state.short_text3)

# Create a card-like container using markdown with styling
with st.container():
    st.markdown(
        """
        <div style="
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            ">
            <h3 style="color: #333;">Current Values</h3>
            <p><strong>Short Text 1:</strong> {short_text1}</p>
            <p><strong>Short Text 2:</strong> {short_text2}</p>
            <p><strong>Short Text 3:</strong> {short_text3}</p>
            <p><strong>Text 1:</strong> {text1}</p>
            <p><strong>Text 2:</strong> {text2}</p>
        </div>
        """.format(
            short_text1=st.session_state.get("short_text1", ""),
            short_text2=st.session_state.get("short_text2", ""),
            short_text3=st.session_state.get("short_text3", ""),
            text1=st.session_state.get("text1", ""),
            text2=st.session_state.get("text2", ""),
        ),
        unsafe_allow_html=True,
    )
