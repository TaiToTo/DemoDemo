import os
import math
import yaml
import json 
import ast
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv() 

import pandas as pd
import igraph
from igraph import Graph
from igraph import EdgeSeq
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter



weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

# Load from the JSON file
with open("subject_color_map.json", "r") as f:
    subject_color_map = json.load(f)


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
    

def query_vectors(query_text, search_limit, selected_subjects):
    class_name = "CurriculumDemo"
        # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get("CurriculumDemo")

    object_filter_list = [Filter.by_property("subject").equal(subject) for subject in selected_subjects]

    response = collection.query.near_text(
        query=query_text,
        limit=search_limit, 
        return_metadata=MetadataQuery(distance=True, certainty=True), 
        filters=(
            Filter.any_of(object_filter_list)
        )
    )

    client.close()  # Free up resources

    row_dict_list = []
    for obj in response.objects:
        row_dict = {}
        row_dict["text"] = obj.properties["text"]
        row_dict["subject"] = obj.properties["subject"]
        row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        row_dict_list.append(row_dict)

    df_queried = pd.DataFrame(row_dict_list, columns=["text", "subject", "certainty", "distance", "paragraph_idx"])
    
    df_queried["text"] = df_queried["text"].map(lambda x: x.replace("\n", "<br>") )

    df_queried["hover_text"] = df_queried.apply(
        lambda x: "<b>Relevance:</b> {:.3f}<br><b>Paragraph idx:</b> {}<br><b>Paragraph idx:</b> {}<br>{}".format(x["certainty"], x["subject"], x["paragraph_idx"], x["text"]),
        axis=1
    )

    return df_queried


def get_rag_answer_and_sources(query_text, task_text, search_limit):
    class_name = "CurriculumDemo"
    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get("CurriculumDemo")

    response = collection.generate.near_text(
        query=query_text,
        limit=search_limit,
        grouped_task=task_text, 
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    client.close()  # Free up resources

    row_dict_list = []
    for obj in response.objects:
        row_dict = {}
        row_dict["text"] = obj.properties["text"]
        row_dict["subject"] = obj.properties["subject"]
        row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        row_dict_list.append(row_dict)

    df_queried = pd.DataFrame(row_dict_list, columns=["text", "subject", "certainty", "distance", "paragraph_idx"])
    df_queried["text"] = df_queried["text"].map(lambda x: x.replace("\n", "<br>") )
    df_queried["hover_text"] = df_queried.apply(
        lambda x: "<b>Relevance:</b> {:.3f}<br><b>Paragraph idx:</b> {}<br><b>Paragraph idx:</b> {}<br>{}".format(x["certainty"], x["subject"], x["paragraph_idx"], x["text"]),
        axis=1
    )

    generated_text = response.generated

    return generated_text, df_queried

    

def evaluate_teaching_plan_llm(evaluation_criteria, national_curriculum, teaching_plan):

    with open("static/syllabus_evaluation_prompt_template.txt", "r", encoding="utf-8") as file:
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


def make_curriculum_graph(parent_node, edges=[], labels=[], annotations=[]):
    def format_annoation_text(_text):
        split_texts = _text.split(" ")
        current_line = split_texts[0] + " "
        formatted_text = ""

        for elem in split_texts[1:]:
            # print("elem: ", elem)

            if len(current_line) + len(elem) < 20:
                # print("Checkpoint A")
                if elem != split_texts[-1]:
                    current_line += elem + " "
                else:
                    formatted_text += current_line+ elem
            else:
                if elem != split_texts[-1]:
                    # print("Checkpoint B")
                    formatted_text += current_line + "<br>" 
                    current_line = elem + " "
                else:
                    # print("Checkpoint C")
                    formatted_text += current_line + "<br>" + elem
        
        formatted_text = formatted_text.replace(". ", "<br>")
        return formatted_text
    
    parent_id = parent_node["id"]

    for child in parent_node["children"]:
        child_id = child["id"]
        edges.append((parent_id, child_id))
        # labels.append("Node ID: {}<br>{}".format(child["id"], child["label"]))
        labels.append(child["label"][:1024] + "...")
        annotations.append(format_annoation_text(child["annotations"]))

        if len(child["children"]) > 0:
            edges, labels, annotations = make_curriculum_graph(child, edges=edges, labels=labels,
                                                               annotations=annotations)

    return edges, labels, annotations


def make_igraph_tree_plot_data(section_edges, annotations, queried_node_list=None):
    # TODO: not a very readable function, but this should be reimplmented in frontend in the future
    def make_annotations(pos, annotations, font_size=5, font_color='black'):
        L=len(pos)
        if len(annotations)!=L:
            raise ValueError('The lists pos and text must have the same len')
        hover_text_dicts = []
        for k in range(L):
            hover_text_dicts.append(
                dict(
                    text=annotations[k], 
                    x=pos[k][0], y=2*M-position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return hover_text_dicts

    def make_node_opacities(pos, queried_node_list):
        L = len(pos)

        opacities = [0.25 for _ in range(L)]
        node_colors = ["white" for _ in range(L)]

        # print("L: {}".format(L))

        # print([node["node_id"] for node in queried_node_list])
        # print(queried_node_list)
        for node in queried_node_list:
            opacities[int(node["node_id"])] = float(node["certainty"])
            # opacities[int(node["node_id"])] = 1
            node_colors[int(node["node_id"])] = "orange"

        return opacities, node_colors

    G = Graph.TupleList(section_edges, directed=True)
    nr_vertices = G.vcount()
    lay = G.layout('rt', root=[0])
    position = {k: (lay[k][0], lay[k][1] + math.sin(lay[k][0]) * 0.5) for k in range(nr_vertices)}

    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)
    es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges
    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    igraph_annotations = make_annotations(position, annotations)
    node_opacities = 0.8
    node_colros = "white"

    if queried_node_list is not None:
        node_opacities, node_colors = make_node_opacities(position, queried_node_list)

    return Xe, Ye, Xn, Yn, position, igraph_annotations, node_opacities, node_colors


def get_curriculum_tree_graph_object(curriculum_tree, title=None, queried_node_list=None):
    # TODO: the initial labels could be added to make_curriculum_graph() function
    labels = []
    annotations = []

    labels.append(curriculum_tree["label"])
    annotations.append(curriculum_tree["annotations"])

    section_edges, v_labels, annotations = make_curriculum_graph(curriculum_tree, edges=[], labels=labels,
                                                                 annotations=annotations)

    Xe, Ye, Xn, Yn, position, igraph_annotations, node_opacities, node_colors = make_igraph_tree_plot_data(
        section_edges, annotations, queried_node_list=queried_node_list)

    graph_title = 'Structure of a sections in a curriculum'


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name='bla',
                             marker=dict(
                                 symbol='diamond-wide',
                                size=65,
                                    # color="white",    #'#DB4551',
                                    color=node_colors,
                                        line=dict(color='black', width=0.5), 
                                        opacity=node_opacities
                                    ),
                             text=v_labels,
                             hoverinfo='text',
                             ))


    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    fig.update_layout(title='Estonian Basic School Curriculum as a Tree',
                      annotations=igraph_annotations,
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=10, r=10, b=85, t=100),  # Reduced left and right margins
                      hovermode='closest',
                      plot_bgcolor='white',
                      hoverlabel=dict(
                          bgcolor="white",
                          font_size=15,  # Adjust the font size of hover text here
                          font_family="Rockwell",
                          namelength=10,  # This ensures that the hover text is not truncated
                          align='left',  # Align text to the left
                      ),
                      height=600,  # Set the height of the figure
                      width=1500  # Set the width of the figure
                      )
    
    for annotation in fig['layout']['annotations']:
        annotation['font']['size'] = 10  # Set the font size for annotations

    return fig


def query_node_vectors(selected_subject, query_text, search_limit, selected_subjects=None):
    class_name = "EstonianCurriculumTree"

    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,  # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),  # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},  # Replace with your Cohere API key
    )

    collection = client.collections.get(class_name)

    # object_filter_list = [Filter.by_property("subject").equal(subject) for subject in selected_subjects]

    object_filter_list = [Filter.by_property("subject").equal(selected_subject),
                          Filter.by_property("text_type").equal("text")]

    response = collection.query.near_text(
        query=query_text,
        limit=search_limit,
        return_metadata=MetadataQuery(distance=True, certainty=True),
        filters=(
            Filter.all_of(object_filter_list)
        )
    )

    client.close()  # Free up resources

    queried_node_list = []

    for obj in response.objects:
        row_dict = {}
        row_dict["node_id"] = obj.properties["node_id"]
        row_dict["label"] = obj.properties["label"]
        row_dict["subject"] = obj.properties["subject"]
        # row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance

        queried_node_list.append(row_dict)
    return queried_node_list


st.set_page_config(layout="wide")

# Load data
base_path = os.path.dirname(__file__)

df_scatter = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))
# df_scatter["hover_text"] = df_scatter["text"].replace("\n", "<br>")
color_list = [subject_color_map[row["subject"]] for _, row in df_scatter.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df_scatter.iterrows()]

st.title("AI Curriculum Analyzer Demo")

# Table of Contents
st.sidebar.title("Table of Contents")
st.sidebar.markdown("""
- [Purpose of This Demo](#purpose-of-this-demo)
- [Preparation of Data](#preparation-of-data)
- [Search the Curriculum](#search-the-curriculum)
- [Make a Prompt with Queried Texts](#make-a-prompt-with-queried-texts)
- [Interdisciplinary Idea Analysis](#interdisciplinary-idea-analysis)
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/banner.png", use_container_width =True)


with open('static/stored_samples.txt') as f:
    loaded_data = ast.literal_eval(f.read())

st.header("Purpose of This Demo")
st.write("""
The potential of **generative AI** has become a popular topic of discussion last some years, including in the education sector. However, most conversations seem to focus on the **"generative" aspect**—such as chatbots or text generation.
While *content creation* is certainly a key advantage, it's only one part of what generative AI can offer. Beyond its name, the real power of generative AI lies in its ability to **transform how we interact with and retrieve information**.
Think back to the days when people typed precise commands into black screens or navigated files by clicking and dragging. Now, thanks to generative AI, **information can be accessed with fuzzy, conversational prompts in natural language**. 
""")

col1, col2, col3 = st.columns([1, 6, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/generative_AI_two_aspects.jpg", use_container_width =True)

st.write("""
Through this demo and presentation, we aim to emphasize the following key points:

1. **Efficient Discovery**: How generative AI can quickly and efficiently locate texts of interest.  
2. **Automated Process**: How the contents of searched texts can be processed automatically with minimal effort.  
3. **Intuitive Visualization**: How AI techniques can visualize the relationships between texts in a more intuitive and meaningful way.  

We hope this demo demonstrates that the seemingly magical behaviors of products with generative AI are, at their core, just **combinations of limited functionalities**:  
- **Searching texts** by calculating **relevance of texts** 
- **Executing commands to process the searched texts**.  

By understanding the engineering behind the "magic," we encourage readers to explore **practical methods for curriculum analysis** and unlock new opportunities in education.
""")

# st.write("""We hope this demo shows that seemingly magical behaviors of generative AI are just combinations of the limited funcitonalities: searching texts and conducting commands based on the searched texts. 
# And by knowing the engineering behind the magic, it would be great if the readers could come up with practical methods of curriculum analysis.""")

st.header("Preparation of Data")
st.write("""
To effectively utilize generative AI in curriculum analysis, documents must first be **divided into smaller text units** and then stored in a **database** after being processed by a generative AI model.
Each text is then **converted into a numerical expression**, making it easier for generative AI models to search and process. This numerical representation is called a **vector** or **embedding**.
""")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/embedding_split.png", use_container_width =True)

st.write("""
In this demo, the **national curricula** or basic shools provided by the [Estonian Ministry of Education and Research](https://www.hm.ee/en/national-curricula) are used. 
The texts are **divided roughly by paragraph** and stored in the database with relevant **tags**, such as **"subject"** and **"paragraph number"**, to enable efficient **analysis** and **retrieval**.
The first 10 sample texts stored can be also found below. 
""")

col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/est_basic_school_nat_cur_2014_appendix_1_final-images-0.jpg", use_container_width =True)

with col4:
    st.image("static/est_basic_school_nat_cur_2014_appendix_1_final-images-1.jpg", use_container_width =True)

st.write("The texts actually stored in the database are as follows:")
st.json(loaded_data, expanded=False)


st.header("Search the Curriculum")
st.subheader("Search related content from all the curriculums")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/text_query.jpg", use_container_width =True)

st.write("""
After the texts are stored in the database, the most **relevant** or **semantically closest** texts to an input query in a curriculum can be found.
""")

st.subheader("Search related sections in a single curriculum")

tree_subjects = [
                "mathematics", 
                "art", 
                "career_education", 
                "entrepreneurship_studies", 
                "foreign_languages", 
                "informatics", 
                "language_and_literature", 
                "mathematics", 
                "natural_science", 
                "physical_education", 
                "religious_studies", 
                "technology"
                ]

tree_selected_subject = st.selectbox("Select a subject:", options=list(tree_subjects))

tree_text_search_limit = st.slider("Select a number of texts to query", 
                              min_value=0, 
                              max_value=50, 
                              value=3, 
                              step=1
                              )

if tree_selected_subject:
    # Read YAML file
    with open("../JapaneseDocumentLayoutAnalysis/EstonianCurriculumTreeData/Estonian_basic_school_tree_{}.yaml".format(tree_selected_subject), "r", encoding="utf-8") as file:
        curriculum_tree = yaml.safe_load(file)  # Load YAML content as a dictionary

tree_query_text = st.text_input("Write a query search positions of relevant content in the curriculum:", "Collaborating with teachers in other subjects")

if tree_query_text:

    queried_node_list = query_node_vectors(tree_selected_subject, tree_query_text, tree_text_search_limit)
    fig_tree = get_curriculum_tree_graph_object(curriculum_tree, title=tree_query_text, queried_node_list=queried_node_list)

    st.plotly_chart(fig_tree)


st.write("""
The following graphs display the **queried texts** in all the curriculums and their **relevance scores** relative to the query. These texts are also **grouped by subject** on the right side for easier analysis.
""")

query_text = st.text_input("Write a query to search contents of curriculum:", "Data literacy")

text_search_limit = st.slider("Select a number of texts to query", 
                              min_value=0, 
                              max_value=100, 
                              value=30, 
                              step=1
                              )

# Multi-select filter
selected_subjects = st.multiselect(
    "Select subjects to filter:",
    options=subject_color_map.keys(),  
    default=subject_color_map.keys(),  
)

if query_text :
    df_queried = query_vectors(query_text, text_search_limit, selected_subjects)

    colors = [subject_color_map[cat] for cat in df_queried['subject']]

    # Create the subplots with an additional row for the bottom graph
    fig_1 = make_subplots(
        rows=1,  # Increase the rows to 2
        cols=2,  # Keep the columns as 2 for the top row
        column_widths=[0.3, 0.7],  # Set column widths for the first row
        subplot_titles=(
            "Relevant texts of the query '{}'".format(query_text), 
            "Searched texts grouped by subjects", 
            ),  # Titles for all subplots
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # First row specs
        ]
    )

    # Add bar plot to the first column
    fig_1.add_trace(
        go.Bar(
            # y=df_queried['text'].map(lambda x: x),
            y=[idx for idx in range(len(df_queried))],
            x=df_queried['certainty'][::-1],
            orientation='h',
            textposition='outside',
            marker=dict(color=colors[::-1]), 
            hovertext=df_queried['hover_text'][::-1], 
            hoverinfo='text' , 
            # title="The most relevant text to the query {}".format(query_text)
        ),
        row=1,
        col=1
    )

    fig_1.update_layout(
        xaxis=dict(
            title='Relavance Score',  # Label for the x-axis
            range=[0.5, 0.8], 
                tickfont=dict(
                size=10  # Adjust the font size as desired
            )
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[idx for idx in range(len(df_queried))],  # Custom y-ticks
            ticktext=df_queried['text'][::-1].map(lambda x: x[:20] + "...")  # Use custom labels for y-axis
        ), 
    )

    unique_subject_list = df_queried["subject"].unique().tolist()
    colors = [subject_color_map[cat] for cat in unique_subject_list + df_queried['subject'].tolist()]

    # Add treemap to the second column
    fig_1.add_trace(
        go.Treemap(
            labels=unique_subject_list + df_queried["text"].tolist(),
            parents=[""]*len(unique_subject_list) + df_queried["subject"].tolist(),  # Since "path" isn't directly supported, set parents to empty or adjust accordingly.
            values=[0]*len(unique_subject_list) + df_queried["certainty"].tolist(),
            marker=dict(colors=colors), 
            hovertext=unique_subject_list + df_queried['hover_text'].tolist(), 
            hoverinfo='text' 
        ),
        row=1,
        col=2
    )

    fig_1.update_layout(
        height=500,  # Set the height of the figure (increase as needed)
        hoverlabel=dict(
            align="left"  # Left-align the hover text
        ), 
    )

else:
    # Create the subplots with an additional row for the bottom graph
    fig_1 = make_subplots(
        rows=1,  # Increase the rows to 2
        cols=2,  # Keep the columns as 2 for the top row
        column_widths=[0.2, 0.8],  # Set column widths for the first row
        subplot_titles=(
            "The most relevant texts ot the query '{}'".format(), 
            "Searched texts grouped by subjects", 
            ),  # Titles for all subplots
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # First row specs
        ]
    )

    # Add bar plot to the first column
    fig_1.add_trace(
        go.Bar(
        ),
        row=1,
        col=1
    )

    # Add treemap to the second column
    fig_1.add_trace(
        go.Treemap(
        ),
        row=1,
        col=2
    )

st.plotly_chart(fig_1)


st.header("Make a Prompt with Queried Texts")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/RAG.jpg", use_container_width =True)


st.write("""
The **queried texts** can be processed using another prompt. In the demo below, a query for searching texts from the database can be set first, followed by the ability to perform various **tasks** on the retrieved texts.
This process is known as **RAG (Retrieval-Augmented Generation)**, and it is already widely used at the **product level**.
""")

rag_search_limit = st.slider("Number of top relevant texts used", min_value=0, max_value=10, value=5, step=1)
# Input fields for two texts
prompt_text = st.text_input("Write a prompt", "Extract competency related to data literacy and list them up without rephrasing them.", placeholder="Type something here...")
# task_text = st.text_input("Write a prompt based on the texts queried:", "Extract competency from the text", placeholder="Type something here...")

# Button to combine texts
if st.button("Make a prompt"):
    if prompt_text:
        # Combine the texts and output the result
        generated_text, df_ref = get_rag_answer_and_sources(prompt_text, prompt_text, rag_search_limit)

        st.markdown("### Generated answer:")
        st.write(generated_text)

        unique_subject_list = df_ref["subject"].unique().tolist()

        colors = [subject_color_map[cat] for cat in unique_subject_list + df_ref['subject'].tolist()]

        fig_2 = go.Figure()

        # Add treemap to the second column
        fig_2.add_trace(
            go.Treemap(
                labels=unique_subject_list + df_ref["hover_text"].tolist(),
                parents=[""]*len(unique_subject_list) + df_ref["subject"].tolist(),  # Since "path" isn't directly supported, set parents to empty or adjust accordingly.
                values=[0]*len(unique_subject_list) + df_ref["certainty"].tolist(),
                marker=dict(colors=colors), 
                hovertext=unique_subject_list + df_ref['hover_text'].tolist(), 
                hoverinfo='text' 
            ),

        )

        fig_2.update_layout(
            height=150,  # Set the height of the figure (increase as needed)
            hoverlabel=dict(
                align="left"  # Left-align the hover text
            ), 
            margin=dict(
                l=10,  # Left margin
                r=10,  # Right margin
                t=10,  # Top margin
                b=10   # Bottom margin
            )
        )

        st.markdown("### Queried texts used for generating the answer:")
        st.plotly_chart(fig_2, use_container_width=True)

    else:
        st.warning("Please provide both texts to combine.")


# Create the scatter plot
fig_3 = go.Figure()

# Add the scatter trace with hover text
fig_3.add_trace(go.Scatter(
    x=df_scatter["x"],
    y=df_scatter["y"],
    mode='markers',
    marker=dict(
        color=color_list,
        size=6
    ),
    text=hover_texts,  # Add hover text
    hoverinfo='text',  # Display only the hover text
    name="Scatter Points"
))

# Add legend items for each subject
for subject, color in subject_color_map.items():
    fig_3.add_trace(go.Scatter(
        x=[None], y=[None],  # Dummy points for legend
        mode='markers',
        marker=dict(size=10, color=color),
        name=subject
    ))

# Update layout
fig_3.update_layout(
    title="Semantic map of paragraphs in Estonian basic school curriculum",
    legend_title="Subjects",
    legend=dict(x=1.05, y=1),
    margin=dict(t=40, l=0, r=150, b=40),
    width=600,
    height=600,
    hovermode='closest',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        align="left"
    )
)


st.header("Interdisciplinary Idea Analysis")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/interdisciplinary_analysis.jpg", use_container_width =True)

st.plotly_chart(fig_3, use_container_width=True)

# st.header("Short-term evaluation of a lesson")
# col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
# with col2:
#     st.image("static/llm-evaluator.jpg", use_container_width =True)

# # Initialize session state for long text areas
# if "text1" not in st.session_state:
#     national_curriculum = read_markdown_file("static/sample_national_curriculum.md")
#     st.session_state.national_curriculum = national_curriculum
# if "text2" not in st.session_state:
#     teaching_plan = read_markdown_file("static/sample_teaching_plan.md")
#     st.session_state.teaching_plan = teaching_plan
# if "text3" not in st.session_state:
#     national_curriculum_general_competence = read_markdown_file("static/sample_national_curriculum_general_competence.md")
#     st.session_state.national_curriculum_general_competence = national_curriculum_general_competence



# st.markdown(
#     """
#     <div style="
#         font-weight: bold;
#         font-size: 20px;
#         ">
#         An example part from the Estonian basic school curriculum.
#     </div>
#     <br>
#     """,
#     unsafe_allow_html=True,
# )


# st.markdown(
#     f"""
#     <div style="
#         padding: 15px;
#         border-radius: 10px;
#         background-color: #f8f9fa;
#         box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
#         max-height: 300px;
#         overflow-y: auto;
#         ">
#         {st.session_state.national_curriculum}
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown(
#     """
#     <br>
#     <div style="
#         font-weight: bold;
#         font-size: 20px;
#         ">
#         Draft of a short-term teaching plan (try to edit it and get a higher score!!)
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# st.session_state.teaching_plan = st.text_area("", label_visibility="collapsed", value=st.session_state.teaching_plan, height=300)

# st.markdown(
#     """
#     <br>
#     <div style="
#         font-weight: bold;
#         font-size: 20px;
#         ">
#         Criteria for evaluation scores
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# # Initialize session state for short text inputs
# if "criteria_0" not in st.session_state:
#     st.session_state.criteria_0 = "The contents of the teaching plan is irrelevant to national curriculum or does not make any sense. Or the generated contents could be sensitive or harmful."
# if "criteria_1" not in st.session_state:
#     st.session_state.criteria_1 = "The contents of the teaching plan is relevant to national curriculum and the plan is feasible. But the teaching plan does not cover topics mentioned in national curriculum and it lacks details of activity. "
# if "criteria_2" not in st.session_state:
#     st.session_state.criteria_2 = "The contents of the teaching plan is relevant to national curriculum and the plan is feasible, and it has a lot of details about activities. But it does not cover some topics in the national curriculum. "
# if "criteria_3" not in st.session_state:
#     st.session_state.criteria_3 = "The contents of the teaching plan is relevant to national curriculum and the plan is detailed with activities that will cover the most of the contents in the national curriculum. "

# # Short text inputs (stacked vertically)
# st.session_state.criteria_0 = st.text_input("Criteria for giving a score of 0", value=st.session_state.criteria_0)
# st.session_state.criteria_1 = st.text_input("Criteria for giving a score of 1", value=st.session_state.criteria_1)
# st.session_state.criteria_2 = st.text_input("Criteria for giving a score of 2", value=st.session_state.criteria_2)
# st.session_state.criteria_3 = st.text_input("Criteria for giving a score of 3", value=st.session_state.criteria_3)

# if st.button("Evaluate Teaching Plan"):
#     evaluation_criteria = "[[0]]: {}\n[[1]]: {}\n[[2]]: {}\n[[3]]: {}\n".format(st.session_state.criteria_0, 
#                                                                                 st.session_state.criteria_1, 
#                                                                                 st.session_state.criteria_2, 
#                                                                                 st.session_state.criteria_3
#                                                                                 )

#     national_curriculum = st.session_state.national_curriculum
#     teaching_plan = st.session_state.teaching_plan
#     content, evaluation_prompt = evaluate_teaching_plan_llm(evaluation_criteria, national_curriculum, teaching_plan)

#     print(evaluation_prompt)
#     st.session_state.evaluation_result = content

    

# if "evaluation_result" in st.session_state:
#     st.markdown(
#         f"""
#         <div style="
#             padding: 15px;
#             border-radius: 10px;
#             background-color: #e9ecef;
#             box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
#             ">
#             <h3 style="color: #333;">Evaluation Result</h3>
#             <p>{st.session_state.evaluation_result}</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#     """
#     <br>
#     <div style="
#         font-weight: bold;
#         font-size: 20px;
#         ">
#         The actual prompt used for evaluation
#     </div>
#     """,
#     unsafe_allow_html=True,
#     )   
    
#     st.json({evaluation_prompt}, expanded=False)


# st.header("Suggestion of combining the techniques above: evaluation of Long-Term Teaching Plan")

# col1, col2, col3 = st.columns([1, 8, 1])  # Create three columns, with the center column being larger
# with col2:
#     st.image("static/short-term-evaluation-tree.jpg", use_container_width =True)

# col1, col2, col3 = st.columns([1, 8, 1])  # Create three columns, with the center column being larger
# with col2:
#     st.image("static/long-term-evaluation-tree.jpg", use_container_width =True)