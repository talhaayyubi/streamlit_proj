import streamlit as st
import plotly.graph_objs as go
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns

# Set the page configuration
st.set_page_config(
    page_title="AI Interpretability Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# Title and Introduction
# --------------------------
st.title("🔍 AI Interpretability Dashboard")
st.markdown("""
Welcome to the **AI Interpretability Dashboard** based on the **"Transformer Circuits Thread - Circuits Updates - July 2024"** research paper. This dashboard provides an overview of the key concepts, challenges, and tools involved in understanding the inner workings of large language models like **Claude Sonnet**.
""")

# --------------------------
# Central Pathway Section
# --------------------------
st.header("📈 Central Pathway")
st.markdown("""
The central pathway outlines the progression from identifying interpretable features within the model to achieving a macroscopic understanding of its behavior.

1. **Activation Patterns**: The fundamental "variables" or concepts the model uses.
2. **Circuit Mechanisms**: The pathways and mechanisms that compute these features.
3. **Macroscopic Understanding**: Translating microscopic insights into a comprehensive overview of the model's functionality.
""")

st.subheader("🧩 Central Pathway Diagram")

# Create a NetworkX graph for the Central Pathway
G_central = nx.DiGraph()

# Add nodes with colors
G_central.add_node("Activation Patterns", color="#1f77b4")
G_central.add_node("Circuit Mechanisms", color="#1f77b4")
G_central.add_node("Macroscopic Understanding", color="#1f77b4")

# Add edges with colors
G_central.add_edge("Activation Patterns", "Circuit Mechanisms", color="#1f77b4")
G_central.add_edge("Circuit Mechanisms", "Macroscopic Understanding", color="#1f77b4")

# Define node positions
pos_central = {
    "Activation Patterns": (0, 0),
    "Circuit Mechanisms": (1, 0),
    "Macroscopic Understanding": (2, 0)
}

# Create edge traces
edge_trace_central = []
for edge in G_central.edges(data=True):
    x0, y0 = pos_central[edge[0]]
    x1, y1 = pos_central[edge[1]]
    edge_trace_central.append(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=edge[2]['color']),
            hoverinfo='none',
            mode='lines'
        )
    )

# Create node traces
node_trace_central = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=[],
        size=30,
        line=dict(width=2, color='#ffffff')
    ),
    textposition="bottom center"
)

for node in G_central.nodes(data=True):
    x, y = pos_central[node[0]]
    node_trace_central['x'] += tuple([x])
    node_trace_central['y'] += tuple([y])
    node_trace_central['text'] += tuple([node[0]])
    node_trace_central['marker']['color'] += tuple([node[1].get('color', '#FFFFFF')])  # Default to white if 'color' missing

# Create figure
fig_central = go.Figure()

# Add edges
for edge in edge_trace_central:
    fig_central.add_trace(edge)

# Add nodes
fig_central.add_trace(node_trace_central)

# Update layout
fig_central.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

st.plotly_chart(fig_central, use_container_width=True)

# --------------------------
# Five Hurdles Section
# --------------------------
st.header("🚧 Five Hurdles in Mechanistic Interpretability")
st.markdown("""
Achieving a mechanistic understanding of neural networks involves overcoming several significant challenges:

1. **Missing Features (Dark Matter)**: An enormous number of rare and potentially unextracted features that are difficult to resolve.
2. **Cross-Layer Superposition**: Features that do not clearly map to specific layers due to overlapping representations across multiple layers.
3. **Attention Superposition**: Complexities arising from multiple attention heads representing overlapping or combined features.
4. **Interference Weights**: Confusing weights resulting from feature superposition, complicating circuit analysis.
5. **Zooming Out**: Scaling microscopic understanding to a macroscopic view of the entire neural network.
""")

st.subheader("🧩 Five Hurdles Diagram")

# Create a NetworkX graph for the Five Hurdles
G_hurdles = nx.DiGraph()

# Add central pathway nodes with colors
G_hurdles.add_node("Activation Patterns", color="#1f77b4")
G_hurdles.add_node("Circuit Mechanisms", color="#1f77b4")
G_hurdles.add_node("Macroscopic Understanding", color="#1f77b4")

# Add hurdle nodes with colors
hurdles = [
    "Missing Features (Dark Matter)",
    "Cross-Layer Superposition",
    "Attention Superposition",
    "Interference Weights",
    "Zooming Out"
]

for hurdle in hurdles:
    G_hurdles.add_node(hurdle, color="#ff7f0e")

# Connect hurdles to the central pathway
G_hurdles.add_edge("Missing Features (Dark Matter)", "Activation Patterns", color="#ff7f0e")
G_hurdles.add_edge("Cross-Layer Superposition", "Circuit Mechanisms", color="#ff7f0e")
G_hurdles.add_edge("Attention Superposition", "Circuit Mechanisms", color="#ff7f0e")
G_hurdles.add_edge("Interference Weights", "Circuit Mechanisms", color="#ff7f0e")
G_hurdles.add_edge("Zooming Out", "Macroscopic Understanding", color="#ff7f0e")

# Define node positions
pos_hurdles = {
    "Activation Patterns": (0, 0),
    "Circuit Mechanisms": (1, 0),
    "Macroscopic Understanding": (2, 0),
    "Missing Features (Dark Matter)": (-1, 1),
    "Cross-Layer Superposition": (1, 1),
    "Attention Superposition": (2, 1),
    "Interference Weights": (3, 1),
    "Zooming Out": (4, 1)
}

# Create edge traces
edge_trace_hurdles = []
for edge in G_hurdles.edges(data=True):
    x0, y0 = pos_hurdles[edge[0]]
    x1, y1 = pos_hurdles[edge[1]]
    edge_trace_hurdles.append(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=edge[2]['color']),
            hoverinfo='none',
            mode='lines'
        )
    )

# Create node traces
node_trace_hurdles = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=[],
        size=20,
        line=dict(width=2, color='#ffffff')
    ),
    textposition="bottom center"
)

for node in G_hurdles.nodes(data=True):
    x, y = pos_hurdles[node[0]]
    node_trace_hurdles['x'] += tuple([x])
    node_trace_hurdles['y'] += tuple([y])
    node_trace_hurdles['text'] += tuple([node[0]])
    node_trace_hurdles['marker']['color'] += tuple([node[1].get('color', '#FFFFFF')])  # Default to white if 'color' missing

# Create figure
fig_hurdles = go.Figure()

# Add edges
for edge in edge_trace_hurdles:
    fig_hurdles.add_trace(edge)

# Add nodes
fig_hurdles.add_trace(node_trace_hurdles)

# Update layout
fig_hurdles.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

st.plotly_chart(fig_hurdles, use_container_width=True)

# --------------------------
# Tools and Techniques Section
# --------------------------
st.header("🛠️ Tools and Techniques")
st.markdown("""
To address the challenges in mechanistic interpretability, several tools and techniques are employed:

- **Dictionary Learning**: A method used to extract interpretable features from neuron activations.
- **Attention Pivot Tables**: Tools for visualizing and interpreting attention mechanisms within transformer models.
- **Feature Sensitivity Analysis**: Techniques for measuring how sensitive certain features are to specific inputs or concepts.
""")

st.subheader("🧩 Tools and Techniques Diagram")

# Create a NetworkX graph for Tools and Techniques
G_tools = nx.DiGraph()

# Add central pathway nodes with colors (ensure they are present in G_tools)
G_tools.add_node("Activation Patterns", color="#1f77b4")
G_tools.add_node("Attention Superposition", color="#ff7f0e")
G_tools.add_node("Missing Features (Dark Matter)", color="#ff7f0e")

# Add tool nodes with colors
tools = [
    "Dictionary Learning",
    "Attention Pivot Tables",
    "Feature Sensitivity Analysis"
]

for tool in tools:
    G_tools.add_node(tool, color="#2ca02c")

# Connect tools to relevant nodes
G_tools.add_edge("Dictionary Learning", "Activation Patterns", color="#2ca02c")
G_tools.add_edge("Attention Pivot Tables", "Attention Superposition", color="#2ca02c")
G_tools.add_edge("Feature Sensitivity Analysis", "Missing Features (Dark Matter)", color="#2ca02c")

# Define node positions
pos_tools = {
    "Dictionary Learning": (-2, -1),
    "Attention Pivot Tables": (3, -1),
    "Feature Sensitivity Analysis": (0, -1),
    "Activation Patterns": (0, 0),
    "Attention Superposition": (2, 1),
    "Missing Features (Dark Matter)": (-1, 1)
}

# Create edge traces
edge_trace_tools = []
for edge in G_tools.edges(data=True):
    x0, y0 = pos_tools[edge[0]]
    x1, y1 = pos_tools[edge[1]]
    edge_trace_tools.append(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=edge[2]['color']),
            hoverinfo='none',
            mode='lines'
        )
    )

# Create node traces
node_trace_tools = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=[],
        size=20,
        line=dict(width=2, color='#ffffff')
    ),
    textposition="bottom center"
)

for node in G_tools.nodes(data=True):
    x, y = pos_tools[node[0]]
    node_trace_tools['x'] += tuple([x])
    node_trace_tools['y'] += tuple([y])
    node_trace_tools['text'] += tuple([node[0]])
    node_trace_tools['marker']['color'] += tuple([node[1].get('color', '#FFFFFF')])  # Default to white if 'color' missing

# Create figure
fig_tools = go.Figure()

# Add edges
for edge in edge_trace_tools:
    fig_tools.add_trace(edge)

# Add nodes
fig_tools.add_trace(node_trace_tools)

# Update layout
fig_tools.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

st.plotly_chart(fig_tools, use_container_width=True)

# --------------------------
# Additional Concepts Section
# --------------------------
st.header("💡 Additional Concepts")
st.markdown("""
- **Linear Representation Hypothesis**: The idea that features can be represented as linear directions in activation space, allowing for vector arithmetic manipulation.
- **Multidimensional Features**: Features that span multiple dimensions, adding complexity to their representation and interaction within the model.
""")

st.subheader("🧩 Additional Concepts Diagram")

# Create a NetworkX graph for Additional Concepts
G_concepts = nx.DiGraph()

# Add central pathway node with colors
G_concepts.add_node("Activation Patterns", color="#1f77b4")

# Add concept nodes with colors
concepts = [
    "Linear Representation",
    "Multidimensional Features"
]

for concept in concepts:
    G_concepts.add_node(concept, color="#d62728")

# Connect concepts to relevant nodes
G_concepts.add_edge("Linear Representation", "Activation Patterns", color="#d62728")
G_concepts.add_edge("Multidimensional Features", "Activation Patterns", color="#d62728")

# Define node positions
pos_concepts = {
    "Linear Representation": (-3, 2),
    "Multidimensional Features": (3, 2),
    "Activation Patterns": (0, 0)
}

# Create edge traces
edge_trace_concepts = []
for edge in G_concepts.edges(data=True):
    x0, y0 = pos_concepts[edge[0]]
    x1, y1 = pos_concepts[edge[1]]
    edge_trace_concepts.append(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color=edge[2]['color']),
            hoverinfo='none',
            mode='lines'
        )
    )

# Create node traces
node_trace_concepts = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=[],
        size=20,
        line=dict(width=2, color='#ffffff')
    ),
    textposition="bottom center"
)

for node in G_concepts.nodes(data=True):
    x, y = pos_concepts[node[0]]
    node_trace_concepts['x'] += tuple([x])
    node_trace_concepts['y'] += tuple([y])
    node_trace_concepts['text'] += tuple([node[0]])
    node_trace_concepts['marker']['color'] += tuple([node[1].get('color', '#FFFFFF')])  # Default to white if 'color' missing

# Create figure
fig_concepts = go.Figure()

# Add edges
for edge in edge_trace_concepts:
    fig_concepts.add_trace(edge)

# Add nodes
fig_concepts.add_trace(node_trace_concepts)

# Update layout
fig_concepts.update_layout(
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

st.plotly_chart(fig_concepts, use_container_width=True)

# --------------------------
# New Visualizations Section
# --------------------------
st.header("📊 New Visualizations")

st.markdown("""
In this section, we present advanced visualizations that simulate interpretability metrics based on the research updates. These visualizations include a non-linear scatter plot, a bar chart, and an interactive 3D scatter plot.
""")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("⚙️ Controls")

# Slider for number of data points
num_points = st.sidebar.slider("Number of Data Points", min_value=50, max_value=500, value=100, step=50)

# --------------------------
# Random Data Generation
# --------------------------
# Generate random data for visualization
np.random.seed(42)  # For reproducibility

# Simulate AI interpretability metrics
data = pd.DataFrame({
    "Concept Strength": np.random.uniform(0, 1, num_points),
    "Circuit Efficiency": np.random.uniform(0, 1, num_points),
    "Attention Weight": np.random.uniform(0, 1, num_points),
    "Feature Sensitivity": np.random.uniform(0, 1, num_points),
    "Activation Frequency": np.random.uniform(0, 1, num_points)
})

# --------------------------
# Non-Linear Scatter Plot Section
# --------------------------
st.subheader("🔹 Scatter Plot: Concept Strength vs. Circuit Efficiency")

# Create scatter plot using Seaborn
scatter_fig = sns.scatterplot(
    data=data,
    x="Concept Strength",
    y="Circuit Efficiency",
    hue="Attention Weight",
    size="Feature Sensitivity",
    palette="viridis",
    sizes=(20, 200),
    alpha=0.7,
    edgecolor=None
)

# Convert Seaborn plot to Plotly
scatter_plot = sns.scatterplot(
    data=data,
    x="Concept Strength",
    y="Circuit Efficiency",
    hue="Attention Weight",
    size="Feature Sensitivity",
    palette="viridis",
    sizes=(20, 200),
    alpha=0.7,
    edgecolor=None
)

# Use Plotly for interactive plot
fig_scatter = go.Figure()

# Add scatter points
fig_scatter.add_trace(
    go.Scatter(
        x=data["Concept Strength"],
        y=data["Circuit Efficiency"],
        mode='markers',
        marker=dict(
            size=data["Feature Sensitivity"] * 100,  # Scale sizes
            color=data["Attention Weight"],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Attention Weight"),
            opacity=0.7
        ),
        text=[f"Feature Sensitivity: {fs:.2f}" for fs in data["Feature Sensitivity"]],
        hoverinfo='text'
    )
)

# Update layout
fig_scatter.update_layout(
    xaxis_title="Concept Strength",
    yaxis_title="Circuit Efficiency",
    title="Concept Strength vs. Circuit Efficiency",
    template="plotly_white"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------
# Bar Chart Section
# --------------------------
st.subheader("🔹 Bar Chart: Concept Strength vs. Activation Frequency")

# Create bar chart using Seaborn
bar_fig = sns.barplot(
    data=data,
    x="Concept Strength",
    y="Activation Frequency",
    palette="magma"
)

# Convert Seaborn plot to Plotly
# Use Plotly for interactive bar chart
fig_bar = go.Figure()

# Aggregate data for bar chart
bar_data = data.groupby("Concept Strength")["Activation Frequency"].mean().reset_index()

fig_bar.add_trace(
    go.Bar(
        x=bar_data["Concept Strength"],
        y=bar_data["Activation Frequency"],
        marker_color='indigo',
        opacity=0.7
    )
)

# Update layout
fig_bar.update_layout(
    xaxis_title="Concept Strength",
    yaxis_title="Activation Frequency",
    title="Concept Strength vs. Activation Frequency",
    template="plotly_white"
)

st.plotly_chart(fig_bar, use_container_width=True)

# --------------------------
# 3D Scatter Plot Section
# --------------------------
st.subheader("🔹 3D Scatter Plot: Concept Strength, Circuit Efficiency, & Attention Weight")

# Create 3D scatter plot using Plotly
fig_3d = go.Figure(data=[go.Scatter3d(
    x=data['Concept Strength'],
    y=data['Circuit Efficiency'],
    z=data['Attention Weight'],
    mode='markers',
    marker=dict(
        size=data['Feature Sensitivity'] * 20,  # Scale sizes
        color=data['Activation Frequency'],
        colorscale='Portland',
        opacity=0.8,
        colorbar=dict(title='Activation Frequency')
    ),
    text=[f"Feature Sensitivity: {fs:.2f}" for fs in data["Feature Sensitivity"]],
    hoverinfo='text'
)])

# Update layout
fig_3d.update_layout(
    scene=dict(
        xaxis_title='Concept Strength',
        yaxis_title='Circuit Efficiency',
        zaxis_title='Attention Weight'
    ),
    title="3D Scatter Plot of Concept Strength, Circuit Efficiency, & Attention Weight",
    template="plotly_white",
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------
# Conclusion Section
# --------------------------
st.header("✅ Conclusion")
st.markdown("""
Understanding the intricate mechanisms of large language models is a formidable task fraught with challenges. However, through the use of advanced interpretability techniques and a deep exploration of the models' internal circuits, significant progress is being made. This dashboard encapsulates the foundational elements of this research, highlighting the pathway to interpretability, the hurdles that need to be overcome, and the tools that facilitate this understanding.
""")

# --------------------------
# References Section
# --------------------------
st.header("📚 References")
st.markdown("""
- **Research Paper**: *"Transformer Circuits Thread - Circuits Updates - July 2024"*
- **Streamlit Documentation**: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **Plotly Documentation**: [https://plotly.com/python/](https://plotly.com/python/)
- **NetworkX Documentation**: [https://networkx.org/documentation/stable/](https://networkx.org/documentation/stable/)
- **Seaborn Documentation**: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
""")

# --------------------------
# Sidebar with Navigation
# --------------------------
st.sidebar.title("🗂️ Navigation")
st.sidebar.markdown("""
- [Central Pathway](#central-pathway)
- [Five Hurdles](#five-hurdles-in-mechanistic-interpretability)
- [Tools and Techniques](#tools-and-techniques)
- [Additional Concepts](#additional-concepts)
- [New Visualizations](#new-visualizations)
- [Conclusion](#conclusion)
- [References](#references)
""")
