import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import json
import os

st.set_page_config(page_title="CoCo - Analysis Dashboard", layout="wide")

def load_data(db_path: str):
    conn = sqlite3.connect(db_path)
    
    # Load all tables into DataFrames
    simulations = pd.read_sql_query("SELECT * FROM simulations", conn)
    agents = pd.read_sql_query("SELECT * FROM agents", conn)
    turns = pd.read_sql_query("SELECT * FROM turns", conn)
    interactions = pd.read_sql_query("SELECT * FROM interactions", conn)
    snapshots = pd.read_sql_query("SELECT * FROM agent_snapshots", conn)
    
    conn.close()
    return simulations, agents, turns, interactions, snapshots

def main():
    st.title("Collaborate || Compete (CoCo) - Interaction Analysis")
    
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    if not db_files:
        st.error("No SQLite database files found in the current directory.")
        return

    db_path = st.sidebar.selectbox("Select Simulation Database", db_files)
    
    try:
        sims, agents, turns, interactions, snapshots = load_data(db_path)
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return

    # Sidebar Filter: Simulation
    if not sims.empty:
        sim_id = st.sidebar.selectbox("Select Simulation ID", sims['simulation_id'].unique())
        sim_config = json.loads(sims[sims['simulation_id'] == sim_id]['config_json'].iloc[0])
        st.sidebar.json(sim_config)
    else:
        st.error("No simulation data found.")
        return

    tab_overview, tab_evolution, tab_interactions, tab_agents = st.tabs([
        "📊 Overview", "🧬 Evolution", "🕸️ Social Network", "🤖 Agent Deep-Dive"
    ])

    with tab_overview:
        st.header("Simulation Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Generations", agents[agents['simulation_id'] == sim_id]['generation'].max() + 1)
        col2.metric("Total Agents", len(agents[agents['simulation_id'] == sim_id]))
        col3.metric("Total Turns", len(turns[turns['simulation_id'] == sim_id]))
        col4.metric("Total Interactions", len(interactions[interactions['turn_id'].isin(turns[turns['simulation_id'] == sim_id]['turn_id'])]))

        # Population Summary
        avg_traits = agents[agents['simulation_id'] == sim_id].groupby('generation')[['collaboration_threshold', 'aggression_threshold', 'trust_level']].mean().reset_index()
        fig_pop = px.line(avg_traits, x='generation', y=['collaboration_threshold', 'aggression_threshold', 'trust_level'], 
                          title="Average Trait Evolution", labels={"value": "Level", "variable": "Trait"})
        st.plotly_chart(fig_pop, use_container_width=True)

    with tab_evolution:
        st.header("Trait Dynamics & Fitness")
        
        # Fitness vs Traits scatter
        # Merge snapshots with agents for trait info
        fitness_data = snapshots.merge(turns[['turn_id', 'generation']], on='turn_id')
        fitness_data = fitness_data.merge(agents[['agent_id', 'generation', 'collaboration_threshold', 'aggression_threshold', 'trust_level']], on=['agent_id', 'generation'])
        
        fig_fit = px.scatter(fitness_data, x='aggression_threshold', y='fitness', color='generation',
                             size='collaboration_threshold', hover_name='agent_id',
                             title="Fitness vs Aggression (Size = Collaboration)")
        st.plotly_chart(fig_fit, use_container_width=True)

    with tab_interactions:
        st.header("The Social Graph")
        
        gen_to_show = st.slider("Select Generation for Network View", 0, int(agents['generation'].max()), 0)
        
        # Filter interactions for selected generation
        gen_turns = turns[(turns['simulation_id'] == sim_id) & (turns['generation'] == gen_to_show)]['turn_id']
        gen_interactions = interactions[interactions['turn_id'].isin(gen_turns)]
        
        if not gen_interactions.empty:
            # Create NetworkX graph
            G = nx.DiGraph()
            for _, row in gen_interactions.iterrows():
                source = row['source_id']
                target = row['target_id']
                
                # Only add edges for interactions with a valid target (sharing/stealing)
                if source and target and str(target).strip():
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += 1
                    else:
                        G.add_edge(source, target, weight=1, type=row['interaction_type'])
                elif source:
                    # Just add the node if it doesn't exist yet (for agents that only 'pass')
                    if not G.has_node(source):
                        G.add_node(source)

            # Plot using Plotly
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G)
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                    textposition="top center",
                    marker=dict(showscale=True, colorscale='Viridis', size=20, color=[], 
                                colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'))
                )
                
                node_adjacencies = []
                for node in G.nodes():
                    node_adjacencies.append(len(list(G.neighbors(node))))
                node_trace.marker.color = node_adjacencies

                fig_graph = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(title=f'Interaction Network - Generation {gen_to_show}',
                                showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                
                st.plotly_chart(fig_graph, use_container_width=True)
            else:
                st.info("No directed interactions (stealing/sharing) to display in the graph for this generation.")
            
            st.subheader("Action Reasoning Log")
            st.dataframe(gen_interactions[['source_id', 'target_id', 'interaction_type', 'success', 'reasoning']].dropna(subset=['reasoning']), use_container_width=True)
        else:
            st.info("No interactions recorded for this generation.")

    with tab_agents:
        st.header("Agent Deep-Dive")
        
        agent_list = agents[agents['simulation_id'] == sim_id]['agent_id'].unique()
        selected_agent = st.selectbox("Select Agent", agent_list)
        
        agent_info = agents[(agents['agent_id'] == selected_agent) & (agents['simulation_id'] == sim_id)].iloc[0]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Profile")
            st.write(f"**Agent ID:** {selected_agent}")
            st.write(f"**Generation:** {agent_info['generation']}")
            st.write(f"**Parent ID:** {agent_info['parent_id'] or 'Generation 0'}")
            st.write(f"**Model:** {agent_info['model']}")
            
        with col_b:
            st.subheader("Traits")
            st.progress(agent_info['collaboration_threshold'], text=f"Collaboration: {agent_info['collaboration_threshold']:.2f}")
            st.progress(agent_info['aggression_threshold'], text=f"Aggression: {agent_info['aggression_threshold']:.2f}")
            st.progress(agent_info['trust_level'], text=f"Trust: {agent_info['trust_level']:.2f}")

        st.subheader("Performance & Resources over Turns")
        agent_snaps = snapshots[snapshots['agent_id'] == selected_agent].merge(turns[['turn_id', 'turn_number']], on='turn_id')
        agent_snaps['tokens'] = agent_snaps['resources_json'].apply(lambda x: json.loads(x).get('token', 0))
        
        fig_tokens = px.line(agent_snaps, x='turn_number', y='tokens', title=f"Token Balance History - {selected_agent}")
        st.plotly_chart(fig_tokens, use_container_width=True)
        
        st.subheader("Private Reasoning Trace")
        st.dataframe(agent_snaps[['turn_number', 'memory_json', 'fitness']], use_container_width=True)

if __name__ == "__main__":
    main()
