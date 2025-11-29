"""
MLB MVP Predictor - Streamlit Dashboard
WGU C964 Computer Science Capstone
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="MLB MVP Predictor",
    page_icon="⚾",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    model = joblib.load('models/v6.directConfidence/pairwise_model.joblib')
    feature_cols = joblib.load('models/v6.directConfidence/feature_cols.joblib')
    return model, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('models/v6.directConfidence/mvp_dataset_v6.csv')
    predictions = pd.read_csv('models/v6.directConfidence/predictions.csv')
    importance = pd.read_csv('models/v6.directConfidence/feature_importance.csv')
    return df, predictions, importance

model, feature_cols = load_model()
df, predictions, importance = load_data()

# Sidebar
st.sidebar.title("MLB MVP Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Historical Analysis", "Year Explorer", "Robbery Index", "2025: Judge vs Raleigh", "Make Prediction", "Model Versions"]
)

# Helper function for confidence scoring (v6 method)
def score_candidates(candidates_df, model, feature_cols):
    """Score each candidate by average H2H win probability vs all others (0-100%)."""
    features = candidates_df[feature_cols].values
    n = len(features)
    scores = np.zeros(n)

    for i in range(n):
        win_probs = []
        for j in range(n):
            if i != j:
                diff = features[i] - features[j]
                prob = model.predict_proba(diff.reshape(1, -1))[0, 1]
                win_probs.append(prob)
        # Average win probability as percentage
        scores[i] = np.mean(win_probs) * 100 if win_probs else 50

    return scores

# ============== OVERVIEW PAGE ==============
if page == "Overview":
    st.title("MLB MVP Predictor")
    st.markdown("### Pairwise Learning-to-Rank Model for MVP Selection")

    col1, col2, col3, col4 = st.columns(4)

    total = len(predictions)
    correct = predictions['correct'].sum()
    accuracy = 100 * correct / total

    col1.metric("Overall Accuracy", f"{accuracy:.0f}%", f"{correct}/{total} MVPs")
    col2.metric("Years Covered", "1945-2024", "80 seasons")
    col3.metric("Features Used", len(feature_cols), "player stats")
    col4.metric("Controversial Picks", "6", "Model differs from voters")

    st.markdown("---")

    # Model description
    st.markdown("### How It Works")
    st.markdown("""
    The model uses **pairwise learning-to-rank** to compare MVP candidates head-to-head.
    Instead of predicting absolute values, it learns which player should win when
    directly compared to another, then aggregates these comparisons to rank all candidates.
    This approach mirrors how actual MVP voting works: voters compare candidates and
    decide who is more valuable.
    """)

    st.markdown("---")

    # Feature importance chart
    st.markdown("### Feature Importance")
    feature_labels = {
        'total_WAR': 'Total WAR',
        'team_wins': 'Team Wins',
        'led_rbi': 'Led League in RBI',
        'RBI': 'RBI',
        'TB': 'Total Bases',
        'milestones_hit': 'Milestones Hit',
        'SB': 'Stolen Bases',
        'runs_created': 'Runs Created',
        'H': 'Hits',
        'prev_vote_share': 'Previous Vote Share'
    }
    importance_display = importance.head(10).copy()
    importance_display['feature'] = importance_display['feature'].map(
        lambda x: feature_labels.get(x, x)
    )
    fig = px.bar(
        importance_display,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features',
        color='importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Full feature importance table
    st.markdown("### All Features by Importance")
    all_feature_labels = {
        'total_WAR': 'Total WAR', 'team_wins': 'Team Wins', 'led_rbi': 'Led League in RBI',
        'RBI': 'RBI', 'TB': 'Total Bases', 'milestones_hit': 'Milestones Hit',
        'SB': 'Stolen Bases', 'runs_created': 'Runs Created', 'H': 'Hits',
        'prev_vote_share': 'Previous Vote Share', 'league_titles': 'League Titles',
        'BB': 'Walks', 'won_pennant': 'Won Pennant', 'AB': 'At Bats', '3B': 'Triples',
        'BA': 'Batting Average', 'R': 'Runs', 'OBP': 'On Base Percentage',
        'SLG': 'Slugging', 'won_ws': 'Won World Series', 'OPS': 'OPS',
        'made_playoffs': 'Made Playoffs', 'gold_glove': 'Gold Glove', 'p_K9': 'K/9 (Pitching)',
        'prev_mvp_wins': 'Previous MVP Wins', 'age': 'Age', '2B': 'Doubles', 'G': 'Games',
        'led_hr': 'Led League in HR', 'led_ops': 'Led League in OPS', 'HR': 'Home Runs',
        'XBH': 'Extra Base Hits', 'p_W': 'Wins (Pitching)', 'WAR': 'WAR',
        'ISO': 'Isolated Power', 'p_WHIP_inv': 'WHIP Inverse (Pitching)',
        'led_war': 'Led League in WAR', 'is_allstar': 'All Star', 'big_market': 'Big Market Team',
        'hit_40hr': 'Hit 40+ HR', 'first_serious_run': 'First Serious MVP Run',
        'led_ba': 'Led League in BA', 'p_WAR_est': 'Pitching WAR Estimate', 'IP': 'Innings Pitched',
        'p_ERA_inv': 'ERA Inverse (Pitching)', 'p_SO': 'Strikeouts (Pitching)',
        'prime_age': 'Prime Age', 'hit_100rbi': 'Hit 100+ RBI', 'is_pitcher': 'Is Pitcher',
        'quality_pitcher': 'Quality Pitcher', 'hit_300ba': 'Hit .300+ BA',
        'silver_slugger': 'Silver Slugger', 'is_starter': 'Is Starter',
        'is_two_way': 'Two Way Player', 'elite_pitcher': 'Elite Pitcher'
    }
    importance_table = importance.copy()
    importance_table['Feature'] = importance_table['feature'].map(
        lambda x: all_feature_labels.get(x, x)
    )
    importance_table['Importance'] = (importance_table['importance'] * 100).round(2)
    importance_table['Cumulative'] = importance_table['importance'].cumsum() * 100
    importance_table['Cumulative'] = importance_table['Cumulative'].round(2)
    importance_table = importance_table[['Feature', 'Importance', 'Cumulative']]
    importance_table.columns = ['Feature', 'Importance (%)', 'Cumulative (%)']
    st.dataframe(importance_table, use_container_width=True, hide_index=True)

# ============== MODEL VERSIONS PAGE ==============
elif page == "Model Versions":
    st.title("Model Versions")
    st.markdown("### Development History")
    st.markdown("""
    This project went through several iterations to achieve 96% accuracy. Each version
    introduced new techniques or features that improved prediction quality.

    Note that 100% accuracy is not the goal. Perfect accuracy would imply that MVP voters
    always get it right and are never influenced by narrative, recency bias, or voter fatigue.
    Our model identifies cases where the stats clearly favor one player but voters chose another.
    """)

    # Version data
    version_data = pd.DataFrame({
        'Version': ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'],
        'Accuracy': [34, 38, 59, 83, 87, 89, 96],
        'Name': ['rawLahman', 'fWar', 'reinforcementLearning', 'pairwiseRanking',
                 'enhancedPairwise', 'withPitching', 'directConfidence'],
        'Description': [
            'Basic stats from Lahman database only',
            'Added FanGraphs WAR as primary metric',
            'Reward-based learning for sequential decisions',
            'Head-to-head comparison approach',
            'Added narrative features (All-Star, playoffs)',
            'Included pitching stats for two-way players',
            'Tournament-style elimination with confidence scoring'
        ]
    })

    # Line chart of accuracy progression
    st.markdown("### Accuracy Progression")
    fig = px.line(
        version_data,
        x='Version',
        y='Accuracy',
        markers=True,
        title='Model Accuracy Over Iterations',
        labels={'Accuracy': 'Accuracy (%)'}
    )
    fig.update_traces(line=dict(width=3, color='#1f77b4'), marker=dict(size=12))
    fig.add_hline(y=90, line_dash="dash", line_color="green",
                  annotation_text="90% Target")
    fig.update_layout(margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    # Version details table
    st.markdown("### Version Details")
    display_df = version_data[['Version', 'Name', 'Accuracy', 'Description']].copy()
    display_df['Accuracy'] = display_df['Accuracy'].astype(str) + '%'
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Key improvements
    st.markdown("### Key Improvements")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Biggest Jump: v2 to v3 (+24%)**")
        st.markdown("""
        Switching from reinforcement learning to pairwise ranking was the breakthrough.
        Instead of learning abstract reward signals, the model now directly learns
        "Player A beats Player B" which matches how MVP voting actually works.
        """)

    with col2:
        st.markdown("**Final Push: v5 to v6 (+7%)**")
        st.markdown("""
        Earlier versions suffered from Condorcet cycles, where Player A beats B, B beats C,
        but C beats A. The v6 model solves this with tournament-style comparison where each
        matchup produces a win probability, and candidates are ranked by average dominance
        across all head-to-head comparisons.
        """)

# ============== HISTORICAL ANALYSIS PAGE ==============
elif page == "Historical Analysis":
    st.title("Historical Analysis")

    # Decade breakdown
    st.markdown("### Accuracy by Decade")
    decade_stats = predictions.groupby('decade').agg(
        correct=('correct', 'sum'),
        total=('correct', 'count')
    ).reset_index()
    decade_stats['accuracy'] = 100 * decade_stats['correct'] / decade_stats['total']

    fig = px.bar(
        decade_stats,
        x='decade',
        y='accuracy',
        title='MVP Prediction Accuracy by Decade',
        labels={'decade': 'Decade', 'accuracy': 'Accuracy (%)'},
        color='accuracy',
        color_continuous_scale='RdYlGn',
        text=decade_stats.apply(lambda r: f"{int(r['correct'])}/{int(r['total'])}", axis=1)
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    # Position breakdown
    st.markdown("### Accuracy by Position")
    pos_stats = predictions.groupby('pos').agg(
        correct=('correct', 'sum'),
        total=('correct', 'count')
    ).reset_index()
    pos_stats['accuracy'] = 100 * pos_stats['correct'] / pos_stats['total']
    pos_stats = pos_stats.sort_values('total', ascending=False)

    fig = px.bar(
        pos_stats,
        x='pos',
        y='accuracy',
        title='MVP Prediction Accuracy by Position',
        labels={'pos': 'Position', 'accuracy': 'Accuracy (%)'},
        color='accuracy',
        color_continuous_scale='RdYlGn',
        text=pos_stats.apply(lambda r: f"{int(r['correct'])}/{int(r['total'])}", axis=1)
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)

    # Misses analysis
    st.markdown("### Notable Misses")
    misses = predictions[~predictions['correct']].copy()
    misses['display'] = misses.apply(
        lambda r: f"{int(r['year'])} {r['lg']}: Predicted **{r['predicted']}**, Actual **{r['actual']}** ({r['pos']})",
        axis=1
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Recent Misses (2010+)**")
        recent_misses = misses[misses['year'] >= 2010].sort_values('year', ascending=False)
        for _, row in recent_misses.iterrows():
            st.markdown(f"- {row['display']}")

    with col2:
        st.markdown("**Classic Era Misses (Pre-1970)**")
        classic_misses = misses[misses['year'] < 1970].sort_values('year', ascending=False).head(10)
        for _, row in classic_misses.iterrows():
            st.markdown(f"- {row['display']}")

# ============== YEAR EXPLORER PAGE ==============
elif page == "Year Explorer":
    st.title("Year Explorer")

    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Select Year", sorted(df['yearID'].unique(), reverse=True))
    with col2:
        league = st.selectbox("Select League", ['AL', 'NL'])

    # Filter data
    year_data = df[(df['yearID'] == year) & (df['lgID'] == league)].copy()

    if len(year_data) == 0:
        st.warning(f"No data available for {year} {league}")
    else:
        # Get actual MVP
        actual_mvp = year_data[year_data['is_winner'] == 1]

        # Get model's predicted winner for this year
        pred_result = predictions[(predictions['year'] == year) & (predictions['lg'] == league)]
        model_pick_name = pred_result.iloc[0]['predicted'] if len(pred_result) > 0 else None
        model_pick_row = year_data[year_data['player_name'] == model_pick_name]

        # Get top candidates by OPS
        top_candidates = year_data.nlargest(20, 'OPS').copy()

        # Ensure actual MVP is included
        if len(actual_mvp) > 0 and actual_mvp.index[0] not in top_candidates.index:
            top_candidates = pd.concat([top_candidates, actual_mvp])

        # Ensure model's predicted winner is included
        if len(model_pick_row) > 0 and model_pick_row.index[0] not in top_candidates.index:
            top_candidates = pd.concat([top_candidates, model_pick_row])

        # Score candidates using v6 confidence method
        scores = score_candidates(top_candidates, model, feature_cols)
        top_candidates['confidence'] = scores
        top_candidates = top_candidates.sort_values('confidence', ascending=False)

        # Display results
        st.markdown(f"### {year} {league} MVP Race")

        if len(pred_result) > 0:
            result = pred_result.iloc[0]
            if result['correct']:
                st.success(f"**Correctly predicted**: {result['actual']}")
            else:
                st.error(f"**Predicted**: {result['predicted']} | **Actual**: {result['actual']}")

        # Show top 10 candidates
        st.markdown("### Top 10 Candidates")

        # Mark model's pick and sort with model pick first
        model_pick = model_pick_name

        display_cols = ['player_name', 'primary_pos', 'WAR', 'OPS', 'HR', 'RBI', 'team_wins', 'confidence', 'is_winner']
        display_df = top_candidates[display_cols].head(10).copy()
        display_df.columns = ['Player', 'Pos', 'WAR', 'OPS', 'HR', 'RBI', 'Team Wins', 'Confidence', 'Actual MVP']

        # Add Model Pick column and sort so model pick is first
        display_df['is_model_pick'] = display_df['Player'] == model_pick
        display_df = display_df.sort_values(['is_model_pick', 'Confidence'], ascending=[False, False])
        display_df['Model Pick'] = display_df['is_model_pick'].map({True: 'Yes', False: ''})
        display_df = display_df.drop(columns=['is_model_pick'])

        display_df['Actual MVP'] = display_df['Actual MVP'].map({1: 'Yes', 0: ''})
        display_df['Confidence'] = display_df['Confidence'].round(1).astype(str) + '%'
        display_df['WAR'] = display_df['WAR'].round(1)

        # Reorder columns
        display_df = display_df[['Player', 'Pos', 'WAR', 'OPS', 'HR', 'RBI', 'Team Wins', 'Confidence', 'Model Pick', 'Actual MVP']]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Scatter plot
        st.markdown("### WAR vs OPS (Candidates)")
        fig = px.scatter(
            top_candidates,
            x='WAR',
            y='OPS',
            size='confidence',
            color='confidence',
            hover_name='player_name',
            hover_data=['HR', 'RBI', 'team_wins', 'primary_pos'],
            title=f'{year} {league} MVP Candidates',
            color_continuous_scale='Viridis',
            labels={'confidence': 'Confidence (%)'}
        )

        # Highlight actual MVP
        if len(actual_mvp) > 0:
            mvp = actual_mvp.iloc[0]
            fig.add_trace(go.Scatter(
                x=[mvp['WAR']],
                y=[mvp['OPS']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name=f"Actual MVP: {mvp['player_name']}"
            ))

        st.plotly_chart(fig, use_container_width=True)

# ============== ROBBERY INDEX PAGE ==============
elif page == "Robbery Index":
    st.title("Robbery Index")
    st.markdown("""
    ### When Voters Got It Wrong

    The Robbery Index shows cases where our model **disagreed** with actual MVP voters.
    The H2H% indicates how strongly our model's pick would beat the actual winner head-to-head.
    """)

    # Get misses from predictions
    misses = predictions[~predictions['correct']].copy()

    if 'predicted_h2h' in misses.columns:
        misses = misses.sort_values('predicted_h2h', ascending=False)

        # Create robbery index visualization - horizontal bar chart
        st.markdown("### Robbery Index Rankings")

        misses['label'] = misses.apply(
            lambda r: f"{int(r['year'])} {r['lg']}: {r['predicted']} over {r['actual']}",
            axis=1
        )

        fig = px.bar(
            misses,
            y='label',
            x='predicted_h2h',
            orientation='h',
            title='Head-to-Head Dominance of Model Pick vs Actual Winner',
            labels={'predicted_h2h': 'H2H Win Probability (%)', 'label': ''},
            color='predicted_h2h',
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=400
        )
        fig.add_vline(x=50, line_dash="dash", line_color="gray",
                      annotation_text="50% = Toss-up")
        st.plotly_chart(fig, use_container_width=True)

        # Timeline scatter plot - when did robberies happen?
        st.markdown("### Robberies Over Time")
        fig2 = px.scatter(
            misses,
            x='year',
            y='predicted_h2h',
            color='lg',
            size='predicted_h2h',
            hover_name='predicted',
            hover_data={'actual': True, 'year': True, 'lg': True},
            title='MVP Robberies by Year and League',
            labels={'predicted_h2h': 'H2H Dominance (%)', 'year': 'Year', 'lg': 'League'}
        )
        fig2.add_hline(y=50, line_dash="dash", line_color="gray")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # Detailed table
        st.markdown("### Detailed Robbery Analysis")

        display_df = misses[['year', 'lg', 'predicted', 'actual', 'predicted_h2h', 'pos']].copy()
        display_df.columns = ['Year', 'League', 'Should Have Won', 'Actual Winner', 'H2H %', 'Position']
        display_df['H2H %'] = display_df['H2H %'].round(1).astype(str) + '%'

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Notable robbery callouts
        st.markdown("### Notable Robberies")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Biggest Robbery:**")
            biggest = misses.iloc[0]
            st.markdown(f"""
            - **{int(biggest['year'])} {biggest['lg']}**
            - Model pick: **{biggest['predicted']}**
            - Actual: {biggest['actual']}
            - H2H dominance: **{biggest['predicted_h2h']:.1f}%**
            """)

        with col2:
            st.markdown("**Closest Call:**")
            closest = misses.iloc[-1]
            st.markdown(f"""
            - **{int(closest['year'])} {closest['lg']}**
            - Model pick: **{closest['predicted']}**
            - Actual: {closest['actual']}
            - H2H dominance: **{closest['predicted_h2h']:.1f}%**
            """)
    else:
        st.warning("H2H data not available in predictions. Please regenerate with v6 model.")

# ============== 2025 JUDGE VS RALEIGH PAGE ==============
elif page == "2025: Judge vs Raleigh":
    st.title("2025 AL MVP: Judge vs Raleigh")
    st.markdown("""
    ### The Great 2025 AL MVP Debate

    The 2025 AL MVP race sparked one of the most heated debates in recent memory. Aaron Judge
    claimed an incredible OPS along with a batting title, while Cal Raleigh hit the important
    60 HR milestone, a record for the physically demanding position of catcher. Analysts and
    fans were divided over WAR calculations, positional value, and what "Most Valuable" truly
    means.

    **Was it actually close?** We can use our model to cut through the noise and see what the
    data says about this matchup.
    """)

    # Load actual 2025 stats from test case file
    import json
    with open('data/2025_test_case.json', 'r') as f:
        test_case = json.load(f)

    judge_data = test_case['candidates']['Aaron Judge']
    raleigh_data = test_case['candidates']['Cal Raleigh']

    st.markdown("### Actual 2025 Stats")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Aaron Judge** (NYY)")
        st.write(f"fWAR: **{judge_data['fWAR']}** | bWAR: {judge_data['bWAR']}")
        st.write(f"HR: **{judge_data['HR']}** | RBI: **{judge_data['RBI']}**")
        st.write(f"AVG/OBP/SLG: **{judge_data['BA']:.3f}/{judge_data['OBP']:.3f}/{judge_data['SLG']:.3f}**")
        st.write(f"OPS: **{judge_data['OPS']:.3f}**")
        st.write(f"Team Wins: **{judge_data['team_wins']}**")
        st.caption(judge_data['notes'])

    with col2:
        st.markdown("**Cal Raleigh** (SEA)")
        st.write(f"fWAR: **{raleigh_data['fWAR']}** | bWAR: {raleigh_data['bWAR']}")
        st.write(f"HR: **{raleigh_data['HR']}** | RBI: **{raleigh_data['RBI']}**")
        st.write(f"AVG/OBP/SLG: **{raleigh_data['BA']:.3f}/{raleigh_data['OBP']:.3f}/{raleigh_data['SLG']:.3f}**")
        st.write(f"OPS: **{raleigh_data['OPS']:.3f}**")
        st.write(f"Team Wins: **{raleigh_data['team_wins']}**")
        st.caption(raleigh_data['notes'])

    # Build feature vectors from actual stats (using fWAR)
    def build_features(data, is_judge=False):
        tb = data['H'] + data['2B'] + 2*data['3B'] + 3*data['HR']
        xbh = data['2B'] + data['3B'] + data['HR']
        runs_created = ((data['H'] + data['BB']) * tb) / (data['AB'] + data['BB']) if (data['AB'] + data['BB']) > 0 else 0
        iso = data['SLG'] - data['BA']
        return {
            'G': data['G'], 'AB': data['AB'], 'R': data['R'], 'H': data['H'],
            '2B': data['2B'], '3B': data['3B'], 'HR': data['HR'], 'RBI': data['RBI'],
            'SB': data['SB'], 'BB': data['BB'], 'BA': data['BA'], 'OBP': data['OBP'],
            'SLG': data['SLG'], 'OPS': data['OPS'], 'ISO': iso, 'TB': tb, 'XBH': xbh,
            'runs_created': runs_created, 'WAR': data['fWAR'], 'team_wins': data['team_wins'],
            'prev_vote_share': 1.0 if is_judge else 0.0,  # Judge won 2024 MVP
            'big_market': 1 if is_judge else 0,
            'is_allstar': data['is_allstar'], 'gold_glove': 0, 'silver_slugger': 1,
            'made_playoffs': data['made_playoffs'], 'won_pennant': 0, 'won_ws': 0,
            'age': 33 if is_judge else 28, 'prime_age': 0 if is_judge else 1,
            'led_hr': 0, 'led_rbi': 0, 'led_ba': 1 if is_judge else 0,
            'led_ops': 1 if is_judge else 0, 'led_war': 1 if is_judge else 0,
            'league_titles': 0, 'hit_40hr': 1, 'hit_100rbi': 1,
            'hit_300ba': 1 if is_judge else 0, 'milestones_hit': 3 if is_judge else 2,
            'prev_mvp_wins': 2 if is_judge else 0, 'first_serious_run': 0 if is_judge else 1,
            'p_W': 0, 'p_SO': 0, 'IP': 0, 'p_ERA_inv': 0, 'p_WHIP_inv': 0, 'p_K9': 0,
            'is_pitcher': 0, 'is_starter': 0, 'is_two_way': 0, 'quality_pitcher': 0, 'elite_pitcher': 0,
            'p_WAR_est': 0, 'total_WAR': data['fWAR']
        }

    judge_stats = build_features(judge_data, is_judge=True)
    raleigh_stats = build_features(raleigh_data, is_judge=False)

    st.markdown("---")

    # Calculate H2H
    st.markdown("### Head-to-Head Analysis")

    judge_features = np.array([[judge_stats[col] for col in feature_cols]])
    raleigh_features = np.array([[raleigh_stats[col] for col in feature_cols]])

    # Judge vs Raleigh
    diff = judge_features - raleigh_features
    judge_h2h = model.predict_proba(diff)[0, 1] * 100
    raleigh_h2h = 100 - judge_h2h

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Judge wins H2H", f"{judge_h2h:.1f}%")
    with col2:
        st.metric("Raleigh wins H2H", f"{raleigh_h2h:.1f}%")

    # Visual bar
    fig = go.Figure(go.Bar(
        x=[judge_h2h, raleigh_h2h],
        y=['Aaron Judge', 'Cal Raleigh'],
        orientation='h',
        marker_color=['#003087', '#005C5C'],  # Yankees blue, Mariners teal
        text=[f'{judge_h2h:.1f}%', f'{raleigh_h2h:.1f}%'],
        textposition='inside'
    ))
    fig.update_layout(
        title='Head-to-Head Win Probability',
        xaxis_title='Win Probability (%)',
        showlegend=False,
        height=200
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # Robbery assessment
    st.markdown("### If Raleigh Wins MVP...")

    if raleigh_h2h < 50:
        robbery_severity = judge_h2h
        st.error(f"""
        **This would be a ROBBERY**

        Judge dominates the H2H matchup at **{judge_h2h:.1f}%** win probability.

        For comparison with historical robberies:
        """)

        # Compare to historical robberies
        misses = predictions[~predictions['correct']].copy()
        if 'predicted_h2h' in misses.columns:
            misses = misses.sort_values('predicted_h2h', ascending=False)

            # Where would this rank? (count how many have HIGHER H2H, then add 1)
            rank = sum(misses['predicted_h2h'] > judge_h2h) + 1
            total_robberies = len(misses)

            st.markdown(f"""
            - This would rank **#{rank}** out of {total_robberies + 1} total robberies
            - Judge's {judge_h2h:.1f}% H2H dominance vs historical max of {misses['predicted_h2h'].max():.1f}%
            """)

            # Show comparison table
            comparison = pd.DataFrame([
                {'Year': '2025 (hypothetical)', 'Should Win': 'Aaron Judge', 'Actual': 'Cal Raleigh', 'H2H %': f'{judge_h2h:.1f}%'}
            ])
            for _, row in misses.head(5).iterrows():
                comparison = pd.concat([comparison, pd.DataFrame([{
                    'Year': f"{int(row['year'])} {row['lg']}",
                    'Should Win': row['predicted'],
                    'Actual': row['actual'],
                    'H2H %': f"{row['predicted_h2h']:.1f}%"
                }])], ignore_index=True)

            st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.success(f"""
        **Not a robbery** - Raleigh would legitimately beat Judge in H2H ({raleigh_h2h:.1f}%)
        """)

# ============== MAKE PREDICTION PAGE ==============
elif page == "Make Prediction":
    st.title("Make a Prediction")
    st.markdown("Enter player statistics to see how they would rank against a historical MVP race.")

    # Year and league selection
    st.markdown("### Select Comparison Year")
    col_year, col_league = st.columns(2)
    with col_year:
        available_years = sorted(df['yearID'].unique(), reverse=True)
        selected_year = st.selectbox("Year", available_years, index=0)
    with col_league:
        selected_league = st.selectbox("League", ["AL", "NL"])

    st.markdown("### Enter Player Stats")

    # Presets
    preset = st.selectbox("Load Preset", [
        "Custom",
        "Power Hitter",
        "Two-Way Player",
        "Five-Tool Player"
    ])

    # Set defaults based on preset
    if preset == "Power Hitter":
        # Based on Aaron Judge 2022
        defaults = {'g': 157, 'ab': 570, 'r': 133, 'h': 177, 'doubles': 28, 'triples': 0,
                    'hr': 62, 'rbi': 131, 'sb': 16, 'bb': 111, 'war': 11.1, 'team_wins': 99,
                    'is_allstar': 1, 'made_playoffs': 1, 'age': 30, 'two_way': False}
    elif preset == "Two-Way Player":
        # Based on Shohei Ohtani 2023
        defaults = {'g': 135, 'ab': 497, 'r': 102, 'h': 151, 'doubles': 26, 'triples': 8,
                    'hr': 44, 'rbi': 95, 'sb': 20, 'bb': 91, 'war': 6.6, 'team_wins': 73,
                    'is_allstar': 1, 'made_playoffs': 0, 'age': 29, 'two_way': True,
                    'p_W': 10, 'p_SO': 167, 'IP': 132.0, 'p_ERA': 3.14, 'p_WHIP': 1.06, 'p_WAR': 2.7}
    elif preset == "Five-Tool Player":
        # Based on Barry Bonds 1993
        defaults = {'g': 159, 'ab': 539, 'r': 129, 'h': 181, 'doubles': 38, 'triples': 4,
                    'hr': 46, 'rbi': 123, 'sb': 29, 'bb': 126, 'war': 10.5, 'team_wins': 103,
                    'is_allstar': 1, 'made_playoffs': 0, 'age': 29, 'two_way': False}
    else:  # Custom
        defaults = {'g': 150, 'ab': 550, 'r': 100, 'h': 175, 'doubles': 35, 'triples': 3,
                    'hr': 30, 'rbi': 100, 'sb': 10, 'bb': 70, 'war': 6.0, 'team_wins': 90,
                    'is_allstar': 0, 'made_playoffs': 0, 'age': 28, 'two_way': False}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Counting Stats**")
        g = st.number_input("Games (G)", 0, 162, defaults['g'])
        ab = st.number_input("At Bats (AB)", 0, 700, defaults['ab'])
        r = st.number_input("Runs (R)", 0, 200, defaults['r'])
        h = st.number_input("Hits (H)", 0, 300, defaults['h'])
        doubles = st.number_input("Doubles (2B)", 0, 70, defaults['doubles'])
        triples = st.number_input("Triples (3B)", 0, 25, defaults['triples'])
        hr = st.number_input("Home Runs (HR)", 0, 75, defaults['hr'])
        rbi = st.number_input("RBI", 0, 200, defaults['rbi'])

    with col2:
        st.markdown("**More Counting Stats**")
        sb = st.number_input("Stolen Bases (SB)", 0, 100, defaults['sb'])
        bb = st.number_input("Walks (BB)", 0, 200, defaults['bb'])

        # Calculate slash line from counting stats
        ba = h / ab if ab > 0 else 0.0
        obp = (h + bb) / (ab + bb) if (ab + bb) > 0 else 0.0
        tb = h + doubles + 2*triples + 3*hr
        slg = tb / ab if ab > 0 else 0.0
        ops = obp + slg
        iso = slg - ba

        st.markdown("**Calculated Slash Line**")
        st.metric("BA", f"{ba:.3f}")
        st.metric("OBP", f"{obp:.3f}")
        st.metric("SLG", f"{slg:.3f}")
        st.metric("OPS", f"{ops:.3f}")

    with col3:
        st.markdown("**Advanced & Context**")
        war = st.number_input("WAR", 0.0, 15.0, defaults['war'], format="%.1f")
        team_wins = st.number_input("Team Wins", 50, 120, defaults['team_wins'])
        is_allstar = st.selectbox("All-Star?", [0, 1], index=defaults['is_allstar'], format_func=lambda x: "Yes" if x else "No")
        made_playoffs = st.selectbox("Made Playoffs?", [0, 1], index=defaults['made_playoffs'], format_func=lambda x: "Yes" if x else "No")
        age = st.number_input("Age", 18, 45, defaults['age'])

    # Two-way player / pitching section
    st.markdown("### Pitching Stats (Optional)")
    is_two_way = st.checkbox("Two-Way Player (also pitches)", value=defaults['two_way'])

    if is_two_way:
        p_defaults = {
            'p_W': defaults.get('p_W', 10),
            'p_SO': defaults.get('p_SO', 150),
            'IP': defaults.get('IP', 100.0),
            'p_ERA': defaults.get('p_ERA', 3.50),
            'p_WHIP': defaults.get('p_WHIP', 1.20),
            'p_WAR': defaults.get('p_WAR', 2.0)
        }
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            p_W = st.number_input("Wins (W)", 0, 25, p_defaults['p_W'])
            p_SO = st.number_input("Strikeouts (K)", 0, 350, p_defaults['p_SO'])
            IP = st.number_input("Innings Pitched", 0.0, 250.0, p_defaults['IP'], format="%.1f")
        with pcol2:
            p_ERA = st.number_input("ERA", 0.0, 10.0, p_defaults['p_ERA'], format="%.2f")
            p_WHIP = st.number_input("WHIP", 0.0, 3.0, p_defaults['p_WHIP'], format="%.2f")
        with pcol3:
            p_WAR = st.number_input("Pitching WAR", 0.0, 12.0, p_defaults['p_WAR'], format="%.1f")
    else:
        p_W, p_SO, IP, p_ERA, p_WHIP, p_WAR = 0, 0, 0.0, 0.0, 0.0, 0.0

    st.markdown("---")

    if st.button("Calculate MVP Score", type="primary"):
        # Get top 3 candidates from selected year/league by vote share
        year_candidates = df[(df['yearID'] == selected_year) & (df['lgID'] == selected_league)].copy()
        year_candidates = year_candidates.nlargest(3, 'vote_share')

        if len(year_candidates) == 0:
            st.error(f"No data available for {selected_year} {selected_league}")
        else:
            # Build player stats dict with all required features
            xbh = doubles + triples + hr
            runs_created = ((h + bb) * tb) / (ab + bb) if (ab + bb) > 0 else 0

            # Create a row matching the dataset structure
            player_stats = {col: 0 for col in feature_cols}  # Initialize all to 0

            # Calculate pitching derived stats
            p_ERA_inv = 1 / p_ERA if p_ERA > 0 else 0
            p_WHIP_inv = 1 / p_WHIP if p_WHIP > 0 else 0
            p_K9 = (p_SO / IP * 9) if IP > 0 else 0
            total_WAR = war + p_WAR if is_two_way else war
            prime_age = 1 if 26 <= age <= 32 else 0

            # Fill in the stats we have
            player_stats.update({
                'G': g, 'AB': ab, 'R': r, 'H': h, '2B': doubles, '3B': triples,
                'HR': hr, 'RBI': rbi, 'SB': sb, 'BB': bb,
                'BA': ba, 'OBP': obp, 'SLG': slg, 'OPS': ops, 'ISO': iso,
                'TB': tb, 'XBH': xbh, 'runs_created': runs_created,
                'WAR': war, 'team_wins': team_wins,
                'is_allstar': is_allstar, 'made_playoffs': made_playoffs,
                'age': age, 'prime_age': prime_age,
                'hit_40hr': 1 if hr >= 40 else 0,
                'hit_100rbi': 1 if rbi >= 100 else 0,
                'hit_300ba': 1 if ba >= 0.300 else 0,
                'milestones_hit': (1 if hr >= 40 else 0) + (1 if rbi >= 100 else 0) + (1 if ba >= 0.300 else 0),
                # Pitching stats
                'is_two_way': 1 if is_two_way else 0,
                'p_W': p_W, 'p_SO': p_SO, 'IP': IP,
                'p_ERA_inv': p_ERA_inv, 'p_WHIP_inv': p_WHIP_inv, 'p_K9': p_K9,
                'p_WAR_est': p_WAR, 'total_WAR': total_WAR
            })

            player_features = np.array([[player_stats[col] for col in feature_cols]])

            # Score against top 3 candidates
            st.markdown(f"### Results: {selected_year} {selected_league} MVP Race")

            # Calculate H2H win probability against each top 3 candidate
            h2h_results = []
            for _, candidate in year_candidates.iterrows():
                candidate_features = candidate[feature_cols].values.reshape(1, -1)
                diff = player_features - candidate_features
                win_prob = model.predict_proba(diff)[0, 1] * 100
                h2h_results.append({
                    'player': candidate['player_name'],
                    'win_prob': win_prob,
                    'war': candidate['WAR'],
                    'ops': candidate['OPS'],
                    'is_winner': candidate['is_winner']
                })

            # Determine ranking: count how many of top 3 you beat (>50%)
            wins = sum(1 for r in h2h_results if r['win_prob'] > 50)

            # Display result
            if wins == 3:
                st.success(f"**1st Place** - You beat all three top vote-getters!")
            elif wins == 2:
                st.info(f"**2nd Place** - You beat 2 of 3 top vote-getters.")
            elif wins == 1:
                st.warning(f"**3rd Place** - You beat 1 of 3 top vote-getters.")
            else:
                st.error(f"**Not in top 3 MVP candidates** - You lost to all three top vote-getters.")

            # Show H2H breakdown
            st.markdown("### Head-to-Head Breakdown")
            ranking_data = []
            for r in h2h_results:
                result = "Win" if r['win_prob'] > 50 else "Loss"
                ranking_data.append({
                    'Actual Finish': '1st' if r['is_winner'] else ('2nd' if r == h2h_results[1] else '3rd'),
                    'Player': r['player'],
                    'Your H2H': f"{r['win_prob']:.1f}%",
                    'Result': result,
                    'Their WAR': round(r['war'], 1),
                    'Their OPS': round(r['ops'], 3)
                })

            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**WGU C964 Capstone**")
st.sidebar.markdown("MLB MVP Predictor v6")
st.sidebar.markdown(f"Accuracy: {100*predictions['correct'].sum()/len(predictions):.0f}%")
