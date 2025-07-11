import streamlit as st
import sqlite3
import pandas as pd
import openai
from openai import OpenAI
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import math


with open("theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["openai_api_key"])

DB_PATH = "bot_data_archetypes.db"
MODEL = "gpt-4"

def execute_query(sql):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return col_names, rows
    except Exception as e:
        return [], [[f"Error: {e}"]]

TIER_COLORS = {
    "Bronze": "#8B5A2B",
    "Silver": "#C0C0C0",
    "Gold": "#FFD700",
    "Platinum": "#9D4EDD",
}

TIER_ORDER = ["None", "Bronze", "Silver", "Gold", "Platinum"]
TIER_INDEX = {tier: i for i, tier in enumerate(TIER_ORDER)}

TIER_LABELS_WITH_EMOJIS = {
    "None": "⚫ None",
    "Bronze": "🟤 Bronze",
    "Silver": "⚪ Silver",
    "Gold": "🟡 Gold",
    "Platinum": "🟣 Platinum"
}

@st.cache_data
def load_data():
    conn = sqlite3.connect("bot_data_archetypes.db")
    query = """
    SELECT
        p.name AS position,
        a.name AS archetype,
        ab.name AS ability,
        t.tier,
        t.stat_1_name,
        t.stat_1_value,
        t.stat_2_name,
        t.stat_2_value,
        t.sp_cost
    FROM ability_tiers t
    JOIN abilities ab ON t.ability_id = ab.id
    JOIN archetypes a ON t.archetype_id = a.id
    JOIN positions p ON a.position_id = p.id
    ORDER BY a.name, ab.name, t.tier;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def blend_color(percent):
    # percent = 0.0 (red) → 1.0 (green)
    r = int(255 * (1 - percent))
    g = int(255 * percent)
    b = 0
    return f"rgb({r},{g},{b})"



# ───────────── AI Assistant ─────────────

df = load_data()

def calculate_efficiency_scores(df_data, blend_weight=0.5):
    """Calculate efficiency scores for all upgrade paths"""
    tier_order = ["None", "Bronze", "Silver", "Gold", "Platinum"]
    rows = []
    
    for (pos, arch, ab), group in df_data.groupby(["position", "archetype", "ability"]):
        tiers = {row["tier"]: row for _, row in group.iterrows()}
        
        # Calculate all possible upgrade paths
        for i in range(len(tier_order)):
            for j in range(i + 1, len(tier_order)):
                start_tier = tier_order[i]
                end_tier = tier_order[j]
                
                if start_tier in tiers and end_tier in tiers:
                    start_row = tiers[start_tier]
                    end_row = tiers[end_tier]
                    
                    # Calculate SP cost
                    sp_sum = sum(
                        tiers[t]["sp_cost"]
                        for t in tier_order[i + 1:j + 1]
                        if t in tiers and pd.notnull(tiers[t]["sp_cost"])
                    )
                    
                    if sp_sum > 0:
                        # Calculate attribute changes
                        start_stat_1 = start_row["stat_1_value"] if pd.notnull(start_row["stat_1_value"]) else 0
                        start_stat_2 = start_row["stat_2_value"] if pd.notnull(start_row["stat_2_value"]) else 0
                        end_stat_1 = end_row["stat_1_value"] if pd.notnull(end_row["stat_1_value"]) else 0
                        end_stat_2 = end_row["stat_2_value"] if pd.notnull(end_row["stat_2_value"]) else 0
                        
                        start_max_stat = max(start_stat_1, start_stat_2)
                        end_max_stat = max(end_stat_1, end_stat_2)
                        
                        attribute_increase = (end_stat_1 + end_stat_2) - (start_stat_1 + start_stat_2)
                        
                        # Apply difficulty multiplier
                        difficulty_multiplier = max(1.0, math.exp((end_max_stat - 70) / 20))
                        weighted_attr_increase = attribute_increase * difficulty_multiplier
                        
                        # Calculate efficiency score
                        sp_weight = blend_weight
                        attr_weight = 1.0 - blend_weight
                        
                        efficiency_score = 100 - (
                            sp_weight * sp_sum +
                            attr_weight * weighted_attr_increase
                        )
                        
                        rows.append({
                            "position": pos,
                            "archetype": arch,
                            "ability": ab,
                            "tier_increase": f"{start_tier} → {end_tier}",
                            "sp_increase": sp_sum,
                            "attribute_increase": attribute_increase,
                            "efficiency_score": round(efficiency_score, 1),
                        })
    
    return pd.DataFrame(rows)

def ask_upgrade_assistant(user_question):
    schema = """
Tables:
- abilities(id, name)
- archetypes(id, name, position_id)
- positions(id, name)
- ability_tiers(id, ability_id, archetype_id, tier, stat_1_name, stat_1_value, stat_2_name, stat_2_value, sp_cost)
- ability_descriptions(id, ability_id, tier, description)

Relationships:
- archetypes.position_id → positions.id
- ability_tiers.ability_id → abilities.id
- ability_tiers.archetype_id → archetypes.id
- ability_descriptions.ability_id → abilities.id
"""

# Query the database for relevant data and calculate efficiency scores
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Gather data
        query = """
        SELECT
            p.name AS position,
            a.name AS archetype,
            ab.name AS ability,
            t.tier,
            t.stat_1_name,
            t.stat_1_value,
            t.stat_2_name,
            t.stat_2_value,
            t.sp_cost
        FROM ability_tiers t
        JOIN abilities ab ON t.ability_id = ab.id
        JOIN archetypes a ON t.archetype_id = a.id
        JOIN positions p ON a.position_id = p.id
        ORDER BY p.name, a.name, ab.name, 
                 CASE t.tier 
                     WHEN 'Bronze' THEN 1
                     WHEN 'Silver' THEN 2
                     WHEN 'Gold' THEN 3
                     WHEN 'Platinum' THEN 4
                 END;
        """
        
        df_data = pd.read_sql_query(query, conn)
        
        # Get ability descriptions
        desc_query = """
        SELECT 
            ab.name AS ability,
            ad.tier,
            ad.description
        FROM ability_descriptions ad
        JOIN abilities ab ON ad.ability_id = ab.id
        ORDER BY ab.name, ad.tier;
        """
        
        descriptions_df = pd.read_sql_query(desc_query, conn)
        conn.close()
        
        # Filter data based on user question keywords
        filtered_data = df_data.copy()
        question_lower = user_question.lower()
        
        keyword_map = {
            # Positions
            'quarterback': 'QB',
            'qb': 'QB',
            'halfback': 'HB', 
            'hb': 'HB',
            'running back': 'HB',
            'runningback': 'HB',
            'fullback': 'FB',
            'fb': 'FB',
            'wide receiver': 'WR',
            'wide-receiver': 'WR',
            'receiver': 'WR',
            'wr': 'WR',
            'tight end': 'TE',
            'tight-end': 'TE',
            'te': 'TE',
            'offensive line': 'OL',
            'offensive-line': 'OL',
            'o-line': 'OL',
            'oline': 'OL',
            'ol': 'OL',
            'center': 'OL',
            'guard': 'OL',
            'tackle': 'OL',
            'defensive line': 'DL',
            'defensive-line': 'DL',
            'd-line': 'DL',
            'dline': 'DL',
            'dl': 'DL',
            'linebacker': 'LB',
            'lb': 'LB',
            'cornerback': 'CB',
            'corner': 'CB',
            'cb': 'CB',
            'safety': 'S',
            'safeties': 'S',
            'kicker': 'K',
            'k': 'K',
            'punter': 'P',
            'p': 'P',
            
            # QB
            'backfield creator': 'Backfield Creator',
            'backfield-creator': 'Backfield Creator',
            'dual threat': 'Dual Threat',
            'dual-threat': 'Dual Threat',
            'pocket passer': 'Pocket Passer',
            'pocket-passer': 'Pocket Passer',
            'pure runner': 'Pure Runner',
            'pure-runner': 'Pure Runner',

            # HB
            'backfield threat': 'Backfield Threat',
            'backfield-threat': 'Backfield Threat',
            'contact seeker': 'Contact Seeker',
            'contact-seeker': 'Contact Seeker',
            'east west playmaker': 'East/West Playmaker',
            'east-west playmaker': 'East/West Playmaker',
            'elusive bruiser': 'Elusive Bruiser',
            'elusive-bruiser': 'Elusive Bruiser',
            'north south receiver': 'North/South Receiver',
            'north-south receiver': 'North/South Receiver',
            'north south blocker': 'North/South Blocker',
            'north-south blocker': 'North/South Blocker',

            # FB
            'blocking': 'Blocking',
            'utility': 'Utility',

            # WR
            'contested specialist': 'Contested Specialist',
            'contested-specialist': 'Contested Specialist',
            'elusive route runner': 'Elusive Route Runner',
            'elusive-route-runner': 'Elusive Route Runner',
            'gadget': 'Gadget',
            'gritty possession wr': 'Gritty Possession WR',
            'gritty-possession-wr': 'Gritty Possession WR',
            'physical route runner wr': 'Physical Route Runner WR',
            'physical-route-runner-wr': 'Physical Route Runner WR',
            'route artist': 'Route Artist',
            'route-artist': 'Route Artist',
            'speedster': 'Speedster',

            # TE
            'gritty possession te': 'Gritty Possession TE',
            'gritty-possession-te': 'Gritty Possession TE',
            'physical route runner te': 'Physical Route Runner TE',
            'physical-route-runner-te': 'Physical Route Runner TE',
            'possession': 'Possession',
            'pure blocker': 'Pure Blocker',
            'pure-blocker': 'Pure Blocker',
            'vertical threat': 'Vertical Threat',
            'vertical-threat': 'Vertical Threat',

            # OL
            'agile': 'Agile',
            'pass protector': 'Pass Protector',
            'pass-protector': 'Pass Protector',
            'raw strength': 'Raw Strength',
            'raw-strength': 'Raw Strength',
            'well rounded': 'Well Rounded',
            'well-rounded': 'Well Rounded',

            # DL
            'edge setter': 'Edge Setter',
            'edge-setter': 'Edge Setter',
            'gap specialist': 'Gap Specialist',
            'gap-specialist': 'Gap Specialist',
            'physical freak': 'Physical Freak',
            'physical-freak': 'Physical Freak',
            'power rusher': 'Power Rusher',
            'power-rusher': 'Power Rusher',
            'speed rusher': 'Speed Rusher',
            'speed-rusher': 'Speed Rusher',

            # LB
            'lurker': 'Lurker',
            'signal caller': 'Signal Caller',
            'signal-caller': 'Signal Caller',
            'thumper': 'Thumper',

            # CB
            'boundary': 'Boundary',
            'bump and run': 'Bump and Run',
            'bump-and-run': 'Bump and Run',
            'field': 'Field',
            'zone': 'Zone',

            # S
            'box specialist': 'Box Specialist',
            'box-specialist': 'Box Specialist',
            'coverage specialist': 'Coverage Specialist',
            'coverage-specialist': 'Coverage Specialist',
            'hybrid': 'Hybrid',

            # K/P
            'accurate': 'Accurate',
            'power': 'Power',
        }


        matched = [value for key, value in keyword_map.items() if key in question_lower]
        if matched:
            # Use the first match (could also combine multiple matches if needed)
            filtered_data = df_data[df_data['archetype'].str.contains(matched[0], case=False, na=False)]
        else:
            filtered_data = df_data  # fallback to full dataset
        
        # Use filtered data for efficiency calculations
        df_for_efficiency = filtered_data if not filtered_data.empty else df_data
        
        # Calculate efficiency scores for relevant data
        efficiency_data = calculate_efficiency_scores(df_for_efficiency)
        
        # Convert the data to a readable format for the AI
        data_summary = ""
        
        # Group by position and archetype for better organization
        for (position, archetype), group in df_data.groupby(['position', 'archetype']):
            data_summary += f"\n=== {position} - {archetype} ===\n"
            
            # Group by ability within each archetype
            for ability, ability_group in group.groupby('ability'):
                data_summary += f"\n{ability}:\n"
                
                for _, row in ability_group.iterrows():
                    tier = row['tier']
                    sp_cost = row['sp_cost']
                    stat1 = f"{row['stat_1_name']} +{row['stat_1_value']}" if pd.notnull(row['stat_1_value']) else ""
                    stat2 = f"{row['stat_2_name']} +{row['stat_2_value']}" if pd.notnull(row['stat_2_value']) else ""
                    
                    stats_text = f"{stat1}"
                    if stat2:
                        stats_text += f", {stat2}"
                    
                    # Add description if available
                    description = descriptions_df[
                        (descriptions_df['ability'] == ability) & 
                        (descriptions_df['tier'] == tier)
                    ]
                    desc_text = ""
                    if not description.empty:
                        desc_text = f" | {description.iloc[0]['description']}"
                    
                    data_summary += f"  {tier}: {sp_cost} SP → {stats_text}{desc_text}\n"
        
        # Add efficiency rankings
        efficiency_summary = "\n\n=== EFFICIENCY RANKINGS ===\n"
        efficiency_summary += "Top upgrade paths by efficiency score (Higher = Better Value):\n\n"
        
        # Get top 20 most efficient upgrades
        top_upgrades = efficiency_data.nlargest(20, 'efficiency_score')
        for _, row in top_upgrades.iterrows():
            efficiency_summary += f"{row['archetype']} - {row['ability']} ({row['tier_increase']}): "
            efficiency_summary += f"Score {row['efficiency_score']:.1f} | {row['sp_increase']} SP | +{row['attribute_increase']} total stats\n"
        
        data_summary += efficiency_summary
        
    except Exception as e:
        data_summary = f"Error retrieving data: {e}"

    prompt = f"""You are an expert college football upgrade strategist built into a EA Sports College Football 26 upgrade planner.

You have access to a complete database of player upgrade paths, efficiency scores, AND detailed ability descriptions that help identify the best value upgrades and explain what each ability does.

Database schema:
{schema}

Current upgrade data (including ability descriptions):
{data_summary}

CORE INSTRUCTIONS:
1. Always provide specific, actionable advice using real numbers from the data
2. Calculate exact SP costs for any upgrade paths mentioned
3. Include ability descriptions when explaining what abilities do or recommending upgrades
4. Compare multiple options when relevant to help users make informed decisions
5. For efficiency analysis, ALWAYS refer users to the Upgrade Efficiency Model for detailed value comparisons
6. Prioritize clarity and practicality over technical jargon

IMPORTANT DISTINCTION:
- Abilities DO NOT grant attribute increases.
- Instead, each tier of an ability is UNLOCKED based on meeting specific attribute thresholds (e.g. ACC 92 and COD 90 for Silver Shifty).
- The stats listed next to a tier are REQUIREMENTS, not gains.
- Do NOT say an ability "gives" or "boosts" an attribute.
- Instead, say "requires" or "is unlocked by reaching [stat values]."

ABILITY DESCRIPTIONS USAGE:
- When explaining what an ability does, use the descriptions from the database
- When discussing a tier's requirements, explain that the listed attributes are REQUIRED to unlock it
- Example: "Silver Shifty requires 92 ACC and 90 COD - it does not boost those stats."
- If someone asks "what does [ability] do?", provide the descriptions for all available tiers

EFFICIENCY SCORE GUIDANCE:
- Efficiency scores help identify the best "bang for your buck" upgrades
- Higher scores = better value for SP investment
- Scores consider both SP cost and attribute increases, weighted by difficulty
- When recommending upgrades, prioritize high-efficiency options when possible
- Mention efficiency scores for upgrades you recommend (e.g., "Score 85.2")

RESPONSE GUIDELINES:
- Lead with the most important information first
- Use bullet points for multiple recommendations
- Always include total SP costs when discussing upgrade paths
- Mention stat gains in format like "Speed +6" or "Acceleration +4"
- Include efficiency scores when discussing specific upgrades
- Include ability descriptions when explaining what abilities do
- When comparing options, clearly state which is more cost-effective and why
- NEVER say that an ability increases or improves a stat like SPD, ACC, or COD
- ALWAYS say "requires" or "needs at least" when referencing attribute thresholds
- Be clear: upgrades unlock the ability at a tier once the player has the required stats and pays the SP cost
- Always say Skill Points instead of SP
- Always say the Attribute Requirements before before the Skill Point Requirements


CALCULATION RULES:
- Players start at "None" tier by default
- To reach any tier, you must pay for ALL previous tiers
- None → Bronze = Bronze SP cost
- None → Silver = Bronze SP + Silver SP  
- None → Gold = Bronze SP + Silver SP + Gold SP
- None → Platinum = Bronze SP + Silver SP + Gold SP + Platinum SP
- Bronze → Silver = Silver SP cost only
- Silver → Gold = Gold SP cost only
- etc.

COMMON QUESTION TYPES & HOW TO HANDLE:
- "What should I upgrade with X SP?" → Show options within budget, refer to efficiency model for optimization
- "How much to max out [ability]?" → Calculate total cost and explain what each tier does
- "Which upgrades are most efficient?" → Refer to Upgrade Efficiency Model for detailed analysis
- "Best upgrades for [position/archetype]?" → Show options, refer to efficiency model for value comparison
- "What does [ability] do?" → Provide descriptions for all tiers of that ability
- "Explain [ability] tiers" → Detail what each tier does and costs

TONE & STYLE:
- Speak as a knowledgeable coach making strategic decisions
- Be confident but explain your reasoning
- Acknowledge when efficiency analysis would be helpful and direct to the model
- Reference ability descriptions naturally in recommendations

EXAMPLE QUALITY RESPONSES:
Bad: "Silver upgrades cost varying amounts"
Good: "For Pocket Passers, Silver Quick Release costs 8 SP and gives you 'Moderately improved ability to quickly release the ball', while Silver Pocket Presence costs 10 SP for 'Enhanced awareness and composure in the pocket'.For detailed efficiency comparisons of all upgrade paths, check the Upgrade Efficiency Model."

Now answer this question with specific numbers, ability descriptions, clear recommendations, and strategic reasoning:

"{user_question}"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Assistant failed to answer: {e}"



# ───────────── Streamlit App ─────────────

st.set_page_config("DynastyLab - AI Assistant & Analytics for CFB Dynasties", layout="wide")
# st.title("DynastyLab")

st.markdown('<div class="gradient-hero">DynastyLab</div>', unsafe_allow_html=True)

# ───────────── UI ─────────────

st.markdown("""

Master your College Football Dynasty Mode with AI-powered upgrade optimization, data-driven insights, and strategic planning tools.""")

with st.expander("Explore Features & Get Started"):
    st.markdown("""
    <div class="gradient-text">What You Can Do</div>

    <span class="dropdown-highlight">Ask the AI Assistant</span> - Get instant, expert advice on upgrade strategies. Ask questions like:
    - "What's the most efficient way to spend 25 SP on a Pocket Passer?"
    - "Which Silver tier upgrades give the best value?"
    - "How much SP does it cost to max out Quick Jump?"

    <span class="dropdown-highlight">Plan Your Upgrades</span> - Use the visual upgrade planner to:
    - See exactly how much SP each upgrade path costs
    - Compare current vs. target ability tier
    - Track your total SP budget in real-time
    - Watch the tier bars light up as you plan upgrades

    <span class="dropdown-highlight">Find the Best Value</span> - The Efficiency Model reveals:
    - Which upgrades give you the most "bang for your buck"
    - How different tier transitions compare in cost-effectiveness
    - Smart filtering by position, archetype, and ability type
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("""
    <div class="gradient-text">How to Get Started</div>

    1. **Ask the AI Assistant** below for personalized recommendations  
    2. **Use the Upgrade Planner** to visualize and cost out your upgrade paths  
    3. **Explore the Efficiency Model** to discover hidden gems and avoid SP traps

    Ready to dominate the field? Let's optimize those upgrades!
    """, unsafe_allow_html=True)


EXAMPLE_QUESTION = "Which archetypes get Shifty?"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.expander("Build Smarter. Ask the AI.", expanded=True):

    # Form to prevent reloads
    with st.form(key='assistant_form'):
        user_input = st.text_input(
            "Ask anything about archetypes, abilities, or upgrade strategies.",
            placeholder=EXAMPLE_QUESTION,
            key="user_question"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Get Your Answer")

    # Only process when form is submitted
    if submit_button and user_input:
        with st.spinner("Thinking..."):
            assistant_reply = ask_upgrade_assistant(user_input)
            # Add to chat history
            st.session_state.chat_history.append({
                'question': user_input,
                'answer': assistant_reply
            })

    # Display chat history
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # Auto-expand only the most recent message
        is_latest = i == 0
        with st.expander(f"Q: {chat['question']}", expanded=is_latest):
            st.markdown(f"<div class='chat-response'>{chat['answer']}</div>", unsafe_allow_html=True)



    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


st.markdown("<br>", unsafe_allow_html=True)

view_mode = st.radio(
    "Select Tool",
    ["Upgrade Planner", "Upgrade Efficiency Model", "Tier Progression Visualization"],
    horizontal=True
)

# ───────────── Upgrade Planner ─────────────

if view_mode == "Upgrade Planner":
    st.markdown("""
    <div style="
        background: linear-gradient(to right, #222, #0d0d0d);
        border-left: 5px solid #bdff00;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        color: #f8f8f8;
        margin-bottom: 1rem;
    ">
    Use the planner to simulate SP costs, choose upgrades, and manage tier progress.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    

    col1, col2, col3 = st.columns([5, 3, 1.5])

    # ───────────── COLUMN 1: Dropdowns and Bars ─────────────
    with col1:
        st.markdown("### Upgrade Planner")

        col1_sp,col_pos, col_arch = st.columns(3)
        with col1_sp:
            max_sp = st.number_input("Player SP", value=20, min_value=1, step=1)
        with col_pos:
            position = st.selectbox("Position", sorted(df["position"].unique()))
        with col_arch:
            archetype = st.selectbox("Archetype", sorted(df[df["position"] == position]["archetype"].unique()))

        st.markdown("### Abilities")
        
        filtered_df = df[(df["position"] == position) & (df["archetype"] == archetype)]

        ability_dict = {}
        for ability in filtered_df["ability"].unique():
            tier_rows = filtered_df[filtered_df["ability"] == ability]
            ability_dict[ability] = {
                row["tier"]: {
                    "sp_cost": row["sp_cost"],
                    "stat_1": (
                        f"{row['stat_1_name']} <span class='stat-number'>{int(row['stat_1_value'])}</span>"
                        if pd.notnull(row["stat_1_name"]) and pd.notnull(row["stat_1_value"]) else ""
                    ),
                    "stat_2": (
                        f"{row['stat_2_name']} <span class='stat-number'>{int(row['stat_2_value'])}</span>"
                        if pd.notnull(row["stat_2_name"]) and pd.notnull(row["stat_2_value"]) else None
                    ),
                }
                for _, row in tier_rows.iterrows()
            }

        bar_cols = st.columns(len(ability_dict))
        for i, (ability, tiers) in enumerate(ability_dict.items()):
            current = st.session_state.get(f"{ability}_current", "None")
            upgrade = st.session_state.get(f"{ability}_upgrade", "None")
            cur_idx = TIER_INDEX.get(current, 0)
            upg_idx = TIER_INDEX.get(upgrade, 0)

            with bar_cols[i]:
                st.markdown(f"<p class='ability-header'>{ability}</h3>", unsafe_allow_html=True)

                for tier in reversed(TIER_ORDER[1:]):
                    tier_idx = TIER_INDEX[tier]
                    color = "#171717"
                    glow = ""
                    

                    if current == "None":
                        if tier_idx <= upg_idx:
                            color = TIER_COLORS.get(tier, "#171717")
                            glow = (
                                "box-shadow: 0 0 6px 3px #a8e600, inset 0 0 15px 0px #a8e600;"
                            )
                    else:
                        if tier_idx == cur_idx:
                            color = TIER_COLORS.get(tier, "#171717")  # Full Color
                        elif tier_idx < cur_idx:
                            base = TIER_COLORS.get(tier, "#171717")
                            color = base + "33"  # Opacity
                        elif cur_idx < tier_idx <= upg_idx:
                            color = TIER_COLORS.get(tier, "#171717")
                            glow = (
                                "box-shadow: 0 0 15px 3px #a8e600, inset 0 0 15px 0px #a8e600;"
                            )


                    style = f"""
                        background-color: {color};
                        width: 100px;
                        height: 120px;
                        border-radius: 10px;
                        margin: 5px auto;
                        border: 2px solid rgba(255, 255, 255, 0.2);  /* Inner stroke */
                        box-shadow:
                            inset 0 0 15px 4px rgba(0, 0, 0, 0.8),  /* Inner glow */
                            0 0 20px 3px #a8e600;  /* Outer glow for upgrade */
                    """ if glow else f"""
                        background-color: {color};
                        width: 100px;
                        height: 120px;
                        border-radius: 10px;
                        margin: 5px auto;
                        border: 2px solid rgba(255, 255, 255, 0.2);  /* Inner stroke */
                        box-shadow: inset 0 0 15px 4px rgba(0, 0, 0, 0.7);
                    """
                    st.markdown(f"<div style='{style}'></div>", unsafe_allow_html=True)
                    

    # ───────────── COLUMN 2: Current & Target Tier ─────────────

    with col2:
        header_col1, header_col2, header_col3 = st.columns([1.4, 1.4, 1])
        with header_col1:
            st.markdown("""
            <div class="ability-column-header">Current Tier</div>
            """, unsafe_allow_html=True)

        with header_col2:
            st.markdown("""
            <div class="ability-column-header">Target Tier</div>
            """, unsafe_allow_html=True)
        with header_col3:
            st.markdown("")

        for ability in ability_dict:
            tiers = ability_dict[ability]

            st.markdown(f"<p style='margin-bottom: 1rem; '>{ability}</p>", unsafe_allow_html=True)
            col_current, col_upgrade, col_cost = st.columns([1.4, 1.4, 1])

            with col_current:
                current_tier = st.selectbox(
                    key=f"{ability}_current",
                    label_visibility="collapsed",
                    label="",
                    options=TIER_ORDER,
                    format_func=lambda x: TIER_LABELS_WITH_EMOJIS.get(x, x)
                )
                
                stat1 = tiers.get(current_tier, {}).get("stat_1", " ")
                stat2 = tiers.get(current_tier, {}).get("stat_2")
                st.markdown(stat1, unsafe_allow_html=True)
                if stat2:
                    st.markdown(stat2, unsafe_allow_html=True)

            with col_upgrade:
                upgrade_tier = st.selectbox(
                    key=f"{ability}_upgrade",
                    label_visibility="collapsed",
                    label="",
                    options=TIER_ORDER,
                    format_func=lambda x: TIER_LABELS_WITH_EMOJIS.get(x, x)
                )
                stat1 = tiers.get(upgrade_tier, {}).get("stat_1", " ")
                stat2 = tiers.get(upgrade_tier, {}).get("stat_2")
                st.markdown(stat1, unsafe_allow_html=True)
                if stat2:
                    st.markdown(stat2, unsafe_allow_html=True)

            with col_cost:
                cur_idx = TIER_INDEX.get(current_tier, 0)
                upg_idx = TIER_INDEX.get(upgrade_tier, 0)
                cost = sum(
                    tiers.get(TIER_ORDER[i], {}).get("sp_cost", 0)
                    for i in range(cur_idx + 1, upg_idx + 1)
                ) if upgrade_tier != "None" else 0
                st.markdown(f"<div style='padding-top: 6px; font-size: 20px; font-weight: bold;'>+{cost} SP</div>", unsafe_allow_html=True)

            st.markdown("<hr style='border: 1px solid #333; margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)



    # ───────────── COLUMN 3: SP Summary ─────────────
    with col3:
        st.markdown("""
        <div class="ability-column-header">SP Summary</div>
        """, unsafe_allow_html=True)
        total_cost = 0

        for ability in ability_dict:
            tiers = ability_dict[ability]
            current = st.session_state.get(f"{ability}_current", "None")
            upgrade = st.session_state.get(f"{ability}_upgrade", "None")

            if upgrade == "None":
                cost = 0
            else:
                cur_idx = TIER_INDEX.get(current, 0)
                upg_idx = TIER_INDEX.get(upgrade, 0)
                cost = sum(
                    tiers.get(TIER_ORDER[i], {}).get("sp_cost", 0)
                    for i in range(cur_idx + 1, upg_idx + 1)
                )
            st.caption(f"{ability}: +{cost} SP")
            total_cost += cost

        st.markdown("""
        <div class="totalsp-header">Total SP Cost</div>
        """, unsafe_allow_html=True)
        st.markdown(f"# {total_cost} / {max_sp}")

        if total_cost > max_sp:
            st.error("Player SP Budget Exceeded!")   

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    

# ───────────── Upgrade Efficiency Model ─────────────

elif view_mode == "Upgrade Efficiency Model":
    st.markdown("""
    <div style="
        background: linear-gradient(to right, #222, #0d0d0d);
        border-left: 5px solid #bdff00;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        color: #f8f8f8;
        margin-bottom: 1rem;
    ">
    View upgrade value analysis across all tiers with interactive scoring to compare how cost-effective different ability upgrades are.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Upgrade Efficiency Model")
   
   
    with st.expander("How the Efficiency Model Works - See why some upgrades are smarter than others"):
        st.markdown("""
        <span class="dropdown-highlight">The Efficiency Score </span>helps you identify the smartest upgrades by balancing SP cost, attribute increase, and difficulty.
        It shows which upgrades provide the best overall value, whether through high stat boosts, low cost, or a strong balance between the two.
        """, unsafe_allow_html=True)        

        # Section 1
        st.markdown('<div class="gradient-text">Analysis Options</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul>
        <li><span class="dropdown-highlight">Efficiency vs SP Cost:</span> Best when trying to stretch your SP budget</li>
        <li><span class="dropdown-highlight">Efficiency vs Attribute Increase:</span> Best when you want high efficiency with minimal attribute increases</li>
        <li><span class="dropdown-highlight">SP Cost vs Attribute Increase:</span> Best for raw comparisons without scoring influence</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("---")


        # Section 2
        st.markdown('<div class="gradient-text">Higher Scores = Better Upgrade Value</div>', unsafe_allow_html=True)
        st.markdown("""
        That could mean:
        - A **big stat gain** for a reasonable cost, or
        - A **small cost** for a solid gain, especially at higher attribute levels
        """)
        st.markdown("""
        The model helps you **spot hidden value** in upgrades that aren't obvious at first glance. It's based on two main factors:
        """)
        st.markdown("---")
        
        # Section 3 - Difficulty Modifier
        st.markdown("""
        The model helps you **spot hidden value** in upgrades that aren't obvious at first glance. It's based on two main factors:
        """)
        st.markdown('<div class="gradient-text">1. Attribute Difficulty Adjustment</div>', unsafe_allow_html=True)
        st.markdown("Not all attribute increases are equal.")
        st.markdown("""
        - Upgrading an attribute from 90 to 95 is much harder (and more valuable) than upgrading from 70 to 75. 
        - To reflect that, we apply a **difficulty adjustment** based on the final attribute value.
        - The higher the final attribute, the more weight the model assigns to the stat increase.
        - This prevents elite upgrades from being scored the same as easier ones.
        """)
        st.markdown("*Formula:* `Difficulty Adjustment = max(1.0, math.exp((Final Attribute - 70) / 20))`") 
        st.markdown("*Weighted Attribute Increase:* `Attribute Increase × Difficulty Adjustment`")
        st.markdown("---")

        # Section 4 - Weighted Score System
        st.markdown('<div class="gradient-text">2. Calculating Upgrade Efficiency</div>', unsafe_allow_html=True)
        st.markdown("""
        Each upgrade starts with a score of **100**, and then we subtract penalties based on:
        - The SP Cost of the upgrade
        - The weighted size of the attribute increase
        You control how much each of those factors matters using the **Weight Blending slider**:
        - **0.0**: Focus only on attribute increases
        - **0.5**: A balanced tradeoff between both
        - **1.0**: Focus only on SP Cost  
        """)
        st.markdown("*Weight Mapping:* `SP Weight = Blend Value`, `Attribute Weight = 1 - Blend Value`")
        st.markdown("*Formula:* `Efficiency Score = 100 - (SP Weight × SP Cost + Attribute Weight × Weighted Attribute Increase)`")
        

    
     
    # Add graph view toggle
    eff_chart_mode = st.radio(
        "Select Visualization",
        ["Efficiency vs SP Cost", "Efficiency vs Attribute Increase", "SP Cost vs Attribute Increase (Raw Value)"],
        horizontal=True
    )

    TIER_TRANSITION_STROKES = {
        "Bronze → Silver": {"color": "#C0C0C0", "width": 2},
        "Bronze → Gold": {"color": "#FF9900", "width": 2},
        "Bronze → Platinum": {"color": "#DA7082", "width": 2},
        "Silver → Gold": {"color": "#F3CD22", "width": 2},
        "Silver → Platinum": {"color": "#B29FFF", "width": 2},
        "Gold → Platinum": {"color": "#8A2BE2", "width": 2},
    }
    # Dropdown Filters
    col1, col2 = st.columns(2)
    with col1:
        pos_filter = st.multiselect("Filter by Position", sorted(df["position"].unique()))

    with col2:
        available_archs = (
            df[df["position"].isin(pos_filter)]["archetype"].unique().tolist()
            if pos_filter
            else df["archetype"].unique().tolist()
        )
        arch_filter = st.multiselect("Filter by Archetype", sorted(available_archs))


    col3, col4 = st.columns(2)
    with col3:
        available_abilities = (
            df[df["archetype"].isin(arch_filter)]["ability"].unique().tolist()
            if arch_filter
            else df["ability"].unique().tolist()
        )
        ability_filter = st.multiselect("Filter by Ability", sorted(available_abilities))

    with col4:
        tier_jump = st.multiselect(
            "Filter by Tier Transition",
            [
                "Bronze → Silver",
                "Bronze → Gold",
                "Bronze → Platinum",
                "Silver → Gold",
                "Silver → Platinum",
                "Gold → Platinum",
            ],
        )

    tier_order = ["None", "Bronze", "Silver", "Gold", "Platinum"]

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        st.markdown("""
        <div style='display: flex; justify-content: center;'>
        <div class='russo-label'>WEIGHT BLENDING</div>
        </div>
        """, unsafe_allow_html=True)

        # Calculate Percentages
        if 'blend_weight' not in st.session_state:
            st.session_state.blend_weight = 0.5
        
        attr_pct = 1 - st.session_state.blend_weight
        sp_pct = st.session_state.blend_weight
        
        attr_color = blend_color(attr_pct)
        sp_color = blend_color(sp_pct)
        
        # Percentages
        st.markdown(
            f"""
            <div style='text-align: center; margin-bottom: 0px;'>
                <span style='color:{attr_color}; font-weight:bold;'>{int(attr_pct * 100)}% Attribute Increase</span> · 
                <span style='color:{sp_color}; font-weight:bold;'>{int(sp_pct * 100)}% SP Cost</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Slider
        blend_weight = st.slider(
            "",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="0 = Prioritize Attribute Increase · 1 = Prioritize SP Cost",
            key="blend_weight"
        )

    sp_weight = blend_weight
    attr_weight = 1.0 - blend_weight

    eff_df = df.copy()
    if pos_filter:
        eff_df = eff_df[eff_df["position"].isin(pos_filter)]
    if arch_filter:
        eff_df = eff_df[eff_df["archetype"].isin(arch_filter)]
    if ability_filter:
        eff_df = eff_df[eff_df["ability"].isin(ability_filter)]



    rows = []

    for (pos, arch, ab), group in eff_df.groupby(["position", "archetype", "ability"]):
        tiers = {row["tier"]: row for _, row in group.iterrows()}

        transitions = []
        if not tier_jump:
            for i in range(len(tier_order)):
                for j in range(i + 1, len(tier_order)):
                    transitions.append((tier_order[i], tier_order[j]))
        else:
            for t in tier_jump:
                start_tier, end_tier = [x.strip() for x in t.split("→")]
                transitions.append((start_tier, end_tier))

        for start_tier, end_tier in transitions:
            if start_tier in tiers and end_tier in tiers:
                start_row = tiers[start_tier]
                end_row = tiers[end_tier]

                start_idx = tier_order.index(start_tier)
                end_idx = tier_order.index(end_tier)

                sp_sum = sum(
                    tiers[t]["sp_cost"]
                    for t in tier_order[start_idx + 1:end_idx + 1]
                    if t in tiers and pd.notnull(tiers[t]["sp_cost"])
                )

                stat_changes = []
                if pd.notnull(start_row["stat_1_name"]) and pd.notnull(start_row["stat_1_value"]) and pd.notnull(end_row["stat_1_value"]):
                    stat_changes.append(f"{start_row['stat_1_name']}: {int(start_row['stat_1_value'])} → {int(end_row['stat_1_value'])}")

                if pd.notnull(start_row["stat_2_name"]) and pd.notnull(start_row["stat_2_value"]) and pd.notnull(end_row["stat_2_value"]):
                    stat_changes.append(f"{start_row['stat_2_name']}: {int(start_row['stat_2_value'])} → {int(end_row['stat_2_value'])}")


                primary_stat = "<br>".join(stat_changes)
                
                
                # Compute stat changes
                start_stat_1 = start_row["stat_1_value"] if pd.notnull(start_row["stat_1_value"]) else 0
                start_stat_2 = start_row["stat_2_value"] if pd.notnull(start_row["stat_2_value"]) else 0
                end_stat_1 = end_row["stat_1_value"] if pd.notnull(end_row["stat_1_value"]) else 0
                end_stat_2 = end_row["stat_2_value"] if pd.notnull(end_row["stat_2_value"]) else 0

                start_max_stat = max(start_stat_1, start_stat_2)
                end_max_stat = max(end_stat_1, end_stat_2)

                attribute_increase = (end_stat_1 + end_stat_2) - (start_stat_1 + start_stat_2)
                efficiency_ratio = attribute_increase / sp_sum if sp_sum > 0 else 0

                difficulty_multiplier = max(1.0, math.exp((end_max_stat - 70) / 20))
                weighted_attr_increase = attribute_increase * difficulty_multiplier
                
                # Clamp the ratio penalty to avoid huge negatives
                if efficiency_ratio <= 0:
                    ratio_penalty = 50
                else:
                    ratio_penalty = 1 / efficiency_ratio

                if sp_sum > 0:
                    efficiency_score = round(
                        100 - (
                            sp_weight * sp_sum +
                            attr_weight * weighted_attr_increase
                        ), 2
                    )
                    rows.append(
                        {
                            "position": pos,
                            "archetype": arch,
                            "ability": ab,
                            "tier_increase": f"{start_tier} → {end_tier}",
                            "sp_increase": sp_sum,
                            "attribute_increase": attribute_increase,
                            "efficiency_score": efficiency_score,
                            "stat_change": primary_stat,
                        }
                    )

    eff_df = pd.DataFrame(rows)

    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    eff_df["sp_jitter"] = eff_df["sp_increase"] + np.random.uniform(-0.2, 0.2, size=len(eff_df))
    eff_df["score_jitter"] = eff_df["efficiency_score"] + np.random.uniform(-0.2, 0.2, size=len(eff_df))
    eff_df["attr_jitter"] = eff_df["attribute_increase"] + np.random.uniform(-0.2, 0.2, size=len(eff_df))

    # Tier stroke color
    eff_df["line_color"] = eff_df["tier_increase"].map(lambda x: TIER_TRANSITION_STROKES.get(x, {"color": "white"})["color"])
    eff_df["line_width"] = eff_df["tier_increase"].map(lambda x: TIER_TRANSITION_STROKES.get(x, {"width": 2})["width"])

    # Archetype color map
    custom_palette = (
        px.colors.qualitative.Alphabet +
        px.colors.qualitative.Dark24 +
        px.colors.qualitative.Light24 +
        px.colors.qualitative.Set3
    )

    ARCHETYPE_COLOR_MAP = {
        arch: custom_palette[i % len(custom_palette)]
        for i, arch in enumerate(eff_df["archetype"].unique())
    }
    eff_df["archetype_color"] = eff_df["archetype"].map(ARCHETYPE_COLOR_MAP)

    # Symbol map for ability
    PLOTLY_SYMBOLS = [
        "circle", "square", "diamond", "cross", "x",
        "triangle-up", "triangle-down", "triangle-left", "triangle-right",
        "star", "hexagon", "pentagon", "star-diamond", "octagon",
    ]
    ABILITY_SYMBOL_MAP = {ab: PLOTLY_SYMBOLS[i % len(PLOTLY_SYMBOLS)] for i, ab in enumerate(eff_df["ability"].unique())}
    eff_df["symbol"] = eff_df["ability"].map(ABILITY_SYMBOL_MAP)

    # ────────────────── Helper function to create plot layers ──────────────────
    def add_plot_layers(fig, x_data, y_data, eff_df):
        # Layer 1: Outer stroke - Tier Transition Color
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                symbol=eff_df["symbol"],
                size=30,
                color=eff_df["line_color"],
                line=dict(color=eff_df["line_color"], width=0),
                opacity=1,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Layer 2: Middle border
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                symbol=eff_df["symbol"],
                size=24,
                color="#0d0d0d",
                line=dict(color="#0d0d0d", width=0),
                opacity=1,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Tier transition legend
        first = True
        for transition, style in TIER_TRANSITION_STROKES.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=16, color=style["color"]),
                name=transition,
                showlegend=True,
                legendgroup="tier",
                legendgrouptitle=dict(
                    text="Tier Transition (Outline)",
                    font=dict(size=16, color="white", family = "Russo One")
                ) if first else None,
            ))
            first = False

        # Layer 3: Foreground symbols - Archetype Color
        first_ability = True
        for (arch, ab), group in eff_df.groupby(["archetype", "ability"]):
            group_x = x_data[group.index]
            group_y = y_data[group.index]
            
            fig.add_trace(go.Scatter(
                x=group_x,
                y=group_y,
                mode="markers",
                name=f"{arch} · {ab}",
                marker=dict(
                    symbol=group["symbol"].iloc[0],
                    size=20,
                    color=group["archetype_color"].iloc[0],
                    line=dict(color="black", width=0),
                    opacity=1,
                ),
                customdata=group[[
                    "position", "tier_increase", "sp_increase", "attribute_increase",
                    "efficiency_score", "archetype", "ability", "stat_change"
                ]],
                hovertemplate=(
                    "<span style='font-size:16px'><b>%{customdata[6]}</b></span><br><br>"
                    "Archetype: <b>%{customdata[5]}</b><br>"
                    "Position: <b>%{customdata[0]}</b><br>"
                    "Tier Increase: <b>%{customdata[1]}</b><br>"
                    "Total SP Cost: <b>%{customdata[2]}</b><br>"
                    "Attribute: <b>%{customdata[7]}</b><br>"
                    "Total Attribute Increase: <b>%{customdata[3]}</b><br><br>"
                    "<span style='color:#e0ff8a'><span style='font-size:18px'><b>Upgrade Efficiency Score: %{customdata[4]}</b><extra></extra>"
                ),
                showlegend=True,
                legendgrouptitle=dict(
                    text="Archetype (Color) · Ability (Symbol)",
                    font=dict(size=16, color="white", family="Russo One")
                ) if first_ability else None,
            ))
            first_ability = False

    # ────────────────── Plot Setup ──────────────────
    if eff_df.empty:
        st.warning("No matching data found for the selected position and tier transition.")
    else:
        if eff_chart_mode == "Efficiency vs SP Cost":
            fig = go.Figure()
            add_plot_layers(fig, eff_df["sp_jitter"], eff_df["score_jitter"], eff_df)
            
            fig.update_layout(
                height=800,
                title={
                    "text": "Upgrade Efficiency vs SP Cost<br><span style='font-size:14px; color:#e0ff8a;'>Best when trying to stretch your SP budget</span>",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": dict(size=18)
                },
                plot_bgcolor="#111",
                paper_bgcolor="rgba(0, 0, 0, 0.0)",
                margin=dict(t=120),
                xaxis=dict(
                    title="<br> Total SP Cost",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                yaxis=dict(
                    title="<br>Upgrade Efficiency Score",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                legend=dict(
                    font=dict(
                        family="Inter, sans-serif",
                        size=12,
                        color="#f8f8f8"
                    )
                )
            )
            
            st.plotly_chart(fig, use_container_width=False, height=800)

        elif eff_chart_mode == "Efficiency vs Attribute Increase":
            fig = go.Figure()
            add_plot_layers(fig, eff_df["attr_jitter"], eff_df["score_jitter"], eff_df)
            
            fig.update_layout(
                height=800,
                title={
                    "text": "Upgrade Efficiency vs Attribute Increase<br><span style='font-size:14px; color:#e0ff8a;'>Best when you want high efficiency with minimal attribute increases</span>",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": dict(size=18)
                },
                plot_bgcolor="#111",
                paper_bgcolor="rgba(0, 0, 0, 0.0)",
                margin=dict(t=120),
                xaxis=dict(
                    title="<br>Total Attribute Increase",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                yaxis=dict(
                    title="<br>Upgrade Efficiency Score",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                legend=dict(
                    font=dict(
                        family="Inter, sans-serif",
                        size=12,
                        color="#f8f8f8"
                    )
                )
            )
            
            st.plotly_chart(fig, use_container_width=False, height=800)

        elif eff_chart_mode == "SP Cost vs Attribute Increase (Raw Value)":
            fig = go.Figure()
            add_plot_layers(fig, eff_df["sp_jitter"], eff_df["attr_jitter"], eff_df)
            
            fig.update_layout(
                height=800,
                title={
                    "text": "SP Cost vs Attribute Increase (Raw Value)<br><span style='font-size:14px; color:#e0ff8a;'>Compare the Raw Value of SP Cost to Attribute Increase</span>",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": dict(size=18)
                },
                plot_bgcolor="#111",
                paper_bgcolor="rgba(0, 0, 0, 0.0)",
                margin=dict(t=120),
                xaxis=dict(
                    title="<br>SP Cost",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                yaxis=dict(
                    title="<br>Attribute Increase",
                    title_font=dict(size=22),
                    showline=True,
                    linewidth=2,
                    linecolor='white',
                    mirror=True,
                    ticklabelposition="outside",
                    ticklabelstandoff=10
                ),
                legend=dict(
                    font=dict(
                        family="Inter, sans-serif",
                        size=12,
                        color="#f8f8f8"
                    )
                )
            )
            
            st.plotly_chart(fig, use_container_width=False, height=800)
            
            
# ───────────── Upgrade Efficiency Model ─────────────

elif view_mode == "Tier Progression Visualization":
    st.markdown("""
    <div style="
        background: linear-gradient(to right, #222, #0d0d0d);
        border-left: 5px solid #bdff00;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        color: #f8f8f8;
        margin-bottom: 1rem;
    ">
    Visualize how attribute requirements increase across tiers for each ability upgrade path.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Tier Progression Visualization")
    
    with st.expander("How Tier Progression Works - Understanding attribute curves"):
        st.markdown("""
        <span class="dropdown-highlight">Tier Progression </span>shows how attribute requirements scale from Bronze to Platinum.
        This helps you understand which abilities have steep difficulty curves and plan your upgrade path accordingly.
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-text">What You Will See</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul>
        <li><span class="dropdown-highlight">Smooth curves:</span> Abilities with consistent attribute increases</li>
        <li><span class="dropdown-highlight">Steep jumps:</span> Abilities that become much harder at higher tiers</li>
        <li><span class="dropdown-highlight">Multiple lines:</span> Abilities like Shifty that require two different attributes</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="gradient-text">Use Cases</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Compare difficulty curves** between similar abilities
        - **Identify plateau points** where upgrades become expensive
        - **Plan your build** by understanding which abilities scale smoothly vs. dramatically
        - **Multi-attribute abilities** show both stat requirements on the same chart
        """)

    # Filter controls
# Filter controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pos_filter = st.multiselect("Filter by Position", sorted(df["position"].unique()))
    
    with col2:
        available_archs = (
            df[df["position"].isin(pos_filter)]["archetype"].unique().tolist()
            if pos_filter
            else df["archetype"].unique().tolist()
        )
        arch_filter = st.multiselect("Filter by Archetype", sorted(available_archs))
    
    with col3:
        available_abilities = (
            df[df["archetype"].isin(arch_filter)]["ability"].unique().tolist()
            if arch_filter
            else df["ability"].unique().tolist()
        )
        ability_filter = st.multiselect("Filter by Ability", sorted(available_abilities))
    
    with col4:
        tier_transition = st.selectbox(
            "Tier Range",
            [
                "All Tiers (Full Progression)",
                "Bronze → Silver",
                "Bronze → Gold", 
                "Bronze → Platinum",
                "Silver → Gold",
                "Silver → Platinum",
                "Gold → Platinum"
            ]
        )

    # Process data for tier progression
    prog_df = df.copy()
    if pos_filter:
        prog_df = prog_df[prog_df["position"].isin(pos_filter)]
    if arch_filter:
        prog_df = prog_df[prog_df["archetype"].isin(arch_filter)]
    if ability_filter:
        prog_df = prog_df[prog_df["ability"].isin(ability_filter)]

    if prog_df.empty:
        st.warning("No matching data found for the selected filters.")
    else:
        # Create the tier progression chart
        fig = go.Figure()
        
        # Define tier order and colors
        tier_order = ["Bronze", "Silver", "Gold", "Platinum"]
        tier_colors = {
            "Bronze": "#CD7F32",
            "Silver": "#C0C0C0", 
            "Gold": "#FFD700",
            "Platinum": "#E5E4E2"
        }
        
        # Determine which tiers to show based on filter
        if tier_transition == "All Tiers (Full Progression)":
            display_tiers = tier_order
        else:
            start_tier, end_tier = [x.strip() for x in tier_transition.split("→")]
            start_idx = tier_order.index(start_tier)
            end_idx = tier_order.index(end_tier)
            display_tiers = tier_order[start_idx:end_idx + 1]
        
        # Custom color palette for abilities
        custom_palette = (
            px.colors.qualitative.Alphabet +
            px.colors.qualitative.Dark24 +
            px.colors.qualitative.Light24
        )
        
        # Process each ability
        ability_color_map = {}
        color_index = 0
        
        # Add jitter control
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            jitter_amount = st.slider(
                "Vertical Jitter (helps separate overlapping lines)",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Add slight vertical offset to prevent lines from stacking on top of each other"
            )
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Track if we've added the legend group title
        first_ability_trace = True
        
        for (pos, arch, ability), group in prog_df.groupby(["position", "archetype", "ability"]):
            # Sort by tier
            group = group.sort_values("tier", key=lambda x: [tier_order.index(t) for t in x])
            
            # Filter to only show the selected tier range
            group = group[group["tier"].isin(display_tiers)]
            
            # Create base ability identifier
            ability_id = f"{arch} · {ability}"
            
            # Process stat 1 if it exists
            if group["stat_1_name"].notna().any():
                stat_1_name = group["stat_1_name"].iloc[0]
                stat_1_values = []
                tier_labels = []
                
                for tier in display_tiers:
                    tier_data = group[group["tier"] == tier]
                    if not tier_data.empty and pd.notna(tier_data["stat_1_value"].iloc[0]):
                        stat_1_values.append(tier_data["stat_1_value"].iloc[0])
                        tier_labels.append(tier)
                
                if len(stat_1_values) > 1:  # Only plot if we have multiple points
                    if ability_id not in ability_color_map:
                        ability_color_map[ability_id] = custom_palette[color_index % len(custom_palette)]
                        color_index += 1
                    
                    # Apply jitter to y-values
                    jitter_offset = np.random.uniform(-jitter_amount, jitter_amount)
                    jittered_values = [val + jitter_offset for val in stat_1_values]
                    
                    fig.add_trace(go.Scatter(
                        x=tier_labels,
                        y=jittered_values,
                        mode='lines+markers',
                        name=f"{ability_id} ({stat_1_name})",
                        line=dict(color=ability_color_map[ability_id], width=3),
                        marker=dict(size=8, color=ability_color_map[ability_id]),
                        hovertemplate=(
                            f"<b>{ability_id}</b><br>"
                            f"Stat: {stat_1_name}<br>"
                            "Tier: %{x}<br>"
                            "Value: %{y:.0f}<br>" 
                            f"Position: {pos}<extra></extra>"
                        ),
                        showlegend=True,
                        legendgrouptitle=dict(
                            text="Archetype (Color) · Ability (Symbol)",
                            font=dict(size=16, color="white", family="Russo One")
                        ) if first_ability_trace else None,
                    ))
                    first_ability_trace = False
            
            # Process stat 2 if it exists (for multi-attribute abilities)
            if group["stat_2_name"].notna().any():
                stat_2_name = group["stat_2_name"].iloc[0]
                stat_2_values = []
                tier_labels = []
                
                for tier in display_tiers:
                    tier_data = group[group["tier"] == tier]
                    if not tier_data.empty and pd.notna(tier_data["stat_2_value"].iloc[0]):
                        stat_2_values.append(tier_data["stat_2_value"].iloc[0])
                        tier_labels.append(tier)
                
                if len(stat_2_values) > 1:  # Only plot if we have multiple points
                    if ability_id not in ability_color_map:
                        ability_color_map[ability_id] = custom_palette[color_index % len(custom_palette)]
                        color_index += 1
                    
                    # Use dashed line for second stat
                    fig.add_trace(go.Scatter(
                        x=tier_labels,
                        y=stat_2_values,
                        mode='lines+markers',
                        name=f"{ability_id} ({stat_2_name})",
                        line=dict(color=ability_color_map[ability_id], width=3, dash='dash'),
                        marker=dict(size=8, color=ability_color_map[ability_id], symbol='diamond'),
                        hovertemplate=(
                            f"<b>{ability_id}</b><br>"
                            f"Stat: {stat_2_name}<br>"
                            "Tier: %{x}<br>"
                            "Value: %{y}<br>"
                            f"Position: {pos}<extra></extra>"
                        ),
                        showlegend=True,
                        legendgrouptitle=dict(
                            text="Archetype (Color) · Ability (Symbol)",
                            font=dict(size=16, color="white", family="Russo One")
                        ) if first_ability_trace else None,
                    ))
                    first_ability_trace = False
        

        title_text = f"Tier Progression: {tier_transition}<br><span style='font-size:14px; color:#e0ff8a;'>Solid lines = Primary stat · Dashed lines = Secondary stat</span>"
        

        fig.update_layout(
            height=700,
            title={
                "text": title_text,
                "x": 0.5,
                "xanchor": "center",
                "font": dict(size=18, color="white")
            },
            plot_bgcolor="#111",
            paper_bgcolor="rgba(0, 0, 0, 0.0)",
            margin=dict(t=120),
            xaxis=dict(
                title="<br>Tier",
                title_font=dict(size=20, color="white"),
                showline=True,
                linewidth=2,
                linecolor='white',
                mirror=True,
                tickfont=dict(size=14, color="white"),
                categoryorder="array",
                categoryarray=display_tiers
            ),
            yaxis=dict(
                title="<br>Attribute Value",
                title_font=dict(size=20, color="white"),
                showline=True,
                linewidth=2,
                linecolor='white',
                mirror=True,
                tickfont=dict(size=14, color="white")
            ),
            legend=dict(
                font=dict(
                    family="Inter, sans-serif",
                    size=12,
                    color="#f8f8f8"
                )
            ),
            font=dict(color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True, height=700)
        
        # Add summary statistics
        if not prog_df.empty:
            st.markdown("### Progression Summary")
            
            summary_data = []
            for (pos, arch, ability), group in prog_df.groupby(["position", "archetype", "ability"]):

                group = group[group["tier"].isin(display_tiers)]
                group = group.sort_values("tier", key=lambda x: [tier_order.index(t) for t in x])

                if len(group) > 1:
                    first_tier = group.iloc[0]
                    last_tier = group.iloc[-1]

                    stat_1_increase = 0
                    stat_2_increase = 0

                    if pd.notna(first_tier["stat_1_value"]) and pd.notna(last_tier["stat_1_value"]):
                        stat_1_increase = last_tier["stat_1_value"] - first_tier["stat_1_value"]

                    if pd.notna(first_tier["stat_2_value"]) and pd.notna(last_tier["stat_2_value"]):
                        stat_2_increase = last_tier["stat_2_value"] - first_tier["stat_2_value"]

                    total_increase = stat_1_increase + stat_2_increase

                    summary_data.append({
                        "Position": pos,
                        "Archetype": arch,
                        "Ability": ability,
                        "Tier Range": f"{first_tier['tier']} → {last_tier['tier']}",
                        "Total Attribute Increase": total_increase,
                        "Primary Stat": f"{first_tier['stat_1_name']}: +{stat_1_increase}" if stat_1_increase > 0 else "",
                        "Secondary Stat": f"{first_tier['stat_2_name']}: +{stat_2_increase}" if stat_2_increase > 0 else ""
                    })

            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values("Total Attribute Increase", ascending=False)
                st.dataframe(summary_df, use_container_width=True)
