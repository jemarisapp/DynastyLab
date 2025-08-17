import streamlit as st
import sqlite3
import pandas as pd
import openai
from openai import OpenAI
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import math
from typing import Optional, Tuple, List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Trilo Ability Lab - AI Assistant & Analytics for CFB Dynasties",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load theme
try:
    with open("theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Theme file not found. Using default styling.")

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Get OpenAI client with caching"""
    try:
        return OpenAI(api_key=st.secrets.get("openai_api_key"))
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return None

client = get_openai_client()

# Constants
DB_PATH = "bot_data_archetypes.db"
MODEL = "gpt-4"

def execute_query(sql: str) -> Tuple[List[str], List[List[Any]]]:
    """Execute SQL query with better error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        return col_names, rows
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return [], [[f"Database Error: {e}"]]
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return [], [[f"Error: {e}"]]

# Tier configuration
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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data() -> pd.DataFrame:
    """Load data with caching and error handling"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()





# ───────────── AI Assistant ─────────────

df = load_data()



def ask_upgrade_assistant(user_question: str) -> str:
    """AI assistant for upgrade advice with improved error handling"""
    if not client:
        return "OpenAI client not available. Please check your configuration."
    
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
                    stat1 = f"{row['stat_1_name']} +{row['stat_1_value']}" if pd.notna(row['stat_1_value']) else ""
                    stat2 = f"{row['stat_2_name']} +{row['stat_2_value']}" if pd.notna(row['stat_2_value']) else ""
                    
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
        

        
    except Exception as e:
        data_summary = f"Error retrieving data: {e}"
        st.error(f"Database error: {e}")

    prompt = f"""You are an expert college football upgrade strategist built into a EA Sports College Football 26 upgrade planner.

You have access to a complete database of player upgrade paths and detailed ability descriptions that help identify the best value upgrades and explain what each ability does.

Database schema:
{schema}

Current upgrade data (including ability descriptions):
{data_summary}

CORE INSTRUCTIONS:
1. Always provide specific, actionable advice using real numbers from the data
2. Calculate exact SP costs for any upgrade paths mentioned
3. Include ability descriptions when explaining what abilities do or recommending upgrades
4. Compare multiple options when relevant to help users make informed decisions
5. Focus on clear cost-benefit analysis using the data provided
6. Prioritize clarity and practicality over technical jargon
7. Be extremely concise - summarize rather than detail every option
8. Lead with the answer, then provide only essential supporting details

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
- Keep descriptions brief - focus on what the ability does, not lengthy explanations



RESPONSE GUIDELINES:
- Lead with a direct, concise answer (1-2 sentences max)
- Use bullet points for multiple recommendations
- Always include total SP costs when discussing upgrade paths
- Include ability descriptions when explaining what abilities do
- When comparing options, clearly state which is more cost-effective and why
- NEVER say that an ability increases or improves a stat like SPD, ACC, or COD
- ALWAYS say "requires" or "needs at least" when referencing attribute thresholds
- Be clear: upgrades unlock the ability at a tier once the player has the required stats and pays the SP cost
- Always say Skill Points instead of SP
- Always say the Attribute Requirements before the Skill Point Requirements
- Keep responses focused and avoid unnecessary repetition
- DO NOT list every tier for every archetype - summarize instead
- For "which archetypes get X" questions, just list the archetypes, don't detail every tier


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
- "What should I upgrade with X SP?" → Show options within budget, explain costs and benefits
- "How much to max out [ability]?" → Calculate total cost and explain what each tier does
- "Which upgrades are most efficient?" → Compare costs and attribute requirements directly
- "Best upgrades for [position/archetype]?" → Show options, explain what each tier provides
- "What does [ability] do?" → Provide descriptions for all tiers of that ability
- "Explain [ability] tiers" → Detail what each tier does and costs
- "Which archetypes get [ability]?" → List archetypes only, don't detail every tier

TONE & STYLE:
- Speak as a knowledgeable coach making strategic decisions
- Be confident but explain your reasoning
- Provide clear, data-driven recommendations based on costs and requirements
- Reference ability descriptions naturally in recommendations
- Keep responses concise: aim for 3-5 bullet points maximum
- Lead with the answer, then provide supporting details

EXAMPLE QUALITY RESPONSES:
Bad: "Silver upgrades cost varying amounts"
Good: "Silver Quick Release (8 SP) and Silver Pocket Presence (10 SP) are your best Silver options for Pocket Passers. Quick Release provides 'Moderately improved ability to quickly release the ball' while Pocket Presence gives 'Enhanced awareness and composure in the pocket'. Choose based on your playstyle needs."

Now answer this question with specific numbers, ability descriptions, clear recommendations, and strategic reasoning:

RESPONSE STRUCTURE:
1. Direct answer (1-2 sentences max)
2. Key details in bullet points (2-3 max)
3. Strategic reasoning (1 sentence max)

CRITICAL: Keep total response under 100 words. Be extremely concise.

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

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load data with error handling
df = load_data()
if df.empty:
    st.error("Failed to load data. Please check your database connection.")
    st.stop()

# Main header
st.markdown('<div class="gradient-hero">Trilo Ability Lab</div>', unsafe_allow_html=True)

st.markdown("""
Master your College Football Dynasty Mode with AI-powered upgrade optimization, data-driven insights, and strategic planning tools.""")

with st.expander("Explore Features"):
    st.markdown("""
    <div class="gradient-text">What You Can Do</div>

    <span class="dropdown-highlight">Ask the AI Assistant</span> - Get instant, expert advice on upgrade strategies. Ask questions like:
    - "What's the best way to spend 25 SP on a Pocket Passer?"
    - "Which Silver tier upgrades give the best value?"
    - "How much SP does it cost to max out Quick Jump?"

    <span class="dropdown-highlight">Plan Your Upgrades</span> - Use the visual upgrade planner to:
    - See exactly how much SP each upgrade path costs
    - Compare current vs. target ability tier
    - Track your total SP budget in real-time
    - Watch the tier bars light up as you plan upgrades
    """, unsafe_allow_html=True)

EXAMPLE_QUESTION = "Which archetypes get Shifty?"

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
        if not client:
            st.error("OpenAI client not available. Please check your configuration.")
        else:
            with st.spinner("Thinking..."):
                try:
                    assistant_reply = ask_upgrade_assistant(user_input)
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_input,
                        'answer': assistant_reply
                    })
                except Exception as e:
                    st.error(f"Failed to get AI response: {e}")

    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            # Auto-expand only the most recent message
            is_latest = i == 0
            with st.expander(f"Q: {chat['question']}", expanded=is_latest):
                st.markdown(f"<div class='chat-response'>{chat['answer']}</div>", unsafe_allow_html=True)

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ───────────── Upgrade Planner ─────────────

col1, col2, col3 = st.columns([5, 3, 1.5])

# ───────────── COLUMN 1: Dropdowns and Bars ─────────────
with col1:
    st.markdown("### Upgrade Planner")

    col1_sp, col_pos, col_arch = st.columns(3)
    with col1_sp:
        max_sp = st.number_input("Player SP", value=20, min_value=1, step=1, help="Total Skill Points available")
    with col_pos:
        position = st.selectbox("Position", sorted(df["position"].unique()), help="Select player position")
    with col_arch:
        archetype = st.selectbox("Archetype", sorted(df[df["position"] == position]["archetype"].unique()), help="Select player archetype")

    st.markdown("### Abilities")
    
    filtered_df = df[(df["position"] == position) & (df["archetype"] == archetype)]

    if filtered_df.empty:
        st.warning("No abilities found for the selected position and archetype.")
    else:
        ability_dict = {}
        for ability in filtered_df["ability"].unique():
            tier_rows = filtered_df[filtered_df["ability"] == ability]
            ability_dict[ability] = {
                row["tier"]: {
                    "sp_cost": row["sp_cost"],
                    "stat_1": (
                        f"{row['stat_1_name']} <span class='stat-number'>{int(row['stat_1_value'])}</span>"
                        if pd.notna(row["stat_1_name"]) and pd.notna(row["stat_1_value"]) else ""
                    ),
                    "stat_2": (
                        f"{row['stat_2_name']} <span class='stat-number'>{int(row['stat_2_value'])}</span>"
                        if pd.notna(row["stat_2_name"]) and pd.notna(row["stat_2_value"]) else None
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
                                "box-shadow: 0 0 6px 3px #f3aa07, inset 0 0 15px 0px #f3aa07;"
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
                                "box-shadow: 0 0 15px 3px #f3aa07, inset 0 0 15px 0px #f3aa07;"
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
                            0 0 20px 3px #f3aa07;  /* Outer glow for upgrade */
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
