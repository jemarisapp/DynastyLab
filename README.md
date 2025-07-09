# 🧠 DynastyLab 🏈  
**AI-powered upgrade optimization for EA SPORTS College Football Dynasty Mode**

Transform your Dynasty Mode experience with data-driven upgrade decisions, interactive visualizations, and intelligent recommendations.

[🚀 Try Live Demo](https://dynastylab.streamlit.app/) • [📊 View Case Study](https://www.jemarisapp.com/projects/dynastylab)

---

## ✨ Features

### 🤖 AI Assistant
- Ask natural language questions about upgrade strategies  
- Get personalized recommendations based on your archetype and playstyle  
- Powered by GPT-4o for contextual, strategic advice  

### 📈 Efficiency Scoring Model
- Dynamic algorithm that weighs SP cost vs. attribute gain  
- Adjustable weighting system for personalized optimization  
- Identifies the most efficient upgrade paths across all tiers  

### 🎯 Interactive Upgrade Planner
- Visual tier progression with real-time SP calculations  
- Color-coded ability grids showing upgrade status  
- Simulate upgrade sequences before spending points  

### 📊 Multi-View Analytics
- **Efficiency vs SP Cost:** Find the best value upgrades  
- **Efficiency vs Attribute Gain:** Optimize for maximum stat improvement  
- **Cost vs Gain Analysis:** Raw comparison without scoring influence  

---

## 🎮 How It Works

1. Select your player details (position, archetype, current SP)  
2. Choose your optimization preference using the weight slider  
3. Explore upgrade options through interactive visualizations  
4. Ask the AI assistant for strategic recommendations  
5. Plan your upgrade path with the visual progression tool  

---

## 🛠️ Technical Stack

- **Frontend:** VSCode, Python, Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Plotly  
- **Database:** SQLite  
- **AI Integration:** OpenAI GPT-4o API  
- **Deployment:** Streamlit Cloud  

---

## 📊 Key Insights

The tool reveals several strategic insights:

- Speedster archetypes offer **23% higher attribute gain per SP**
- Silver → Gold upgrades are **34% more efficient** than Gold → Platinum
- **Diminishing returns** become severe after 90 attribute points
- **Mid-tier optimization** often outperforms premium upgrade paths

---

## 📁 Project Structure

```bash
dynastylab-ai/
├── .streamlit/
│   ├── config.toml          # Streamlit theme + settings
│   └── secrets.toml         # API key for OpenAI
├── app.py                   # Main Streamlit app
├── theme.css                # Custom CSS styling
├── bot_data_archetypes.db   # SQLite data
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🎨 Design Philosophy

DynastyLab AI bridges the gap between serious data analysis and engaging user experience:

- Gaming-first aesthetic with dark theme and neon accents  
- Intuitive interactions that feel natural to Dynasty Mode players  
- Professional analytics capabilities wrapped in user-friendly design  
- Responsive layout that works across desktop and mobile  

---

## 🔧 Configuration

### Efficiency Model Parameters

```python
efficiency_score = 100 - (
    sp_weight * sp_cost + 
    attribute_weight * (attribute_gain * difficulty_modifier)
)
```

- **SP Weight:** Emphasizes cost efficiency  
- **Attribute Weight:** Prioritizes stat improvement  
- **Difficulty Modifier:** `1 + (final_attribute / 100)` to account for diminishing returns  

### Database Schema

- **upgrades:** All upgrade paths with costs and stat gains  
- **archetypes:** Player archetype definitions  
- **tiers:** Tier requirements and thresholds  

---

## 📞 Contact

**Your Name** – your.email@example.com  

[GitHub Project Link](https://github.com/yourusername/dynastylab-ai)

<details>
<summary>🔧 Developer Setup (Optional)</summary>

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dynastylab-ai.git
cd dynastylab-ai

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your OpenAI API key to the .env file
```
