# 🧠 DynastyLab AI 🏈  
**AI-powered upgrade optimization for EA SPORTS College Football Dynasty Mode**

Transform your Dynasty Mode experience with data-driven upgrade decisions, interactive visualizations, and intelligent recommendations.

[🚀 Try Live Demo](#) • [📊 View Case Study](#)

![DynastyLab Screenshot](assets/screenshots/hero.png)

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

- **Frontend:** Streamlit  
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

## 🚀 Getting Started

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

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env
```

### Running Locally

```bash
streamlit run app.py
# Navigate to http://localhost:8501
```

---

## 📁 Project Structure

```
dynastylab-ai/
├── app.py                  # Main Streamlit application
├── components/
│   ├── ai_assistant.py     # GPT-4o integration
│   ├── efficiency_model.py # Scoring algorithm
│   ├── visualizations.py   # Plotly charts
│   └── upgrade_planner.py  # Interactive planner
├── data/
│   ├── dynastylab.db       # SQLite database
│   └── load_data.py        # Data loading utilities
├── assets/
│   └── screenshots/        # Interface screenshots
├── requirements.txt
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or enhancement requests.

### Development Setup

```bash
# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📈 Future Enhancements

- Multi-player team optimization  
- Historical upgrade tracking  
- League-wide analytics dashboard  
- Mobile app version  
- Integration with EA Sports API (if available)  

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- EA SPORTS for creating Dynasty Mode  
- The College Football gaming community for inspiration  
- OpenAI for GPT-4o capabilities  

---

## 📞 Contact

**Your Name** – your.email@example.com  
[GitHub Project Link](https://github.com/yourusername/dynastylab-ai)

> Built with ❤️ for the Dynasty Mode community
