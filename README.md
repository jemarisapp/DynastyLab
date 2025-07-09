# DynastyLab-IQ

# Upgrade Efficiency Model for Dynasty Mode

An interactive dashboard for analyzing the cost-efficiency of ability upgrades in EA SPORTS College Football 26’s Dynasty Mode.

Built with **Streamlit**, **Plotly**, and **custom data modeling**, this tool helps players make smarter upgrade decisions by visualizing efficiency tradeoffs across archetypes, tiers, and stats.

---

## Features

- **Efficiency Score Modeling**: Calculates upgrade value using SP cost, attribute gain, and tier difficulty scaling.
- **Blend Slider**: Adjust weighting between SP cost and stat gain for flexible prioritization.
- **Three Visual Views**:
  - Efficiency vs SP Cost
  - Efficiency vs Attribute Gain
  - SP Cost vs Attribute Gain (Raw Value)
- **Custom Visualization Layers**:
  - Archetype = Fill Color
  - Tier Transition = Border Stroke
  - Ability = Marker Shape
- **Explainer Panel**: Expandable guide on how the model works and how to interpret each view.

---

## Skills Demonstrated

- Data modeling and normalization
- Feature engineering (difficulty weighting, scoring formulas)
- Data visualization with Plotly
- UX/UI thinking for BI tools
- Interactive filtering and toggles via Streamlit
- Dashboard storytelling for decision-making

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app with logic and visuals |
| `theme.css` | dark theme with gradient accents |
| `bot_data_archetypes.db` | (Optional) SQLite DB with upgrade paths |
| `README.md` | This file |

---

## Getting Started

Visit - https://dynastylab.streamlit.app/

---

## Contact

Built by [Jemari Sapp](https://github.com/jemarisapp)  
Interested in BI, product analytics, and game data.

