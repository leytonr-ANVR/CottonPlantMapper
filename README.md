
# Cotton Plant Mapper

A Streamlit web app for entering cotton fruiting data by **node** and **position** and generating a visual cotton plant map.

## Fruit types
- Boll
- Square
- White Flower
- Missing Fruit
- None

## Features
- Editable node × position data table
- Quick-entry controls for field use
- Cotton plant diagram with alternating fruiting branches
- Node-position labels (for example 12-1, 12-2, 12-3)
- Summary counts and fruit retention percentage
- CSV import/export
- PNG and PDF map downloads

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit address shown in your terminal.

## Notes
Position 1 is treated as the fruiting site closest to the main stem, with Positions 2 and 3 further out along the branch.
