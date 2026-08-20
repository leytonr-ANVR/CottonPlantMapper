# Cotton Plant Mapper v7

Adds the following key totals to both the dashboard and exported PDF:

- **Total Nodes**
- **Total Positions**
- **Held Positions**

**Held Positions** counts mapped positions currently holding a Boll, Cracked Boll, Square or White Flower. Missing fruit is not counted as held.

The PDF summary strip also shows Missing Fruit and Retention for quick reporting.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
