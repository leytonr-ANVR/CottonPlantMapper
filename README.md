# Cotton Plant Mapper v31

Bug fix for Streamlit Cloud NameError in `interactive_map()`.

Cause:
A CSS rule for the R / V / VL node-type dropdown was accidentally inserted inside an HTML f-string using normal `{ }` CSS braces. Python treated the CSS as an f-string expression and raised a NameError at `font-weight:700`.

Fix:
- removed the CSS rule from the interactive map iframe
- moved it into the main Streamlit CSS section
- validated the app with Python `compile()` before packaging
- Zoom In, Zoom Out and Fit remain unchanged
