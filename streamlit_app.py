import streamlit as st
from main import graph
import io

st.set_page_config(page_title="Study Guide Generator", layout="centered")

st.title("Study Guide Generator Agent")

topic = st.text_input("Enter a topic to study (e.g., Graph Theory)")

# Add toggle for 7-day plan
include_plan = st.toggle("Include 7-Day Study Plan", value=True)

if st.button("Generate Study Guide"):
    if not topic:
        st.error("Please enter a topic to generate a study guide.")
    else:
        with st.spinner("Running agent..."):
            #
            initial_state = {"topic": topic, "include_plan": include_plan}
            result = graph.invoke(initial_state)
            
            st.success("Guide ready!")

            st.subheader("Summary")
            st.markdown(result['summary'])

            
            if include_plan and 'study_plan' in result and result['study_plan']:
                st.subheader("7-Day Plan")
                st.markdown(result['study_plan'])

            st.subheader("Resources")
            st.markdown(result['resources'])

            
            filename = f"{topic.replace(' ', '_').lower()}_study_guide.pdf"
            
            
            if 'pdf_bytes' in result:
                st.download_button(
                    "Download PDF",
                    data=io.BytesIO(result['pdf_bytes']),
                    file_name=filename,
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate PDF")