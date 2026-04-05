import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load API key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please add it as an environment variable.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

prompt_template = PromptTemplate(
    input_variables=["matchup", "no_of_paras", "language"],
    template="""
You are an expert in college basketball.

Your job is to predict games from the 2026 NCAA championship tournament.

Prediction rules:
- Give 50% weight to the quality of current NBA players from each NCAA team.
- Give 30% weight to which mascot would defeat the other in a fight.
- Give 20% weight to the current seedings. Where a higher seed gets higher weight. 

The user will provide a matchup in this format: team 1 versus team 2.

Write the response in {language}.
Use about {no_of_paras} paragraph(s).
Include:
1. Predicted winner
2. Predicted final score
3. Brief reasoning
4. A mascot battle tiebreaker if relevant

Matchup: {matchup}
"""
)

st.title("NCAA Predict")

matchup = st.text_input("Enter a matchup (e.g., Duke vs UConn):")
no_of_paras = st.number_input("Enter the number of paragraphs:", min_value=1, max_value=5, value=1)
language = st.text_input("Enter a language:", value="English")

if matchup:
    try:
        prompt = prompt_template.format(
            matchup=matchup,
            no_of_paras=no_of_paras,
            language=language
        )

        response = llm.invoke(prompt)
        st.write(response.content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
