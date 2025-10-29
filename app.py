import streamlit as st
from main import solve_math_question, store_feedback, is_math_question

st.set_page_config(page_title="Math Tutor Agent", page_icon="🧮")
st.title("Math Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a math question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # ✅ Check if question is math-related first
    if not is_math_question(user_input):
        reply = "⚠️ Please enter a math question."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.stop()

    # ✅ Proceed with solving since it *is* a math question
    with st.chat_message("assistant"):
        answer, kb_used, web_used = solve_math_question(user_input)
        st.write(answer)

        if answer == "No solution found":
            st.error("No solution found")
        else:
            if kb_used:
                st.info("Knowledge base used")
            elif web_used:
                st.info("Tavily Web Search Used")
            else:
                st.warning("Pure Model Reasoning (No External Data)")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Provide Feedback"):
        feedback = st.radio("Was this helpful?", ["Yes", "No"], horizontal=True)
        if st.button("Submit Feedback"):
            store_feedback(user_input, answer, feedback)
            st.success("✅ Feedback Saved")
