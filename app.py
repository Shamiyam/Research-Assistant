import streamlit as st
import asyncio
from Kbot import ResearchBot, GeminiProvider, DuckDuckGoSearchProvider  # <-- your existing file

# Page config (title, icon, wide mode)
st.set_page_config(page_title="ResearchBot", page_icon="🔍", layout="centered")

st.title("🔍 ResearchBot")
st.caption("Ask any question — I search the web, read articles, and give you a cited answer (powered by Gemini + DuckDuckGo)")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", value=st.secrets.get("GOOGLE_API_KEY", ""), type="password")
    provider = st.selectbox("Search Provider", ["ddg", "google"], index=0)
    verbose = st.checkbox("Verbose mode (show sub-queries & sources)", value=False)

if not api_key:
    st.warning("⚠️ Enter your Google Gemini API Key in the sidebar to start.")
    st.stop()

# Initialize providers (only once)
@st.cache_resource
def get_bot():
    llm = GeminiProvider(api_key)
    search = DuckDuckGoSearchProvider() if provider == "ddg" else None  # add Google provider later if you want
    return ResearchBot(llm, search)

bot = get_bot()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"[{src['id']}] [{src['description']}]({src['url']})")

# User input
if prompt := st.chat_input("What do you want to research?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            # Run your existing async workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(bot.research(prompt))
            loop.close()

            answer = result["answer"]
            st.markdown(answer)

            # Optional verbose details
            if verbose:
                with st.expander("Sub-queries used"):
                    st.write(result["sub_queries"])
                with st.expander("Raw sources found"):
                    st.write(result["search_results_count"])

            # Store for citation display
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": result.get("sources", [])
            })