from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document,
    primary_llm # Imported to generate the titles
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os
import json

TITLES_FILE = "chat_titles.json"

# ========================= Chat Title Persistence Helper =========================

def get_chat_title(thread_id):
    """
    Fetches a saved title from the local JSON file. 
    If it doesn't exist, it generates one using the primary LLM and saves it permanently.
    """
    # 1. Read existing titles from disk if file exists
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                titles = json.load(f)
                if thread_id in titles:
                    return titles[thread_id]
        except Exception:
            pass

    # 2. Extract the first human message from the graph history to summarize
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)

    if not state or not state.values or "messages" not in state.values:
        return f"Chat ({thread_id[:6]})"

    first_user_msg = None
    for msg in state.values["messages"]:
        if isinstance(msg, HumanMessage):
            first_user_msg = msg.content
            break

    if not first_user_msg:
        return f"Chat ({thread_id[:6]})"

    # 3. Request your primary Gemini model to generate a short headline
    try:
        prompt = (
            f"Summarize the following user prompt into a short, descriptive 3 to 4 word chat menu title. "
            f"Output ONLY the title itself with no quotation marks, formatting, or punctuation: {first_user_msg}"
        )
        title = primary_llm.invoke(prompt).content.strip()
    except Exception:
        # Fallback to snippet if LLM is busy/unavailable
        title = first_user_msg[:25] + "..." if len(first_user_msg) > 25 else first_user_msg

    # 4. Permanently write to disk
    titles = {}
    if os.path.exists(TITLES_FILE):
        try:
            with open(TITLES_FILE, "r") as f:
                titles = json.load(f)
        except Exception:
            pass

    titles[thread_id] = title

    with open(TITLES_FILE, "w") as f:
        json.dump(titles, f)

    return title


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):
    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():
    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):
    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    # Return saved messages or empty list
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Handle direct interrupts
        direct_interrupts = getattr(state_snapshot, "interrupts", ()) or ()
        if direct_interrupts:
            return direct_interrupts[0]

        # Handle nested task interrupts
        tasks = getattr(state_snapshot, "tasks", ()) or ()
        for task in tasks:
            task_interrupts = getattr(task, "interrupts", ()) or ()
            if task_interrupts:
                return task_interrupts[0]

    except Exception:
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """
    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.
    """
    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:
        save_pending_interrupt(thread_id, pending_interrupt)
    else:
        current_pending = st.session_state.get("pending_hitl")
        if current_pending is not None and current_pending.get("thread_id") == thread_id:
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.
    """
    pending_hitl = st.session_state.get("pending_hitl")

    if not pending_hitl:
        st.warning("There is no pending action to approve or reject.")
        return

    interrupted_thread_id = pending_hitl["thread_id"]

    resume_config = {
        "configurable": {"thread_id": interrupted_thread_id},
        "metadata": {"thread_id": interrupted_thread_id},
        "run_name": "hitl_resume_trace",
    }

    try:
        with st.chat_message("assistant"):
            status_holder = {
                "box": st.status("🔄 Resuming the requested action...", expanded=True)
            }

            def resumed_ai_only_stream():
                # Resume the graph with the human decision string ("yes"/"no")
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if isinstance(message_chunk, AIMessage):
                        if message_chunk.content:
                            yield message_chunk.content

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        if resumed_ai_message:
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        st.rerun()

    except Exception as error:
        st.error(f"Could not resume the requested action: {error}")


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖"
)

st.title("Agentic Chatbot with LangGraph")

# Session state checks
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

add_thread(st.session_state["thread_id"])
sync_pending_interrupt(st.session_state["thread_id"])


# ========================= Sidebar threading feature =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")

# Display permanent custom titles instead of long raw IDs
for thread_id in st.session_state["chat_threads"][::-1]:
    
    # Run title persistence calculation
    display_name = get_chat_title(thread_id)

    # Styling active session differently in the sidebar
    if thread_id == st.session_state["thread_id"]:
        button_label = f"💬 {display_name}"
    else:
        button_label = f"📄 {display_name}"

    if st.sidebar.button(button_label, key=thread_id, use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages
        sync_pending_interrupt(thread_id)
        st.rerun()


# ========================= Main chat interface =========================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ========================= HITL approval interface =========================

pending_hitl = st.session_state.get("pending_hitl")
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id") == st.session_state["thread_id"]
)

if current_thread_has_pending_hitl:
    st.warning(
        "🧑 **Human approval required**\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    with approve_column:
        if st.button(
            "✅ Approve Action",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):
            resume_hitl_execution("yes")

    with reject_column:
        if st.button(
            "❌ Reject Action",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(current_thread_has_pending_hitl)
)

user_input = None

if submission:
    user_input = submission.text
    uploaded_files = submission.files

    if uploaded_files:
        uploaded_pdf = uploaded_files[0]
        temporary_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temporary_file.name

            with st.spinner(f"Processing {uploaded_pdf.name}..."):
                ingest_rag_document(temporary_file_path)

            st.toast(f"{uploaded_pdf.name} processed successfully.", icon="✅")

        except Exception as error:
            st.error(f"PDF processing failed: {error}")
        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

if user_input:
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            pending_interrupt = get_pending_interrupt(st.session_state["thread_id"])
            if pending_interrupt is not None:
                save_pending_interrupt(st.session_state["thread_id"], pending_interrupt)
                yield "\n\n⚠️ **This action requires confirmation.** Please review and respond using the decision panel below."

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(label="⏸️ Waiting for human approval", state="complete", expanded=False)
            else:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # Force a rerun if an interrupt happened, to load the permanent titles file and display controls right away
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id") == st.session_state["thread_id"]
    ):
        st.rerun()
    else:
        st.rerun()