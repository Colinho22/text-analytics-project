# import libraries
import streamlit as st
import pandas as pd
from utils import (load_models, extract_text_from_pdf, classify_with_model,  
                   save_results, load_results_history, get_summary_stats)

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide"
)


# load all models once at startup and cache them
@st.cache_resource
def initialize_models():
    return load_models()


# initialize session state variables
def initialize_session_state():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = None
    if "filename" not in st.session_state:
        st.session_state.filename = None
    if "classification_results" not in st.session_state:
        st.session_state.classification_results = None
    if "correctness_feedback" not in st.session_state:
        st.session_state.correctness_feedback = {}
    if "results_submitted" not in st.session_state:
        st.session_state.results_submitted = False


# reset all session state for new document
def reset_session():
    st.session_state.uploaded_file = None
    st.session_state.extracted_text = None
    st.session_state.filename = None
    st.session_state.classification_results = None
    st.session_state.correctness_feedback = {}
    st.session_state.results_submitted = False


def calculate_rankings(results, correctness):
    models_data = []

    for model_name, result in results.items():
        models_data.append({
            "name": model_name,
            "prediction": result["prediction"],
            "time_ms": result["time_ms"],
            "correct": correctness.get(model_name, False)
        })

    # sort: correct first (by speed), then incorrect (by speed)
    correct_models = sorted(
        [m for m in models_data if m["correct"]],
        key=lambda x: x["time_ms"]
    )
    incorrect_models = sorted(
        [m for m in models_data if not m["correct"]],
        key=lambda x: x["time_ms"]
    )

    # assign ranks
    ranked = []
    for rank, model in enumerate(correct_models + incorrect_models, 1):
        model["rank"] = rank
        ranked.append(model)

    return ranked


def main():
    st.title("🤖 Model Comparison")
    st.markdown("Compare 4 different text classification approaches")

    initialize_session_state()

    # load models
    with st.spinner("Loading models..."):
        models = initialize_models()

    st.sidebar.header("About")
    st.sidebar.info(
        "Upload a PDF document and see how different ML models classify it.\\n\\n"
        "Models: N-grams, Embeddings, LSTM, Transformer"
    )

    # Create tabs
    tab1, tab2 = st.tabs(["🧪 Test Models", "📊 View Results"])

    with tab1:
        show_test_tab(models)

    with tab2:
        show_results_tab()


# tab 1: test models with PDF upload
def show_test_tab(models):
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to classify"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.filename = uploaded_file.name

        # extract text
        if st.session_state.extracted_text is None:
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.extracted_text = text
                st.success(f"✅ Extracted {len(text)} characters from {uploaded_file.name}")

        # show preview
        st.subheader("Text Preview")
        preview_text = st.session_state.extracted_text[:500]
        st.text_area(
            "First 500 characters:",
            preview_text,
            height=150,
            disabled=True
        )

        if len(st.session_state.extracted_text) > 0:
            st.success("✅ Ready to classify!")
        else:
            st.warning("⚠️ No text extracted from PDF. Please upload a different file.")

        # document classification
        st.header("🔍 Document Classification")

        if st.button("🚀 Classify with All Models", type="primary"):
            results = {}

            # create columns for real-time updates
            cols = st.columns(4)
            model_names = ["N-grams", "Embeddings", "LSTM", "Transformer"]
            status_placeholders = {}

            # set up status displays
            for idx, model_name in enumerate(model_names):
                with cols[idx]:
                    st.subheader(model_name)
                    status_placeholders[model_name] = st.empty()
                    status_placeholders[model_name].info("⏳ Waiting...")

            # run models sequentially
            for model_name in model_names:
                status_placeholders[model_name].warning("🔄 Deciding...")

                prediction, time_ms = classify_with_model(
                    models[model_name],
                    st.session_state.extracted_text,
                    model_name
                )

                results[model_name] = {
                    "prediction": prediction,
                    "time_ms": time_ms
                }

                # format time
                if time_ms < 1000:
                    time_str = f"{time_ms}ms"
                else:
                    time_str = f"{time_ms/1000:.2f}s"

                status_placeholders[model_name].success(
                    f"✅ **{prediction}** ⏱️ {time_str}"
                )

            st.session_state.classification_results = results

        # display results persistently (even after interactions)
        if st.session_state.classification_results is not None:
            st.subheader("Classification Results")
            cols = st.columns(4)

            for idx, model_name in enumerate(["N-grams", "Embeddings", "LSTM", "Transformer"]):
                with cols[idx]:
                    result = st.session_state.classification_results[model_name]
                    time_ms = result["time_ms"]

                    # format time nicely
                    if time_ms < 1000:
                        time_str = f"{time_ms}ms"
                    else:
                        time_str = f"{time_ms/1000:.2f}s"

                    st.metric(
                        label=model_name,
                        value=result["prediction"],
                        delta=time_str
                    )

        # select if correct or not
        if st.session_state.classification_results is not None:
            st.header("📊 Results & Feedback")

            st.subheader("Mark Correct Predictions")
            cols = st.columns(4)

            for idx, model_name in enumerate(["N-grams", "Embeddings", "LSTM", "Transformer"]):
                with cols[idx]:
                    result = st.session_state.classification_results[model_name]
                    is_correct = st.checkbox(
                        f"{model_name} correct?",
                        key=f"correct_{model_name}",
                        help=f"Predicted: {result['prediction']}"
                    )
                    st.session_state.correctness_feedback[model_name] = is_correct

            # display final rankings
            if st.button("Submit Results", disabled=st.session_state.results_submitted):
                # check again to prevent double submission
                if not st.session_state.results_submitted:
                    # mark as submitted immediately
                    st.session_state.results_submitted = True

                    # calculate rankings for saving
                    ranked_results = calculate_rankings(
                        st.session_state.classification_results,
                        st.session_state.correctness_feedback
                    )

                    # save results
                    save_data = {
                        "filename": st.session_state.filename,
                        "results": {
                            model["name"]: {
                                "prediction": model["prediction"],
                                "correct": model["correct"],
                                "time_ms": model["time_ms"],
                                "rank": model["rank"]
                            }
                            for model in ranked_results
                        }
                    }
                    save_results(save_data)
                    st.success("✅ Results saved to demo/results/classification_history.jsonl")

            # show rankings if already submitted
            if st.session_state.results_submitted:
                ranked_results = calculate_rankings(
                    st.session_state.classification_results,
                    st.session_state.correctness_feedback
                )

                st.header("🏆 Final Rankings")

                # display results table
                for result in ranked_results:
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣"][result["rank"] - 1]
                    correctness_emoji = "✅" if result["correct"] else "❌"

                    st.markdown(
                        f"{rank_emoji} **Rank {result['rank']}** - {result['name']} | "
                        f"Prediction: {result['prediction']} {correctness_emoji} | "
                        f"Time: {result['time_ms']}ms"
                    )

            # reset button (always show after classification)
            if st.session_state.classification_results is not None:
                if st.button("🔄 Test Another Document"):
                    reset_session()
                    st.rerun()


# tab 2: view results of all tests
def show_results_tab():
    st.header("📊 Results Analytics")
    
    # load results
    results_history = load_results_history()
    
    if not results_history:
        st.info("📭 No test results yet. Go to the 'Test Models' tab to run some tests!")
        return
    
    # summary stats
    stats = get_summary_stats(results_history)
    
    st.subheader(f"Summary ({stats['total_tests']} tests)")
    
    # model comparison metrics
    cols = st.columns(4)
    for idx, model_name in enumerate(["N-grams", "Embeddings", "LSTM", "Transformer"]):
        with cols[idx]:
            model_stats = stats["models"][model_name]
            st.metric(
                label=model_name,
                value=f"{model_stats['total_points']} pts",
                delta=f"Avg rank: {model_stats['avg_rank']:.2f}"
            )
            st.caption(f"🏆 Wins: {model_stats['wins']} ({model_stats['win_rate']:.1f}%)")
            st.caption(f"✅ Accuracy: {model_stats['accuracy']:.1f}%")
            st.caption(f"🥈 Top-2: {model_stats['top2_rate']:.1f}%")
            st.caption(f"⏱️ Avg time: {model_stats['avg_time_ms']:.0f}ms")
    
    st.divider()
    
    # Test history table
    st.subheader("Test History")
    
    # Convert to DataFrame
    history_data = []
    for result in reversed(results_history[-20:]):  # Last 20 tests
        timestamp = result.get("timestamp", "Unknown")
        filename = result.get("filename", "Unknown")
        
        for model_name in ["N-grams", "Embeddings", "LSTM", "Transformer"]:
            model_data = result["results"].get(model_name, {})
            history_data.append({
                "Timestamp": timestamp[:19],  # Remove milliseconds
                "File": filename,
                "Model": model_name,
                "Prediction": model_data.get("prediction", "N/A"),
                "Correct": "✅" if model_data.get("correct") else "❌",
                "Time (ms)": model_data.get("time_ms", 0),
                "Rank": model_data.get("rank", "-")
            })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    # download option
    st.download_button(
        label="📥 Download Results as CSV",
        data=df.to_csv(index=False),
        file_name="model_comparison_results.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()