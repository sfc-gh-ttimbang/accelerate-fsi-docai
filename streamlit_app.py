import streamlit as st
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.session import Session # For type hinting
import os
import tempfile
import json
import shutil

# --- Configuration ---
# Stage for user uploads. Assumed to be in the current DB/Schema, or a fully qualified name.
# This stage is also where the Document AI UDF is expected to read from for single file processing.
DEFAULT_UPLOAD_STAGE = "DOCS"
# Fully qualified name of the Document AI Model Function (UDF).
DOC_AI_MODEL_NAME = "DEMO_FSI_ACCELERATE.BANK_STATEMENT_DOCAI.BANK_STATEMENT_DOCAI"
# The specific stage (literal string for SQL) that the Document AI UDF's PREDICT function
# is configured to use with GET_PRESIGNED_URL and for batch processing with DIRECTORY table function.
DOC_AI_UDF_TARGET_STAGE_SQL = "@DEMO_FSI_ACCELERATE.BANK_STATEMENT_DOCAI.DOCS"

# --- Snowpark Session Initialization ---
session: Session | None = None
session_init_error: Exception | None = None
try:
    session = get_active_session()
except Exception as e:
    session_init_error = e
# Further checks and UI messages for session status are handled in the UI section.

# --- Snowpark Helper Functions ---

def list_files_in_stage(current_session: Session, stage_name: str) -> list[str]:
    """Lists files in a specified Snowflake stage."""
    if not current_session:
        st.error("No active Snowpark session.")
        return []
    try:
        # Stage name for LS command should be @-prefixed if not already.
        # User input is expected without '@', so we add it.
        formatted_stage_name = f"@{stage_name.lstrip('@')}"
        rows = current_session.sql(f"LS {formatted_stage_name}").collect()
        # "name" column from LS usually contains the full path including the stage name itself.
        return [row["name"] for row in rows]
    except SnowparkSQLException as e:
        st.error(f"Error listing files from stage '{formatted_stage_name}': {e}")
        if "does not exist or not authorized" in str(e).lower():
            st.warning(f"Ensure stage '{formatted_stage_name}' exists and role has USAGE privilege.")
        return []
    except Exception as e:
        st.error(f"Unexpected error listing files: {e}")
        return []

def upload_file_to_stage(current_session: Session, local_file_path: str, target_stage_name_no_at: str) -> bool:
    """Uploads a local file to a Snowflake stage directory."""
    if not current_session:
        st.error("No active Snowpark session.")
        return False

    # target_stage_name_no_at is like "DOCS" or "MYDB.MYSCHEMA.DOCS"
    # For session.file.put, stage_location is the path on stage, e.g., @DOCS
    stage_location_for_put = f"@{target_stage_name_no_at.lstrip('@')}"
    remote_filename = os.path.basename(local_file_path)

    st.info(f"Uploading '{local_file_path}' as '{remote_filename}' to '{stage_location_for_put}'...")
    try:
        current_session.file.put(
            local_file_path,
            stage_location_for_put,
            auto_compress=False,
            overwrite=True
        )
        st.success(f"Successfully uploaded to '{stage_location_for_put}/{remote_filename}'.")
        return True
    except SnowparkSQLException as e:
        st.error(f"Error uploading to stage '{stage_location_for_put}': {e}")
        return False
    except Exception as e:
        st.error(f"Unexpected error during upload: {e}")
        return False

def run_doc_ai_on_file(current_session: Session, file_path_in_udf_target_stage: str, doc_ai_model: str) -> str | None:
    """
    Runs Document AI on a file located in the DOC_AI_UDF_TARGET_STAGE_SQL.
    'file_path_in_udf_target_stage' is the relative path of the file within DOC_AI_UDF_TARGET_STAGE_SQL.
    """
    if not current_session:
        st.error("No active Snowpark session.")
        return None

    st.info(f"Running Document AI ('{doc_ai_model}') on '{DOC_AI_UDF_TARGET_STAGE_SQL}/{file_path_in_udf_target_stage}'...")
    try:
        # Ensure file path is correctly escaped for SQL.
        escaped_file_path = file_path_in_udf_target_stage.replace("'", "''")
        # The GET_PRESIGNED_URL is hardcoded to use DOC_AI_UDF_TARGET_STAGE_SQL.
        sql_query = f"""
        SELECT {doc_ai_model}!PREDICT(
            GET_PRESIGNED_URL({DOC_AI_UDF_TARGET_STAGE_SQL}, '{escaped_file_path}'),
            1
        );"""
        result_rows = current_session.sql(sql_query).collect()

        if result_rows and len(result_rows) > 0 and result_rows[0][0] is not None:
            st.success(f"Document AI processed for '{file_path_in_udf_target_stage}'.")
            return result_rows[0][0]
        else:
            st.warning(f"Document AI returned no result or NULL for '{file_path_in_udf_target_stage}'.")
            return None
    except SnowparkSQLException as e:
        st.error(f"Snowpark SQL Error running Document AI on '{file_path_in_udf_target_stage}': {e}")
        # Simplified error hints
        if "invalid identifier" in str(e).lower() or "no such function" in str(e).lower():
             st.warning(f"Check Document AI model '{doc_ai_model}' naming, deployment, and permissions.")
        elif "failed to cast variant value" in str(e).lower() or "Error processing file" in str(e).lower():
             st.warning(f"File '{file_path_in_udf_target_stage}' in '{DOC_AI_UDF_TARGET_STAGE_SQL}' might be unsupported or corrupted.")
        elif "File not found" in str(e).lower(): # Check for file not found by GET_PRESIGNED_URL
             st.warning(f"File '{file_path_in_udf_target_stage}' not found in stage '{DOC_AI_UDF_TARGET_STAGE_SQL}'. Check path and stage contents.")
        return None
    except Exception as e:
        st.error(f"Unexpected error during Document AI processing: {e}")
        return None

def run_doc_ai_batch_on_udf_target_stage(current_session: Session, doc_ai_model: str) -> list[dict]:
    """
    Runs Document AI on all files in the predefined DOC_AI_UDF_TARGET_STAGE_SQL.
    Returns a list of dictionaries with 'file' and 'prediction'.
    """
    if not current_session:
        st.error("No active Snowpark session.")
        return []

    st.info(f"Starting batch Document AI ('{doc_ai_model}') on all files in stage '{DOC_AI_UDF_TARGET_STAGE_SQL}'...")
    try:
        # Both DIRECTORY and GET_PRESIGNED_URL use the same predefined target stage.
        sql_query = f"""
        SELECT
            RELATIVE_PATH,
            {doc_ai_model}!PREDICT(
                GET_PRESIGNED_URL({DOC_AI_UDF_TARGET_STAGE_SQL}, RELATIVE_PATH),
                1
            ) AS PREDICTION
        FROM DIRECTORY({DOC_AI_UDF_TARGET_STAGE_SQL});
        """
        result_rows = current_session.sql(sql_query).collect()

        batch_results = []
        if result_rows:
            for row in result_rows:
                batch_results.append({"file": row["RELATIVE_PATH"], "prediction": row["PREDICTION"]})
            st.success(f"Batch Document AI completed for {len(batch_results)} file(s) in '{DOC_AI_UDF_TARGET_STAGE_SQL}'.")
        else:
            st.info(f"No files found or processed in stage '{DOC_AI_UDF_TARGET_STAGE_SQL}'.")
        return batch_results
    except SnowparkSQLException as e:
        st.error(f"Snowpark SQL Error during batch Document AI on stage '{DOC_AI_UDF_TARGET_STAGE_SQL}': {e}")
        if "Directory not found" in str(e) or "does not exist" in str(e):
            st.warning(f"Ensure stage '{DOC_AI_UDF_TARGET_STAGE_SQL}' and its directory table are correctly set up.")
        return []
    except Exception as e:
        st.error(f"Unexpected error during batch Document AI: {e}")
        return []

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Snowflake Document AI Processor")
st.title("❄️ Snowflake Document AI: Automated Document Processor")
st.markdown(f"Using Document AI Model: **BANK_STATEMENT_DOCAI**")

# --- Snowpark Session Info & Initialization Check ---
st.sidebar.header("❄️ Snowpark Session")
if session:
    session_details_getters = {
        "Account": session.get_current_account,
        "Role": session.get_current_role,
        "Warehouse": session.get_current_warehouse,
        "Database": session.get_current_database,
        "Schema": session.get_current_schema,
    }
    for detail_name, getter_func in session_details_getters.items():
        try:
            value = getter_func()
            value_str = str(value).strip('"') if value is not None else "N/A"
            status_func = st.sidebar.success if detail_name == "Account" else st.sidebar.info
            status_func(f"{detail_name}: {value_str}")
        except Exception:
            st.sidebar.warning(f"Could not retrieve {detail_name.lower()}.")
else:
    error_msg = "Failed to establish Snowpark session. Ensure app runs in a Snowflake Native App environment."
    if session_init_error:
        error_msg += f"\nDetails: {type(session_init_error).__name__}: {str(session_init_error)}"
    st.sidebar.error("Snowpark session not available.")
    st.error(error_msg)
    st.stop() # Critical to stop if no session

# --- Main App Logic ---
st.header("📁 Name of Document Repository")
# Input for listing files from any stage (user's choice, not necessarily the DocAI target stage)
stage_name_for_file_listing = st.text_input(
    "Enter Snowflake Stage Name to List Files (e.g., MY_STAGE or MY_DB.MY_SCHEMA.MY_STAGE)",
    value=st.session_state.get('stage_name_for_file_listing', DEFAULT_UPLOAD_STAGE.split('.')[-1]), # Default to simple name
    help="Do not include '@'. This is for browsing files only."
).lstrip('@')
st.session_state.stage_name_for_file_listing = stage_name_for_file_listing

col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader(f"📄 Files in Repository:")
    if st.button("Refresh File List", key="refresh_files"):
        if not stage_name_for_file_listing:
            st.warning("Please enter a stage name for listing.")
        else:
            with st.spinner(f"Listing files from '@{stage_name_for_file_listing}'..."):
                st.session_state.files_in_stage_listing = list_files_in_stage(session, stage_name_for_file_listing)

    if 'files_in_stage_listing' in st.session_state:
        files_to_display = st.session_state.files_in_stage_listing
        if files_to_display:
            pdf_files = [f for f in files_to_display if str(f).lower().endswith(".pdf")]
            other_items = [f for f in files_to_display if not str(f).lower().endswith(".pdf")]

            if pdf_files:
                st.write(f"PDF files found ({len(pdf_files)}):")
                with st.container(height=200):
                    for f_path in pdf_files:
                        # Attempt to show relative path if stage name is a prefix
                        display_f_path = f_path
                        if stage_name_for_file_listing and f_path.startswith(stage_name_for_file_listing + '/'):
                            display_f_path = f_path[len(stage_name_for_file_listing)+1:]
                        st.markdown(f"- `{display_f_path}`")
            else:
                st.info("No PDF files found in this view.")

            if other_items:
                with st.expander(f"Show {len(other_items)} non-PDF items (e.g., folders)"):
                    for item_path in other_items:
                        display_item_path = item_path
                        if stage_name_for_file_listing and item_path.startswith(stage_name_for_file_listing + '/'):
                            display_item_path = item_path[len(stage_name_for_file_listing)+1:]
                        st.markdown(f"- `{display_item_path}`")
        elif files_to_display is not None: # Empty list means no files
             st.info(f"No files found in stage '@{stage_name_for_file_listing}'.")
    else:
        st.caption("Click 'Refresh File List' to see files.")


with col2:
    st.subheader(f"📤 Upload PDF for DocAI Processing")
    st.markdown(f"Files will be uploaded to **`@{DEFAULT_UPLOAD_STAGE}`**")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")
    
    single_upload_messages_container = st.container()

    if uploaded_file:
        with single_upload_messages_container:
            st.write(f"Selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")

        if st.button(f"Upload and Run Document AI", key="upload_and_run_single"):
            single_upload_messages_container.empty() # Clear previous messages
            temp_dir = None
            local_file_path = None
            try:
                with single_upload_messages_container, st.spinner(f"Processing '{uploaded_file.name}'..."):
                    temp_dir = tempfile.mkdtemp()
                    local_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(local_file_path, "wb") as tmp_f:
                        tmp_f.write(uploaded_file.getvalue())

                    # Upload to DEFAULT_UPLOAD_STAGE (e.g., "DOCS")
                    # This stage name should resolve to where DOC_AI_UDF_TARGET_STAGE_SQL points.
                    upload_success = upload_file_to_stage(session, local_file_path, DEFAULT_UPLOAD_STAGE)

                    if upload_success:
                        st.markdown("---")
                        # File path for DocAI is just the filename, as it's in the root of DOC_AI_UDF_TARGET_STAGE_SQL
                        # (assuming DEFAULT_UPLOAD_STAGE maps to the root of DOC_AI_UDF_TARGET_STAGE_SQL)
                        doc_ai_result = run_doc_ai_on_file(session, uploaded_file.name, DOC_AI_MODEL_NAME)
                        if doc_ai_result is not None:
                            st.subheader(f"📄 Document AI Result for '{uploaded_file.name}':")
                            try:
                                st.json(json.loads(doc_ai_result), expanded=True)
                            except (json.JSONDecodeError, TypeError):
                                st.text_area("Raw Result", str(doc_ai_result), height=200)
                        # If DEFAULT_UPLOAD_STAGE is the same as what stage_name_for_file_listing is set to,
                        # user might expect to see it. A manual refresh is better than auto-rerun.
                        if stage_name_for_file_listing.endswith(DEFAULT_UPLOAD_STAGE): # Simple check
                             st.caption("To see this uploaded file in the list on the left, please click 'Refresh File List'.")
            finally:
                if local_file_path and os.path.exists(local_file_path):
                    try: os.remove(local_file_path)
                    except OSError as e: single_upload_messages_container.warning(f"Could not remove temp file: {e}")
                if temp_dir and os.path.exists(temp_dir):
                    try: shutil.rmtree(temp_dir)
                    except OSError as e: single_upload_messages_container.warning(f"Could not remove temp dir: {e}")
    else:
        single_upload_messages_container.empty()


st.divider()
st.header("🚀 Batch Processing Files in Repository")
st.markdown(f"""
This will run the Document AI model on all PDF files found within the repository.""")

if st.button(f"Process ALL Files", key="batch_run_main_stage", type="primary"):
    batch_results_container = st.container()
    with batch_results_container, st.spinner(f"Initiating batch processing with Document AI"):
        batch_doc_ai_results = run_doc_ai_batch_on_udf_target_stage(session, DOC_AI_MODEL_NAME)

    if batch_doc_ai_results:
        with batch_results_container: # Display results in the same container
            st.subheader(f"Document AI Extraction Results")
            
            results_for_df = []
            for item in batch_doc_ai_results:
                file_p = item.get("file", "N/A")
                pred_output = item.get("prediction", "")
                pred_preview = str(pred_output)
                if len(pred_preview) > 300: # Shorten for table display
                    pred_preview = pred_preview[:300] + "..."
                results_for_df.append({
                    "File Path": file_p,
                    "Prediction Preview": pred_preview,
                    "Full Prediction": str(pred_output) # Keep full for expander
                })

            st.dataframe(
                [{"File Path": r["File Path"], "Prediction Preview": r["Prediction Preview"]} for r in results_for_df],
                use_container_width=True,
                hide_index=True
            )
            with st.expander("View Full Predictions", expanded=False):
                for item_detail in results_for_df:
                    st.markdown(f"**File:** `{item_detail['File Path']}`")
                    try:
                        # Try to pretty-print if JSON
                        parsed_json = json.loads(item_detail['Full Prediction'])
                        st.json(parsed_json, expanded=False)
                    except (json.JSONDecodeError, TypeError):
                        st.text_area(f"Raw Output", item_detail['Full Prediction'], height=150, key=f"batch_res_{item_detail['File Path']}")
                    st.markdown("---")
    elif batch_doc_ai_results is not None: # Empty list means no files or no results
        with batch_results_container:
            st.info(f"No results from batch processing on '{DOC_AI_UDF_TARGET_STAGE_SQL}'. Check logs if files were expected.")

st.sidebar.markdown("---")
st.sidebar.caption("Ensure role privileges: USAGE on WAREHOUSE, DATABASE, SCHEMA; USAGE, READ, WRITE on STAGE(s); USAGE on Document AI Model.")
