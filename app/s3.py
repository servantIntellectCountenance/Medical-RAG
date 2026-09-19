from pathlib import Path
import boto3
import streamlit as st


BUCKET_NAME = "medical-rag-xuntao"

FILES = {
    "data/medical_rag.index": "medical_rag.index",
    "data/documents.pkl": "documents.pkl",
}


@st.cache_resource
def download_rag_data():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["aws"]["access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["secret_access_key"],
        region_name=st.secrets["aws"]["region"],
    )

    for local_path, s3_key in FILES.items():
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(
            BUCKET_NAME,
            s3_key,
            str(path),
        )

    return True