"""GCP session/credentials resolution — the one place the Google SDKs are touched for this provider.

Every GCP adapter takes an injectable client (so tests pass a fake and no adapter imports a google
library directly). This helper resolves project + location + credentials the standard way
(Application Default Credentials / env) and builds the per-service clients lazily. The google
libraries are OPTIONAL (`context-runtime[gcp]`) and imported only when a builder is actually called.
"""
from __future__ import annotations

import os


class GcpSession:
    def __init__(self, project: str | None = None, location: str = "us-central1",
                 credentials=None, api_key: str | None = None):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        self.credentials = credentials
        # api_key → the Gemini Developer API (no project/ADC needed); absent → Vertex AI (project + ADC).
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # each builder lazy-imports only its own SDK; adapters inject a client in tests and never reach here
    def genai_client(self):
        from google import genai  # context-runtime[gcp]
        if self.api_key:
            return genai.Client(api_key=self.api_key)      # Gemini Developer API
        return genai.Client(vertexai=True, project=self.project, location=self.location,
                            credentials=self.credentials)  # Vertex AI

    def discoveryengine_client(self):
        from google.cloud import discoveryengine_v1 as de
        return de.SearchServiceClient(credentials=self.credentials)

    def bigquery_client(self):
        from google.cloud import bigquery
        return bigquery.Client(project=self.project, credentials=self.credentials)

    def modelarmor_client(self):
        from google.cloud import modelarmor_v1
        return modelarmor_v1.ModelArmorClient(credentials=self.credentials)

    def monitoring_client(self):
        from google.cloud import monitoring_v3
        return monitoring_v3.MetricServiceClient(credentials=self.credentials)
