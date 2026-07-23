"""AWS session/credential resolution — the one place boto3 is touched for the AWS provider.

Every AWS adapter takes an injectable client (so tests pass a fake and no adapter imports boto3
directly). This helper is the default client factory: it resolves a region + credentials the standard
way (env / shared config / instance role), optionally assumes a role, and hands back boto3 clients.
boto3 is an OPTIONAL dependency (`context-runtime[aws]`) and is imported lazily — importing this
module never requires it.
"""
from __future__ import annotations

import os
from typing import Any


class AwsSession:
    """Lazy boto3 session with optional STS role assumption. Region resolves from an explicit arg,
    then ``AWS_REGION``/``AWS_DEFAULT_REGION``, then ``us-east-1``. Pass ``role_arn`` to assume a
    least-privilege role (the demo's Vault→STS flow lives in redevops-aws-demo; here we accept a
    role to assume with whatever base credentials the environment already provides)."""

    def __init__(self, region: str | None = None, role_arn: str | None = None,
                 profile: str | None = None, session: Any = None):
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self.role_arn = role_arn
        self.profile = profile
        self._session = session      # inject a boto3.Session (or a fake) to bypass boto3 entirely
        self._assumed = None

    def _base_session(self):
        if self._session is not None:
            return self._session
        try:
            import boto3  # optional dep: context-runtime[aws]
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "the AWS provider needs boto3 — install `context-runtime[aws]`, "
                "or inject a session/client for tests"
            ) from e
        self._session = boto3.Session(profile_name=self.profile, region_name=self.region)
        return self._session

    def _effective_session(self):
        if not self.role_arn:
            return self._base_session()
        if self._assumed is not None:
            return self._assumed
        sts = self._base_session().client("sts", region_name=self.region)
        creds = sts.assume_role(RoleArn=self.role_arn, RoleSessionName="context-runtime")["Credentials"]
        import boto3
        self._assumed = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=self.region,
        )
        return self._assumed

    def client(self, service: str):
        """A boto3 client for ``service`` (e.g. 'bedrock-runtime', 'opensearchserverless', 'athena').
        Adapters call this lazily; tests inject clients directly and never reach here."""
        return self._effective_session().client(service, region_name=self.region)
