import boto3
import pandas as pd
from io import StringIO
from typing import Dict, Optional
from rich import print as rprint
from .s3_handler import ConfigHandler


class S3IOHandler:
    def __init__(self, config_handler: ConfigHandler, verbose: bool = True):
        """Initialize S3 IO handler with config."""
        self.config = config_handler
        self.verbose = verbose
        self.bucket_name = self.config.s3_bucket
        self.s3_client = self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client using config credentials."""
        credentials = self.config.get_aws_credentials()
        return boto3.client(
            's3',
            endpoint_url=credentials['aws_endpoint_url'],
            aws_access_key_id=credentials['aws_access_key_id'],
            aws_secret_access_key=credentials['aws_secret_access_key']
        )

    def load_data(self, file_key: str) -> pd.DataFrame:
        """Load DataFrame from S3."""
        if self.verbose:
            rprint(f"[blue]Loading data from S3: {file_key}[/blue]")
        
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
        return pd.read_csv(obj['Body'])

    def save_data(self, df: pd.DataFrame, file_key: str) -> None:
        """Save DataFrame to S3."""
        try:
            if self.verbose:
                rprint(f"[blue]Saving data to S3: {file_key}[/blue]")
            
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_key,
                Body=csv_buffer.getvalue()
            )
            
            if self.verbose:
                rprint(f"[bold green]Data saved to s3://{self.bucket_name}/{file_key}[/bold green]")
        except Exception as e:
            rprint(f"[red]Failed to save data to S3: {e}[/red]")